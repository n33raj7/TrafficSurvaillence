import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from collections import deque

os.environ["YOLO_TRACK_MAX_AGE"] = "50"
os.environ["YOLO_TRACK_MAX_DIST"] = "0.9"

VIDEO_PATH = "footage/intersection.mp4"
MODEL_PATH = "weights/best.pt"
CONF_THRES = 0.15
PERSIST = True


TARGET_W = 1280
TARGET_H = 720
TARGET_PROCESS_FPS = 15 

# Vehicle classes selected by you
VEHICLE_CLASSES = [1, 2, 4, 6, 7, 9]

# Class weights (keep keyed by original class IDs)
CLASS_WEIGHTS = {
    1: 3,    # bus
    2: 2,    # car
    4: 1,    # motorcycle
    6: 4,    # truck
    7: 2,    # van
    9: 1,    # Motor (bike)
}

# Merge map: classes that should display as "vehicle"
MERGE_TO_VEHICLE = {cid: "vehicle" for cid in VEHICLE_CLASSES}

MIN_GREEN = 10   # seconds (min per single side)
MAX_GREEN = 75   # seconds (max per single side)
CYCLE_BUFFER = 2 # seconds between phases (approx yellow)
VIOLATION_SIDE = "N"
REGION_COORDS = {
    "N": [(640, 150), (680, 150), (880, 520), (600, 520)],
    "S": [(600, 720), (930, 720), (930, 720), (600, 720)],
    "E": [(900, 600), (1280, 600), (1280, 720), (930, 720)],
    "W": [(0, 480), (320, 510), (150, 720), (0, 720)],
}
    
STOP_LINES = {
    "N": ( (570, 520), (880, 520) ),  
    "S": ( (600, 720), (930, 720) ),   
    "E": ( (900, 600), (930, 720) ),
    "W": ( (320, 510), (150, 720) ),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Smart Intersection Demo (ESC to quit)"
VIOLATION_DIR = "violations"
os.makedirs(VIOLATION_DIR, exist_ok=True)

COUNTING_WINDOW_SECONDS = 10

# Phase order (you set this)
PHASE_ORDER = ["W", "N", "E", "S"]  

def letterbox(image, new_shape=(TARGET_W, TARGET_H), color=(114,114,114)):
    """Resize and pad to new_shape (width,height) while preserving aspect ratio."""
    h0, w0 = image.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))
    resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h
    left = dw // 2
    top = dh // 2
    result = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=color)
    return result

def point_in_polygon(pt, polygon):
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, pt, False) >= 0

def detection_weight(cls):
    return CLASS_WEIGHTS.get(int(cls), 1)

def compute_side_times(counts, min_t=MIN_GREEN, max_t=MAX_GREEN):
    """Return dict side->seconds proportional to counts, clipped to [min_t, max_t]."""
    total = sum(counts.values())
    if total == 0:
        equal = int(max(min_t, (min_t + max_t) // 4))
        return {s: equal for s in counts.keys()}
    budget = max_t
    times = {}
    for s, v in counts.items():
        times[s] = int(np.clip((v / total) * budget, min_t, max_t))
    return times

print("[INFO] Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"[INFO] Source video FPS: {video_fps:.2f}")
process_interval = 1.0 / TARGET_PROCESS_FPS
print(f"[INFO] Process up to {TARGET_PROCESS_FPS} FPS -> interval {process_interval:.4f}s")

# runtime structures
last_processed_time = 0.0
frame_idx = 0
results = None
active_tracks = {}           # track_id -> (side, cls)
side_windows = {s: deque() for s in REGION_COORDS.keys()}  # timestamped weighted values
track_last_bottom = {}       # track_id -> last bottom y
violations_logged = set()

# signal state: only one side green at a time
signal_state = {s: "RED" for s in REGION_COORDS.keys()}
phase_idx = 0
current_side = PHASE_ORDER[phase_idx]
def set_phase_single(side_key):
    for s in signal_state.keys():
        signal_state[s] = "GREEN" if s == side_key else "RED"

set_phase_single(current_side)
phase_start_ts = time.time()
# initial per-side durations (will be recomputed)
side_durations = {s: MIN_GREEN for s in PHASE_ORDER}
next_phase_duration = side_durations[current_side]

print("[INFO] Starting main loop. Press ESC to stop.")
# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached.")
        break

    frame_idx += 1

    # fixed processing resolution (letterbox)
    proc_frame = letterbox(frame, (TARGET_W, TARGET_H))
    ph, pw = proc_frame.shape[:2]
    vis = proc_frame.copy()

    now = time.time()
    do_process = (now - last_processed_time) >= process_interval

    if do_process:
        last_processed_time = now
        # detection + tracking
        results = model.track(proc_frame, conf=CONF_THRES, persist=PERSIST, verbose=False)
        active_tracks.clear()

        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None:
                for box in res.boxes:
                    cls = int(box.cls[0])
                    # Only consider classes that are part of VEHICLE_CLASSES
                    if cls not in VEHICLE_CLASSES:
                        continue
                    tid = int(box.id[0]) if box.id is not None else None
                    if tid is None:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, y2

                    # determine which side region bottom-center belongs to
                    detected_side = None
                    for side_key, poly in REGION_COORDS.items():
                        if point_in_polygon((cx, cy), poly):
                            detected_side = side_key
                            break

                    active_tracks[tid] = (detected_side, cls)

                    # --- North-only red-light violation (top -> bottom crossing) ---
                    if detected_side == VIOLATION_SIDE:
                        prev_bottom = track_last_bottom.get(tid, None)
                        stop_y = int(round((STOP_LINES[VIOLATION_SIDE][0][1] + STOP_LINES[VIOLATION_SIDE][1][1]) / 2))
                        # top -> bottom crossing: prev < stop_y and curr >= stop_y
                        if prev_bottom is not None and prev_bottom < stop_y and cy >= stop_y:
                            # capture only if North signal is RED
                            if signal_state.get(VIOLATION_SIDE, "RED") == "RED" and tid not in violations_logged:
                                x1c, y1c, x2c, y2c = max(0,x1), max(0,y1), min(pw,x2), min(ph,y2)
                                crop = proc_frame[y1c:y2c, x1c:x2c]
                                ts = int(time.time())
                                fname = os.path.join(VIOLATION_DIR, f"violation_N_tid{tid}_{ts}.jpg")
                                cv2.imwrite(fname, crop)
                                violations_logged.add(tid)
                                print(f"[VIOLATION] North RED captured: {fname}")
                        track_last_bottom[tid] = cy
                    else:
                        # update last bottom for other tracks too (keeps tracking consistent)
                        track_last_bottom[tid] = cy

    # Update sliding window counts using active_tracks (may be empty if not processed this loop)
    weighted = {s: 0.0 for s in REGION_COORDS.keys()}
    for tid, (side, cls) in active_tracks.items():
        if side in weighted:
            weighted[side] += detection_weight(cls)

    ts_now = time.time()
    for s, val in weighted.items():
        side_windows[s].append((ts_now, val))
        while side_windows[s] and (ts_now - side_windows[s][0][0] > COUNTING_WINDOW_SECONDS):
            side_windows[s].popleft()

    window_sums = {s: int(sum(v for _, v in dq)) for s, dq in side_windows.items()}

    # Phase management: single-side rotation
    elapsed_phase = now - phase_start_ts
    if elapsed_phase >= (next_phase_duration + CYCLE_BUFFER):
        # compute per-side durations (proportional)
        side_durations = compute_side_times(window_sums, MIN_GREEN, MAX_GREEN)
        # rotate to next side
        phase_idx = (phase_idx + 1) % len(PHASE_ORDER)
        current_side = PHASE_ORDER[phase_idx]
        set_phase_single(current_side)
        next_phase_duration = int(np.clip(side_durations.get(current_side, MIN_GREEN), MIN_GREEN, MAX_GREEN))
        phase_start_ts = now
        print(f"[SIGNAL] New green side: {current_side} for {next_phase_duration}s  (window_sums={window_sums})")

    # Visualization: draw regions and counts
    for side_key, poly in REGION_COORDS.items():
        poly_np = np.array(poly, dtype=np.int32)
        cv2.polylines(vis, [poly_np], True, (180,180,180), 2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [poly_np], (50,50,50))
        cv2.addWeighted(overlay, 0.05, vis, 0.95, 0, vis)
        cnt = int(window_sums.get(side_key, 0))
        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = poly[0]
        cv2.putText(vis, f"{side_key}: {cnt}", (cx-30, cy), FONT, 0.8, (255,255,255), 2)

    # draw stop-lines
    for side_key, (p1, p2) in STOP_LINES.items():
        cv2.line(vis, p1, p2, (0,0,255), 2)

    # draw traffic lights (single green)
    for side_key, poly in REGION_COORDS.items():
        poly_np = np.array(poly, dtype=np.int32)
        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = poly[0]
        box_tl = (cx - 50, cy - 60)
        box_br = (cx - 20, cy - 30)
        color = (0,0,255)
        if signal_state.get(side_key) == "GREEN":
            color = (0,255,0)
        cv2.rectangle(vis, box_tl, box_br, color, -1)
        if signal_state.get(side_key) == "GREEN":
            rem = int(max(0, next_phase_duration - (time.time() - phase_start_ts)))
            cv2.putText(vis, f"{side_key} GREEN {rem}s", (box_tl[0], box_tl[1]-8), FONT, 0.6, (255,255,255), 2)
        else:
            cv2.putText(vis, f"{side_key} {signal_state.get(side_key)}", (box_tl[0], box_tl[1]-8), FONT, 0.6, (255,255,255), 2)

    # draw last detection boxes (if processed this loop)
    if do_process and results and len(results) > 0:
        res = results[0]
        if res.boxes is not None:
            for box in res.boxes:
                cls = int(box.cls[0])
                tid = int(box.id[0]) if box.id is not None else None
                if tid is None or cls not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0,255,0)
                if tid in violations_logged:
                    color = (0,0,255)
                cv2.rectangle(vis, (x1,y1),(x2,y2), color, 2)

                # DISPLAY MERGED LABEL: show "vehicle" for merged vehicle classes
                if cls in MERGE_TO_VEHICLE:
                    display_label = MERGE_TO_VEHICLE[cls]
                else:
                    display_label = model.names[cls] if hasattr(model, "names") else str(cls)

                cv2.putText(vis, f"{display_label} ID:{tid}", (x1, y1-6), FONT, 0.5, color, 2)

    # dashboard
    cv2.rectangle(vis, (10,10), (420,160), (20,20,20), -1)
    cv2.putText(vis, "Dashboard", (20,30), FONT, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"E: {window_sums.get('E',0)}", (20,60), FONT, 0.6, (255,255,255), 2)
    cv2.putText(vis, f"N: {window_sums.get('N',0)}", (140,60), FONT, 0.6, (255,255,255), 2)
    cv2.putText(vis, f"W: {window_sums.get('W',0)}", (20,90), FONT, 0.6, (255,255,255), 2)
    cv2.putText(vis, f"S: {window_sums.get('S',0)}", (140,90), FONT, 0.6, (255,255,255), 2)

    remaining = int(max(0, next_phase_duration - (now - phase_start_ts)))
    cv2.putText(vis, f"GREEN: {current_side}", (20,120), FONT, 0.6, (255,255,255), 2)
    cv2.putText(vis, f"Next in: {remaining}s", (220,120), FONT, 0.6, (255,255,255), 2)

    cv2.imshow(WINDOW_NAME, vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Demo stopped.")
