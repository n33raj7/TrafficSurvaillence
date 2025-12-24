import os
import time
import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO

VIOLATION_FOLDER = r"ocr\images"
EXCEL_FILE = "violations_log.xlsx"
PLATE_MODEL_PATH = r"C:\Users\rajpu\yolohome\runs\detect\train\weights\best.pt" 
OCR_LANG = ['en']   # OCR language
CONF_THRESH = 0.3


print("[INFO] Loading YOLO model for plate detection...")
model = YOLO(PLATE_MODEL_PATH)

print("[INFO] Loading OCR engine...")
ocr = easyocr.Reader(OCR_LANG)

if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
else:
    df = pd.DataFrame(columns=["Image", "Plate Number", "Timestamp", "Side"])


def extract_side_from_filename(filename):
    """Extracts side marker like _N_, _S_, _E_, _W_ from filename if present."""
    for s in ["N", "S", "E", "W"]:
        if f"_{s}_" in filename:
            return s
    return "UNKNOWN"


# ocr
files = [f for f in os.listdir(VIOLATION_FOLDER) if f.lower().endswith((".jpg", ".png"))]

print(f"[INFO] Found {len(files)} violation images.")

for file in files:
    if file in df["Image"].values:
        print(f"[SKIPPED] Already logged → {file}")
        continue

    img_path = os.path.join(VIOLATION_FOLDER, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[ERROR] Cannot read {file}")
        continue

    results = model.predict(img, conf=CONF_THRESH, verbose=False)

    plate_text = "NOT FOUND"

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate_crop = img[y1:y2, x1:x2]

            # Run
            ocr_result = ocr.readtext(plate_crop)

            if len(ocr_result) > 0:
                plate_text = "".join([r[1] for r in ocr_result]).replace(" ", "").upper()
                print(f"[OCR] {file} → {plate_text}")
                break
    else:
        print(f"[NO PLATE DETECTED] {file}")

    #  dataframe
    df.loc[len(df)] = [
        file,
        plate_text,
        time.strftime('%Y-%m-%d %H:%M:%S'),
        extract_side_from_filename(file)
    ]

# Excel
df.to_excel(EXCEL_FILE, index=False)
print(f"\n[SUCCESS] Logged results → {EXCEL_FILE}")
