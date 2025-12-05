import os
import glob
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

TARGET_CLASS_ID = 39
RAW_DIR = "raw_png"
OUT_DIR_PNG = "processed_png"

os.makedirs(OUT_DIR_PNG, exist_ok=True)

yolo = YOLO("yolov8m.pt")
sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
sam.to("cpu")
predictor = SamPredictor(sam)

def process_image(img_path):
    name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    H, W = img_np.shape[:2]
    img_center = np.array([W / 2, H / 2])

    results = yolo(img_np, conf=0.01, verbose=False)[0]
    bottle_boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        if cls == TARGET_CLASS_ID:
            bottle_boxes.append([x1, y1, x2, y2])
    if not bottle_boxes:
        print(f"Brak butelki w: {img_path}")
        return

    best_box = min(bottle_boxes, key=lambda b: np.linalg.norm(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) - img_center))
    predictor.set_image(img_np)
    x1, y1, x2, y2 = best_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    point_coords = np.array([[cx, cy]])
    point_labels = np.ones(len(point_coords))
    # Poprawka: box jako numpy array
    box_np = np.array([x1, y1, x2, y2])
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_np,
        multimask_output=False
    )
    mask_255 = masks[0].astype(np.uint8) * 255

    # PNG z przezroczystością
    img_rgba = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
    img_rgba[:, :, 3] = mask_255
    out_path_png = os.path.join(OUT_DIR_PNG, f"{name_no_ext}_processed.png")
    cv2.imwrite(out_path_png, img_rgba)
    print(f"PNG zapisany: {out_path_png}")

def main():
    files = glob.glob(os.path.join(RAW_DIR, "*.png"))
    if not files:
        print(f"Brak plików PNG w '{RAW_DIR}'.")
        return
    for f in files:
        process_image(f)
    print("Gotowe.")

if __name__ == "__main__":
    main()