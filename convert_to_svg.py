import os
import glob
import cv2
import numpy as np
import svgwrite
from scipy.ndimage import uniform_filter1d

INPUT_DIR = "processed_png"
OUTPUT_DIR = "processed_svg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_detailed_svg(mask, svg_path):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Brak konturów w {svg_path}")
        return False

    h, w = mask.shape
    dwg = svgwrite.Drawing(svg_path, size=(w, h))

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        epsilon = 0.005 * cv2.arcLength(contour, True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(smooth_contour) < 0.8 * len(contour):
            pts = np.squeeze(contour)
        else:
            pts = np.squeeze(smooth_contour)
        if pts.ndim != 2:
            continue
        pts_x = uniform_filter1d(pts[:, 0], size=5)
        pts_y = uniform_filter1d(pts[:, 1], size=5)
        points = [(int(x), int(y)) for x, y in zip(pts_x, pts_y)]
        path_data = "M " + " L ".join([f"{x},{y}" for x, y in points]) + " Z"
        dwg.add(dwg.path(d=path_data, fill='none', stroke='black', stroke_width=2))

    dwg.save()
    return True

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*_processed.png"))
    if not files:
        print("Brak plików wejściowych.")
        return
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            print(f"Niepoprawny PNG: {f}")
            continue
        mask = img[:, :, 3]
        svg_path = os.path.join(OUTPUT_DIR, os.path.basename(f).replace("_processed.png", "_detailed.svg"))
        if mask_to_detailed_svg(mask, svg_path):
            print(f"SVG zapisany: {svg_path}")
        else:
            print(f"SVG nie wygenerowany: {f}")

if __name__ == "__main__":
    main()