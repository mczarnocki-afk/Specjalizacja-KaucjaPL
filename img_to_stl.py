import cv2
import numpy as np
import cadquery as cq
import matplotlib.pyplot as plt

desired_height_mm = 359
desired_diameter_mm = 103

input_image = cv2.imread("butelka2.jpg")
if input_image is None:
    raise FileNotFoundError("Nie znaleziono pliku")
image_height, image_width, _ = input_image.shape

gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

foreground_mask = cv2.inRange(gray_image, 0, 230)

bottle_radii_px = []
profile_y_px = []

for y in range(image_height):
    mask_row = foreground_mask[y, :]
    foreground_xs = np.where(mask_row > 0)[0]
    if len(foreground_xs) > 1:
        radius_px = (foreground_xs[-1] - foreground_xs[0]) / 2.0
        bottle_radii_px.append(radius_px)
        profile_y_px.append(y)

if not bottle_radii_px:
    raise ValueError("Nie znaleziono żadnych promieni na obrazie.")

bottle_radii_px = np.array(bottle_radii_px, dtype=float)
profile_y_px = np.array(profile_y_px, dtype=float)

original_height = profile_y_px[-1] - profile_y_px[0]
original_diameter = np.max(bottle_radii_px) * 2.0

height_scale = desired_height_mm / original_height
diameter_scale = desired_diameter_mm / original_diameter

profile_points = [
    (radius * diameter_scale, (y - profile_y_px[0]) * height_scale)
    for radius, y in zip(bottle_radii_px, profile_y_px)
]

closed_profile = [(0, 0)] + profile_points + [(0, desired_height_mm)]

profile_array = np.array(closed_profile)
plt.figure(figsize=(5,9))
plt.plot(profile_array[:,0], profile_array[:,1], '-b')
plt.title("Profil butelki (po skalowaniu)")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()

profile_sketch = cq.Workplane("XY").polyline(closed_profile).close()
bottle_solid = profile_sketch.revolve(360)

cq.exporters.export(bottle_solid, "butelka_scaled2.stl")