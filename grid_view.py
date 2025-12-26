import cv2
import numpy as np
import os
import re
from glob import glob


def create_thumbnail_grid(folder_path, thumb_size=(200, 200)):
    pattern = re.compile(r'_r(\d+)_c(\d+)')
    files = glob(os.path.join(folder_path, "*.tif*"))

    img_map = {}
    for f in files:
        match = pattern.search(os.path.basename(f))
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            img_map[(r, c)] = f

    if not img_map:
        return None

    rows = max(k[0] for k in img_map.keys()) + 1
    cols = max(k[1] for k in img_map.keys()) + 1
    tw, th = thumb_size

    # Create a blank canvas (8-bit for easy viewing)
    # Adding 5 pixels of padding between tiles
    padding = 5
    canvas_w = cols * (tw + padding)
    canvas_h = rows * (th + padding)
    grid_image = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for (c, r), path in img_map.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Normalize to 8-bit for thumbnail visibility
        img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Resize to thumbnail size
        resized = cv2.resize(img_8, thumb_size)

        # Calculate position
        y = r * (th + padding)
        x = c * (tw + padding)
        grid_image[y:y + th, x:x + tw] = resized

    return grid_image


# Usage
thumbnail_view = create_thumbnail_grid('data/Phase_Image_Tiles')
cv2.imwrite('grid_view_before.png', thumbnail_view)