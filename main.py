import cv2
import numpy as np
import os
import re
from glob import glob


def get_micro_shift(img1, img2):
    """Calculates the small adjustment shift between two images."""
    # Convert to float32 for FFT
    im1 = img1.astype(np.float32)
    im2 = img2.astype(np.float32)

    # Create Hanning window to ignore edge artifacts
    win = cv2.createHanningWindow((im1.shape[1], im1.shape[0]), cv2.CV_32F)

    # Phase Correlation
    (dx, dy), response = cv2.phaseCorrelate(im1 * win, im2 * win)

    # If the response is very low, the shift is unreliable; return 0,0
    if response < 0.01:
        return 0, 0
    return int(round(dx)), int(round(dy))


def stitch_tiff_grid(folder_path, overlap_px=150):
    """
    Stitches TIFF tiles using a hybrid theoretical + phase correlation approach.
    overlap_px: Approximate overlap in pixels between adjacent tiles.
    """
    # 1. Gather files and parse row/col
    pattern = re.compile(r'_r(\d+)_c(\d+)')
    files = glob(os.path.join(folder_path, "*.tif*"))

    img_map = {}
    for f in files:
        match = pattern.search(os.path.basename(f))
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            img_map[(r, c)] = f

    if not img_map:
        raise ValueError("No images found with the _rXXX_cXXX format.")

    all_keys = sorted(img_map.keys())
    start_r, start_c = all_keys[0]

    # Load first image to get dimensions
    first_img = cv2.imread(img_map[(start_r, start_c)], cv2.IMREAD_UNCHANGED)
    h, w = first_img.shape[:2]
    print(h,w)
    # Theoretical distance between tile origins
    step_x = w - overlap_px
    step_y = h - overlap_px

    # Dictionary to store calculated global (y, x) coordinates
    coords = {(start_r, start_c): (0, 0)}

    print("Calculating grid coordinates with micro-shifts...")
    for (r, c) in all_keys:
        if (r, c) == (start_r, start_c):
            continue

        # Find a neighbor already in the 'coords' list to calculate current position
        # Prioritize left neighbor, then top neighbor
        ref_pos = None
        if (r, c - 1) in coords:
            ref_r, ref_c = r, c - 1
            ideal_y, ideal_x = coords[(ref_r, ref_c)][0], coords[(ref_r, ref_c)][1] + step_x
        elif (r - 1, c) in coords:
            ref_r, ref_c = r - 1, c
            ideal_y, ideal_x = coords[(ref_r, ref_c)][0] + step_y, coords[(ref_r, ref_c)][1]
        else:
            continue  # Should not happen in a standard grid

        # Detect the micro-shift compared to the neighbor
        img_ref = cv2.imread(img_map[(ref_r, ref_c)], cv2.IMREAD_UNCHANGED)
        img_curr = cv2.imread(img_map[(r, c)], cv2.IMREAD_UNCHANGED)


        dx, dy = get_micro_shift(img_ref, img_curr)

        # New global coordinate = Theoretical Step + Phase Correlation Correction
        coords[(r, c)] = (ideal_y + dy, ideal_x + dx)

    # 2. Build Canvas
    all_y, all_x = zip(*coords.values())
    min_y, min_x = min(all_y), min(all_x)
    max_y, max_x = max(all_y), max(all_x)

    canvas_h = (max_y - min_y) + h
    canvas_w = (max_x - min_x) + w

    print(f"Final canvas size: {canvas_w}x{canvas_h}")
    canvas = np.zeros((canvas_h, canvas_w), dtype=first_img.dtype)

    # 3. Place Images onto Canvas
    for (r, c), (y, x) in coords.items():
        img = cv2.imread(img_map[(r, c)], cv2.IMREAD_UNCHANGED)

        # Calculate local canvas position
        start_y = y - min_y
        start_x = x - min_x

        # Slice the canvas and place the image
        canvas[start_y: start_y + h, start_x: start_x + w] = img

    return canvas


import os
import re
import cv2
import numpy as np
from glob import glob


def concact_tiff_grid(folder_path):
    """
    Simply concatenates TIFF tiles into a grid based on row/col indices.
    No overlaps or micro-shifts are calculated.
    """
    # 1. Gather files and parse row/col
    pattern = re.compile(r'_r(\d+)_c(\d+)')
    files = glob(os.path.join(folder_path, "*.tif*"))

    img_map = {}
    max_r, max_c = 0, 0

    for f in files:
        match = pattern.search(os.path.basename(f))
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            img_map[(r, c)] = f
            max_r = max(max_r, r)
            max_c = max(max_c, c)

    if not img_map:
        raise ValueError("No images found with the _rXXX_cXXX format.")

    # 2. Get dimensions from the first image
    first_key = list(img_map.keys())[0]
    sample_img = cv2.imread(img_map[first_key], cv2.IMREAD_UNCHANGED)
    h, w = sample_img.shape[:2]
    dtype = sample_img.dtype

    # 3. Create Canvas
    # Total size = (number of rows * height) x (number of columns * width)
    # We use max_r + 1 because indices usually start at 0
    num_rows = max_r + 1
    num_cols = max_c + 1

    canvas_h = num_rows * h
    canvas_w = num_cols * w

    print(f"Creating canvas of size {canvas_w}x{canvas_h} for a {num_rows}x{num_cols} grid.")
    canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)

    # 4. Place Images (Concatenate)
    for (r, c), path in img_map.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Calculate top-left corner for this tile
        start_y = r * h
        start_x = c * w

        canvas[start_y: start_y + h, start_x: start_x + w] = img

    return canvas

# --- Execution Block ---
data_seq = 1
data_path = f'data/Phase_Image_Tiles_Trial_{data_seq}'
try:
    # Adjust overlap_px based on your actual microscope settings
    stitched_result = stitch_tiff_grid(data_path, overlap_px=200)
    concat_result = concact_tiff_grid(data_path)

    # Save as 16-bit TIFF
    # cv2.imwrite('final_stitch_raw_main.tif', stitched_result)

    # Normalize to 8-bit for PNG viewing
    normalized = cv2.normalize(stitched_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fileName = f'{data_path}/final_stitch_trial_{data_seq}.png'
    cv2.imwrite(fileName, normalized)

    normalized = cv2.normalize(concat_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fileName_c = f'{data_path}/final_concat_trial_{data_seq}.png'
    cv2.imwrite(fileName_c, normalized)

    print(f'Stitching complete. {fileName}saved.')
except Exception as e:
    print(f"Failed: {e}")