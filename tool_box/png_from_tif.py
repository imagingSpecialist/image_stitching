import os
import numpy as np
from PIL import Image


def normalize_and_convert(source_folder):
    # Setup target directory
    target_folder = os.path.join(source_folder, 'png')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            try:
                img_path = os.path.join(source_folder, filename)

                # 1. Read the image data into a NumPy array
                with Image.open(img_path) as img:
                    data = np.array(img).astype(float)

                # 2. Normalize the data (data / max * 255)
                # We check for max > 0 to avoid division by zero
                data_max = np.max(data)
                if data_max > 0:
                    normalized_data = (data / data_max) * 255
                else:
                    normalized_data = data

                # 3. Convert back to 8-bit unsigned integer (required for standard PNG)
                final_image = Image.fromarray(normalized_data.astype(np.uint8))

                # Save the result
                base_name = os.path.splitext(filename)[0]
                final_image.save(os.path.join(target_folder, f"{base_name}.png"))
                print(f"Normalized and saved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Run the function
normalize_and_convert('data/Phase_Image_Tiles')