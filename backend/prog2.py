import rasterio
from rasterio.transform import from_origin
from PIL import Image
import numpy as np
import os

# Input and output paths
input_folder = r'c:\Users\user\Desktop\internal\data'  # Folder containing the images
output_folder = r'c:\Users\user\Desktop\internal\dataset\output_folder'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 1: Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # Process only JPEG images
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_compressed.tif")  # Set output name

        # Step 2: Open the JPEG image using PIL
        with Image.open(input_file) as img:
            img_data = np.array(img)  # Convert image to numpy array

        # Step 3: Define the transform and metadata (adjust based on your needs)
        transform = from_origin(0, 0, 1, 1)  # dummy transform: (top-left x, top-left y, pixel width, pixel height)

        # Define metadata for GeoTIFF
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',  # Assuming 8-bit JPEG
            'count': img_data.shape[2] if len(img_data.shape) == 3 else 1,  # Number of bands (3 for RGB, 1 for grayscale)
            'height': img_data.shape[0],
            'width': img_data.shape[1],
            'transform': transform,
            'compress': 'LZW',  # Compression method
            'tiled': True,      # Enable tiling
            'blockxsize': 256,  # Tile width
            'blockysize': 256   # Tile height
        }

        # Step 4: Write the data to a compressed GeoTIFF file
        with rasterio.open(output_file, 'w', **profile) as dst:
            if len(img_data.shape) == 3:  # For RGB images
                for i in range(1, img_data.shape[2] + 1):
                    dst.write(img_data[:, :, i - 1], i)
            else:  # For grayscale images
                dst.write(img_data, 1)

        print(f"Compressed GeoTIFF saved at {output_file}")

print("Processing complete.")
