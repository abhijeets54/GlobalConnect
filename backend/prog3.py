from PIL import Image
import io
from flask import Flask, send_file, jsonify

app = Flask(__name__)

# Image file path (update this to your actual image path)
image_path = 'insat.jpg'

# Predefined regions (pixel coordinates for simplicity)
demo_regions = {
    "north": {
        "lon_start": 0,    # Pixel coordinates of the top-left corner
        "lat_start": 0,    # Pixel coordinates of the top-left corner
        "lon_end": 800,    # Pixel coordinates of the bottom-right corner (width)
        "lat_end": 500     # Pixel coordinates of the bottom-right corner (height)
    },
    "south": {
        "lon_start": 800,
        "lat_start": 500,
        "lon_end": 1600,
        "lat_end": 1200
    },
    "central": {
        "lon_start": 400,
        "lat_start": 300,
        "lon_end": 1200,
        "lat_end": 800
    }
}

def get_image_region(region_name):
    """
    Function to extract a specific region from the image based on predefined coordinates.
    :param region_name: Name of the region to be cropped
    :return: BytesIO object of cropped image if successful, else None
    """
    region_meta = demo_regions.get(region_name)
    
    if region_meta:
        try:
            # Open the image from the specified path
            image = Image.open(image_path)
            
            # Crop the image based on the region's pixel coordinates
            cropped_image = image.crop((
                int(region_meta['lon_start']),  # left
                int(region_meta['lat_start']),  # upper
                int(region_meta['lon_end']),    # right
                int(region_meta['lat_end'])     # lower
            ))

            # Save cropped image to a buffer in-memory
            img_io = io.BytesIO()
            cropped_image.save(img_io, 'JPEG')
            img_io.seek(0)

            return img_io
        except Exception as e:
            print(f"Error cropping image: {e}")
            return None
    else:
        print(f"Region {region_name} not found in predefined data.")
        return None

@app.route('/get_region/<region_name>', methods=['GET'])
def get_region(region_name):
    """
    Flask route to serve a specific image region as a downloadable JPEG file.
    :param region_name: Name of the region (e.g., 'north', 'south', etc.)
    :return: Cropped image region or error message
    """
    # Get the cropped image region
    image_data = get_image_region(region_name)
    if image_data:
        return send_file(image_data, mimetype='image/jpg', as_attachment=True, download_name=f'{region_name}_region.jpg')
    else:
        return jsonify({"error": "Region not found or error processing image."}), 404

if __name__ == '__main__':
    app.run(debug=True)
