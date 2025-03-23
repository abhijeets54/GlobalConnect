from PIL import Image
import io
from flask import Flask, send_file, jsonify, render_template, request, redirect, url_for
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded images

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

def get_image_region(image_path, region_name):
    """
    Function to extract a specific region from the uploaded image based on predefined coordinates.
    :param image_path: Path to the uploaded image
    :param region_name: Name of the region to be cropped
    :return: BytesIO object of cropped image if successful, else None
    """
    region_meta = demo_regions.get(region_name)
    
    if region_meta:
        try:
            # Open the uploaded image from the specified path
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

@app.route('/')
def index():
    """
    Serve the HTML page with a form to upload an image and select a region.
    """
    return render_template('index1.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Handle the image upload and selection of the region.
    """
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Get the selected region from the form
        region_name = request.form.get('region')
        
        # Redirect to display the selected region with download option
        return redirect(url_for('show_region', region_name=region_name, image_name=file.filename))

@app.route('/show_region/<region_name>/<image_name>', methods=['GET'])
def show_region(region_name, image_name):
    """
    Render a page that shows the selected region and allows the user to download it.
    """
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    
    # Get the cropped image region
    image_data = get_image_region(image_path, region_name)
    
    if image_data:
        return render_template('show_region.html', region_name=region_name, image_name=image_name)
    else:
        return jsonify({"error": "Region not found or error processing image."}), 404

@app.route('/download/<region_name>/<image_name>', methods=['GET'])
def download_region(region_name, image_name):
    """
    Flask route to serve a specific image region as a downloadable JPEG file.
    :param region_name: Name of the region (e.g., 'north', 'south', etc.)
    :param image_name: Name of the uploaded image file
    :return: Cropped image region or error message
    """
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    
    # Get the cropped image region
    image_data = get_image_region(image_path, region_name)
    if image_data:
        return send_file(image_data, mimetype='image/jpg', as_attachment=True, download_name=f'{region_name}_region.jpg')
    else:
        return jsonify({"error": "Region not found or error processing image."}), 404

if __name__ == '__main__':
    app.run(debug=True)
