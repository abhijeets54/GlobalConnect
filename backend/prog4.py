from flask import Flask, send_file, request, jsonify
from PIL import Image, ImageEnhance
import io

app = Flask(__name__)

# Load the INSAT image (assuming it's stored as 'insat.jpg')
IMAGE_PATH = 'insat.jpg'

def adjust_contrast(image, contrast_factor):
    """Adjust the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)

def adjust_brightness(image, brightness_factor):
    """Adjust the brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

@app.route('/get_image')
def get_image():
    """Endpoint to return the original INSAT image."""
    try:
        # Open the INSAT image
        img = Image.open(IMAGE_PATH)
        
        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/adjust_contrast')
def contrast_adjustment():
    """Endpoint to return the contrast adjusted image."""
    try:
        # Get the contrast factor from request args (default is 1.0 - no change)
        contrast_factor = float(request.args.get('factor', 1.0))

        # Open the INSAT image
        img = Image.open(IMAGE_PATH)

        # Adjust the contrast
        img = adjust_contrast(img, contrast_factor)

        # Save the adjusted image to a BytesIO object
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/adjust_brightness')
def brightness_adjustment():
    """Endpoint to return the brightness adjusted image."""
    try:
        # Get the brightness factor from request args (default is 1.0 - no change)
        brightness_factor = float(request.args.get('factor', 1.0))

        # Open the INSAT image
        img = Image.open(IMAGE_PATH)

        # Adjust the brightness
        img = adjust_brightness(img, brightness_factor)

        # Save the adjusted image to a BytesIO object
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
