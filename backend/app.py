from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

global shi
global exp
shi = 0
exp = 0

def PSNR(arr1, arr2):
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return np.inf
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

def find_delta(img, min_val, max_val):
    H = img.astype(dtype=np.float64)
    h, w = img.shape
    mx_capacity = 0

    for delta in range(min_val, max_val):
        map = np.zeros(w, dtype=int)
        flag = np.zeros(w, dtype=int)
        map[0] = 0  
        flag[0] = 1

        for i in range(1, len(map)):
            map[i] = (map[i - 1] + delta) % w  
            if flag[map[i]] == 1:
                map[i] = (map[i] + 1) % w
            flag[map[i]] = 1

        diff = np.zeros((h, w // 2), dtype=np.float64)
        for i in range(w // 2):
            x = 2 * i
            y = x + 1
            diff[:, i] = H[:, map[x]] - H[:, map[y]]

        u_val, counts = np.unique(diff, return_counts=True)
        if len(u_val) == 1:
            peak = u_val[0]
            capacity = w * h // 2
        else:
            ind = np.argmax(counts)
            capacity = counts[ind]
            peak = u_val[ind]

        if capacity > mx_capacity:
            mx_capacity = capacity
            mx_delta = delta
            mx_map = map
            mx_peak = peak
            diff_img = diff

    return mx_delta, mx_capacity, mx_map, mx_peak, diff_img

def histogram_shifting(arr, peak):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] > peak:
                arr[i][j] += 1
    return arr

def embedding(arr, peak, data):
    ind = 0
    for i in range(len(arr)):
        if ind == len(data):
            break
        for j in range(len(arr[0])):
            if arr[i][j] == peak:
                if data[ind] == '1':
                    arr[i][j] += 1
                    ind += 1
                    if ind == len(data):
                        break
                else:
                    ind += 1
                    if ind == len(data):
                        break
    return arr

def generate_transformed_embedded(transformed_image, embedded_diff_image, embedd=False):
    n = transformed_image.shape[1]
    half_n = n // 2
    transformed_with_embedded = np.copy(transformed_image)
    for i in range(half_n):
        transformed_with_embedded[:, 2 * i] = transformed_image[:, 2 * i + 1] + embedded_diff_image[:, i]
        if embedd:
            global shi
            shi += embedded_diff_image.shape[0]
    return transformed_with_embedded

def construct_image(transformed_with_embedded, map):
    sorted_columns = np.argsort(map)
    marked_image = transformed_with_embedded[:, sorted_columns]
    return marked_image

@app.route('/')
def home():
    return render_template('index.html')  # HTML form for upload

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', output={"embedding_successful": False, "psnr_value": "Error", "relative_capacity": "Error"})

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', output={"embedding_successful": False, "psnr_value": "Error", "relative_capacity": "Error"})

    secret_code = request.form.get('secret_code', '')

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load the image and perform embedding
    img = Image.open(file_path)
    grayscale_image = img.convert("L")
    image = np.array(grayscale_image)

    # Find delta, capacity, map, peak, and difference image
    mx_delta, mx_capacity, mx_map, mx_peak, difference_image = find_delta(image, 1, image.shape[1])

    # Transform and embed the data
    transformed_image = image[:, mx_map]
    inter_diff_image = histogram_shifting(np.copy(difference_image), mx_peak)
    marked_diff_image = embedding(np.copy(inter_diff_image), mx_peak, secret_code)

    # Embed data into transformed image
    transformed_with_embedded = generate_transformed_embedded(transformed_image, marked_diff_image, embedd=True)

    # Construct marked image
    marked_image = construct_image(transformed_with_embedded, mx_map)

    def is_embedding_successful(original_image, marked_image, threshold=30):
        psnr_value = PSNR(original_image, marked_image)
        return psnr_value > threshold, psnr_value

    embedding_successful, psnr_value = is_embedding_successful(image, marked_image)

    marked_image_pil = Image.fromarray(marked_image.astype(np.uint8))
    
    output_path = os.path.join(UPLOAD_FOLDER, "embedded_image.png")
    marked_image_pil.save(output_path)

    total_pixels = image.shape[0] * image.shape[1]
    relative_capacity = mx_capacity / total_pixels

    # Prepare output data for rendering in HTML (without sending the image path)
    output_data = {
        "embedding_successful": embedding_successful,
        "psnr_value": psnr_value,
        "relative_capacity": relative_capacity
    }

    return render_template('index.html', output=output_data)


if __name__ == '__main__':
    app.run(debug=True)
