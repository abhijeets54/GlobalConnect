#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# Path to the image where data will be embedded
path = "tortoise.jpg"

global shi
global exp
shi = 0
exp = 0

# Function to calculate PSNR
def PSNR(arr1, arr2):
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return np.inf
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

# Function to calculate normalized correlation coefficient
def normalized_correlation_coefficient(image1, image2):
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    std1 = np.std(image1)
    std2 = np.std(image2)
    correlation = np.sum((image1 - mean1) * (image2 - mean2)) / (image1.size * std1 * std2)
    return correlation

# Function to calculate SSIM
def ssim(img1, img2, dynamic_range=255.0):
    mu_x = np.mean(img1)
    mu_y = np.mean(img2)
    sigma_x_sq = np.var(img1)
    sigma_y_sq = np.var(img2)
    sigma_xy = np.cov(img1.flatten(), img2.flatten())[0, 1]

    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq) + C2) / (sigma_x_sq + sigma_y_sq + C2)
    s = (sigma_xy + C2 / 2) / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq) + C2 / 2)
    ssim_index = l * c * s

    return ssim_index

# Function to find the optimal delta and peak
def find_delta(img, min_val, max_val):
    H = img.astype(dtype=np.float64)
    h, w = img.shape
    mx_capacity = 0

    for delta in range(min_val, max_val):
        map = np.zeros(w, dtype=int)
        flag = np.zeros(w, dtype=int)
        map[0] = 0  # Initialize with zero-based index
        flag[0] = 1

        for i in range(1, len(map)):
            map[i] = (map[i - 1] + delta) % w  # Adjust index using modulo
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

# Function to perform histogram shifting
def histogram_shifting(arr, peak):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] > peak:
                arr[i][j] += 1
    return arr

# Function to embed data into the image
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

# Function to generate the transformed embedded image
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

# Function to construct the final image from the map
def construct_image(transformed_with_embedded, map):
    sorted_columns = np.argsort(map)
    marked_image = transformed_with_embedded[:, sorted_columns]
    return marked_image

# Main part of the script
img = Image.open(path)
grayscale_image = img.convert("L")
image = np.array(grayscale_image)

# Embedding data
data = "1010100000010101010101"
for i in range(len(data), 1000):  # Extend data to required capacity
    data += "1"

# Find delta, capacity, map, peak, and difference image
mx_delta, mx_capacity, mx_map, mx_peak, difference_image = find_delta(image, 1, image.shape[1])

# Total pixel count for calculating relative capacity
total_pixels = image.shape[0] * image.shape[1]
relative_capacity = mx_capacity / total_pixels

# Transform and embed the data
transformed_image = image[:, mx_map]
inter_diff_image = histogram_shifting(np.copy(difference_image), mx_peak)
marked_diff_image = embedding(np.copy(inter_diff_image), mx_peak, data)

# Embed data into transformed image
transformed_with_embedded = generate_transformed_embedded(transformed_image, marked_diff_image, embedd=True)

# Construct marked image
marked_image = construct_image(transformed_with_embedded, mx_map)

# Check if data was successfully embedded
def is_embedding_successful(original_image, marked_image, threshold=30):
    psnr_value = PSNR(original_image, marked_image)
    return psnr_value > threshold, psnr_value

# Check the embedding and return True/False, PSNR, and embedding capacity
embedding_successful, psnr_value = is_embedding_successful(image, marked_image)

# Convert the marked image to PIL format for saving or displaying
marked_image_pil = Image.fromarray(marked_image.astype(np.uint8))

# Return the results
print(f"Embedding Successful: {embedding_successful}")
print(f"PSNR: {psnr_value:.2f} dB")
print(f"Relative Embedding Capacity: {relative_capacity:.6f} (fraction of total pixels)")

# Save the embedded image if needed
marked_image_pil.save("embedded_image.png")
