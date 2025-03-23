import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys

def enhance_contrast_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split channels
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)  # Adjusted CLAHE parameters
    cl = clahe.apply(l)  # Apply CLAHE to the L-channel
    limg = cv2.merge((cl, a, b))  # Merge channels
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # Convert back to BGR
    
    # Optional smoothing
    enhanced_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)  # Apply mild Gaussian blur
    return enhanced_img

def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, enhanced):
    ssim_value, _ = ssim(original, enhanced, full=True, channel_axis=2, win_size=3)
    return ssim_value

def calculate_brightness_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    brightness = np.mean(l_channel)
    contrast = np.std(l_channel)
    return brightness, contrast

def plot_histograms(original, enhanced):
    original_l = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)[:, :, 0]
    enhanced_l = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)[:, :, 0]

    plt.figure()
    plt.hist(original_l.ravel(), bins=256, color='blue', alpha=0.5, label='Original L Channel')
    plt.hist(enhanced_l.ravel(), bins=256, color='red', alpha=0.5, label='Enhanced L Channel')
    plt.legend()
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('L Channel Histogram Comparison')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('tortoise.jpg')
    if img is None:
        print("Error: Could not read the image.")
        sys.exit(1)

    enhanced_img = enhance_contrast_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))

    # Calculate metrics
    psnr_value = calculate_psnr(img, enhanced_img)
    ssim_value = calculate_ssim(img, enhanced_img)
    original_brightness, original_contrast = calculate_brightness_contrast(img)
    enhanced_brightness, enhanced_contrast = calculate_brightness_contrast(enhanced_img)

    # Display images
    cv2.imshow("Before", img)
    cv2.imshow("After (Enhanced)", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print results
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"Brightness gain: {enhanced_brightness - original_brightness:.2f}")
    print(f"Contrast gain: {enhanced_contrast - original_contrast:.2f}")

    # Show histograms for visual comparison
    plot_histograms(img, enhanced_img)
