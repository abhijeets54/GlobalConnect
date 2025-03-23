import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_flood(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        logging.error("Image not found!")
        return
    
    logging.info("Image loaded successfully.")
    
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for water (you may need to fine-tune this)
    lower_blue = np.array([90, 50, 50])  # Lower bound for blue color
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue color

    # Create a mask for water-like areas in the image
    water_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Calculate the percentage of flood-like pixels (pixels in the mask)
    flood_pixels = np.sum(water_mask == 255)
    total_pixels = water_mask.size
    flood_percentage = (flood_pixels / total_pixels) * 100

    # Set a threshold for what constitutes a flood
    flood_threshold = 10.0  # If 10% or more of the image shows potential flood areas

    # Log and print if the image shows floods or not
    if flood_percentage > flood_threshold:
        logging.info(f"Flood detected! ({flood_percentage:.2f}% of the image shows flood-like areas)")
        print(f"Flood detected! ({flood_percentage:.2f}% of the image shows flood-like areas)")
    else:
        logging.info(f"No flood detected. ({flood_percentage:.2f}% of the image shows flood-like areas)")
        print(f"No flood detected. ({flood_percentage:.2f}% of the image shows flood-like areas)")


def detect_cyclone(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        logging.error("Image not found!")
        return
    
    logging.info("Image loaded successfully for cyclone detection.")
    
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for clouds (grayish/white areas)
    lower_gray = np.array([0, 0, 200])  # Lower bound for white/grayish clouds
    upper_gray = np.array([180, 50, 255])  # Upper bound for white/grayish clouds

    # Create a mask for cyclone-like cloud areas
    cloud_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # Calculate the percentage of cyclone-like pixels (cloud pixels in the mask)
    cyclone_pixels = np.sum(cloud_mask == 255)
    total_pixels = cloud_mask.size
    cyclone_percentage = (cyclone_pixels / total_pixels) * 100

    # Set a threshold for what constitutes a cyclone
    cyclone_threshold = 15.0  # If 15% or more of the image shows potential cyclone areas

    # Log and print if the image shows cyclones or not
    if cyclone_percentage > cyclone_threshold:
        logging.info(f"Cyclone detected! ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)")
        print(f"Cyclone detected! ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)")
    else:
        logging.info(f"No cyclone detected. ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)")
        print(f"No cyclone detected. ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)")


def detect_forest_fire(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        logging.error("Image not found!")
        return
    
    logging.info("Image loaded successfully for forest fire detection.")
    
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for fire (reddish/orange areas)
    lower_red = np.array([0, 50, 50])  # Lower bound for red/orange
    upper_red = np.array([20, 255, 255])  # Upper bound for red/orange

    # Create a mask for fire-like areas in the image
    fire_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Calculate the percentage of fire-like pixels (fire pixels in the mask)
    fire_pixels = np.sum(fire_mask == 255)
    total_pixels = fire_mask.size
    fire_percentage = (fire_pixels / total_pixels) * 100

    # Set a threshold for what constitutes a fire
    fire_threshold = 5.0  # If 5% or more of the image shows potential fire areas

    # Log and print if the image shows fires or not
    if fire_percentage > fire_threshold:
        logging.info(f"Forest fire detected! ({fire_percentage:.2f}% of the image shows fire-like areas)")
        print(f"Forest fire detected! ({fire_percentage:.2f}% of the image shows fire-like areas)")
    else:
        logging.info(f"No forest fire detected. ({fire_percentage:.2f}% of the image shows fire-like areas)")
        print(f"No forest fire detected. ({fire_percentage:.2f}% of the image shows fire-like areas)")

# Example usage
image_path = 'fire.jpeg'  # Replace with your image file path
detect_flood(image_path)
detect_cyclone(image_path)
detect_forest_fire(image_path)