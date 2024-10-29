import tensorflow as tf
import cv2
import numpy as np
import os

# Bradley Local Thresholding Function
def bradley_local_thresholding(gray_image, window_size=30, threshold=10):
    # Calculate the integral image
    integral_image = cv2.integral(gray_image)
    height, width = gray_image.shape

    # Apply Bradley Local Thresholding
    thresholded_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            x1, y1, x2, y2 = max(0, x - window_size // 2), max(0, y - window_size // 2), min(width - 1, x + window_size // 2), min(height - 1, y + window_size // 2)
            area = (x2 - x1) * (y2 - y1)
            threshold_sum = integral_image[y2, x2] - integral_image[y1, x2] - integral_image[y2, x1] + integral_image[y1, x1]
            if gray_image[y, x] * area <= threshold_sum * (100 - threshold) / 100:
                thresholded_image[y, x] = 0
            else:
                thresholded_image[y, x] = 255

    return thresholded_image

# Load the dataset
IMAGE_SIZE = 256
BATCH_SIZE = 32 
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Create output directory if it doesn't exist
output_dir = "ThresholdedImages"
os.makedirs(output_dir, exist_ok=True)

# Process each batch of images
for batch, labels in dataset:
    for i in range(batch.shape[0]):
        # Convert the image to numpy array and then to grayscale
        image = batch[i].numpy().astype(np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Bradley Local Thresholding
        thresholded_image = bradley_local_thresholding(gray_image)

        # Save the thresholded image
        output_path = os.path.join(output_dir, f"thresholded_image_{i}.png")
        cv2.imwrite(output_path, thresholded_image)

print(f"Thresholded images saved in {output_dir}")