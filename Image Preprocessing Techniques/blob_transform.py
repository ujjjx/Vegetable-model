import tensorflow as tf
import cv2
import numpy as np
import os

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
output_dir = "BlobDetectedImages"
os.makedirs(output_dir, exist_ok=True)

# Set up the SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.maxArea = 5000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

def blob_detection(image, params):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(gray)

    # Draw detected blobs as red circles on the original image
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints

# Process each batch of images
for batch, labels in dataset:
    for i in range(batch.shape[0]):
        # Convert the image to numpy array
        image = batch[i].numpy().astype(np.uint8)

        # Apply blob detection
        im_with_keypoints = blob_detection(image, params)

        # Save the result
        output_path = os.path.join(output_dir, f"blob_detected_image_{i}.png")
        cv2.imwrite(output_path, im_with_keypoints)

print(f"Blob detection results saved in {output_dir}")