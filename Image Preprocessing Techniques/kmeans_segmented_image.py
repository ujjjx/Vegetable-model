import cv2
import numpy as np
import matplotlib.pyplot as plt

def initialize_centers(pixels, k):
    np.random.seed(42)  # For reproducibility
    random_indices = np.random.choice(pixels.shape[0], k, replace=False)
    centers = pixels[random_indices]
    return centers

def assign_clusters(pixels, centers):
    distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

def recompute_centers(pixels, labels, k):
    centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
    return centers

def k_means_clustering(image_path, output_path, k=3, max_iters=100, tol=1e-4):
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError("Image not found. Please provide a valid image path.")
    
    # Convert the image from BGR to RGB
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = original_image_rgb.reshape((-1, 3))  # Reshape to (num_pixels, 3)

    # Convert to float for K-means
    pixels = np.float32(pixels)

    # Initialize cluster centers
    centers = initialize_centers(pixels, k)

    for i in range(max_iters):
        # Assign clusters
        labels = assign_clusters(pixels, centers)

        # Recompute centers
        new_centers = recompute_centers(pixels, labels, k)

        # Check for convergence
        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    # Convert labels to the original image dimensions
    segmented_image = centers[labels].reshape(original_image_rgb.shape).astype(np.uint8)

    # Save the segmented image
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    # Create a color palette for the clusters
    palette = np.uint8(centers)

    # Display the original and segmented images with color legend
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title(f"K-means Segmented Image (k={k})")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow([palette])
    plt.title("Cluster Colors")
    plt.axis("off")
    
    plt.show()

if __name__ == "__main__":
    # User input for image paths and number of clusters
    input_image = input("Enter the path to the input image: ")
    output_image = input("Enter the desired path for the output image: ")
    k = int(input("Enter the number of clusters (k): "))

    # Perform K-means clustering
    k_means_clustering(input_image, output_image, k=k)
    print(f"K-means segmented image saved as {output_image}")