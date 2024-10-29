import cv2
import numpy as np

def bilateral_filter(image, diameter, sigma_color, sigma_space):
    # Convert the image to float32 for precision
    image = image.astype(np.float32)
    
    # Get the dimensions of the image
    height, width, channels = image.shape
    
    # Create an empty output image
    smoothed_image = np.zeros_like(image)
    
    # Compute the spatial Gaussian kernel
    half_diameter = diameter // 2
    spatial_kernel = np.zeros((diameter, diameter), dtype=np.float32)
    for i in range(diameter):
        for j in range(diameter):
            x = i - half_diameter
            y = j - half_diameter
            spatial_kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))
    
    
    # Apply the bilateral filter
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                # Define the region of interest
                y_min = max(y - half_diameter, 0)
                y_max = min(y + half_diameter + 1, height)
                x_min = max(x - half_diameter, 0)
                x_max = min(x + half_diameter + 1, width)
                
                # Extract the region of interest
                region = image[y_min:y_max, x_min:x_max, c]
                
                # Compute the range Gaussian kernel
                intensity_diff = region - image[y, x, c]
                range_kernel = np.exp(-(intensity_diff**2) / (2 * sigma_color**2))
                
                # Combine the spatial and range kernels
                spatial_kernel_expanded = spatial_kernel[(y_min - y + half_diameter):(y_max - y + half_diameter), 
                                                         (x_min - x + half_diameter):(x_max - x + half_diameter)]
                bilateral_kernel = spatial_kernel_expanded * range_kernel

                # Normalize the bilateral kernel
                bilateral_kernel /= bilateral_kernel.sum()
                
                # Apply the bilateral filter
                smoothed_image[y, x, c] = np.sum(bilateral_kernel * region)
    
    return smoothed_image.astype(np.uint8)

# Example usage
if __name__ == "__main__":
    # Load the image
    image_path = "image1.jpg"
    original_image = cv2.imread(image_path)
    
    # Parameters for the bilateral filter
    diameter = 15
    sigma_color = 75
    sigma_space = 75
    
    # Apply the bilateral filter
    smoothed_image = bilateral_filter(original_image, diameter, sigma_color, sigma_space)
    
    # Save and display the result
    output_path = "smoothed_image.jpg"
    cv2.imwrite(output_path, smoothed_image)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Smoothed Image", smoothed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()