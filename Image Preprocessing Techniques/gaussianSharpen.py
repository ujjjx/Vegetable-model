import numpy as np
import cv2

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for x in range(size):
        for y in range(size):
            diff = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = np.exp(-diff / (2 * sigma ** 2))
    kernel /= (2 * np.pi * sigma ** 2)
    kernel /= kernel.sum()
    return kernel

def convolve(image, kernel):
    """Apply convolution between an image and a kernel."""
    if len(image.shape) == 3:  # Colored image
        channels = image.shape[2]
        output = np.zeros_like(image)
        for c in range(channels):
            output[:, :, c] = convolve(image[:, :, c], kernel)
        return output
    else:  # Grayscale image
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the image with zeros on all sides
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Initialize the output image
        output = np.zeros_like(image)

        # Perform convolution
        for i in range(image_height):
            for j in range(image_width):
                output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

        return output

def gaussian_blur(image, sigma):
    """Apply Gaussian blur to an image."""
    # Determine the size of the kernel
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(size, sigma)
    return convolve(image, kernel)

def add_weighted(image1, weight1, image2, weight2, gamma):
    """Combine two images with specified weights."""
    return np.clip(weight1 * image1 + weight2 * image2 + gamma, 0, 255).astype(np.uint8)

def sharpen_image(image, sigma):
    """Sharpen the image by subtracting the blurred image from the original image."""
    blurred = gaussian_blur(image, sigma)
    sharpened_image = add_weighted(image, 1.5, blurred, -0.5, sigma)
    return sharpened_image

def gaussian_sharpen(input_path, output_path, sigma):
    """Read an image, apply Gaussian sharpening, and save the result."""
    image = cv2.imread(input_path)
    sharpened_image = sharpen_image(image, sigma)
    cv2.imwrite(output_path, sharpened_image)
    
    # Display the original and sharpened images
    cv2.imshow('Original Image', image)
    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get user input for image paths and filter parameters
    input_image = input("Enter the path to the input image: ")
    output_image = input("Enter the path to save the output image: ")

    # Ask user for the Gaussian sigma value
    sigma = float(input("Enter the Gaussian sigma value (e.g., 1.0): "))

    # Apply Gaussian sharpening
    gaussian_sharpen(input_image, output_image, sigma)