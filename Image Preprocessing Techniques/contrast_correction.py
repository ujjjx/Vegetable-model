import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_correction(image_path, output_path, alpha, beta):
    # Read the image in BGR format (OpenCV default)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found. Please provide a valid image path.")
    
    # Convert the image to float32 for precision
    image = image.astype(np.float32)
    
    # Perform contrast correction using the formula: corrected_pixel = alpha * original_pixel + beta
    corrected_image = alpha * image + beta
    
    # Clip the values to be in the valid range [0, 255] and convert to uint8
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    
    # Display the input and output images using Matplotlib
    plt.figure(figsize=(10, 5))

    # Input Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    # Output Image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    plt.title('Output Image')
    plt.axis('off')

    # Display the images
    plt.tight_layout()
    plt.show()

    # Save the output image
    cv2.imwrite(output_path, corrected_image)
    return output_path

if __name__ == "__main__":
    # User input for image paths and contrast parameters
    input_image = input("Enter the path to the input image: ")
    output_image = input("Enter the desired path for the output image: ")
    alpha = float(input("Enter the alpha value for contrast correction (e.g., 1.5): "))
    beta = float(input("Enter the beta value for brightness adjustment (e.g., 0): "))

    # Perform contrast correction
    contrast_correction(input_image, output_image, alpha, beta)
    print(f"Contrast corrected image saved as {output_image}")