import cv2
import sys

# Function to perform image denoising
def denoise_image(input_image_path, output_image_path, h=10, hForColor=10, templateWindowSize=7, searchWindowSize=21):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Could not read the image.")
        sys.exit()

    # Perform denoising using Fast Non-Local Means Denoising for colored images
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hForColor, templateWindowSize, searchWindowSize)

    # Save the denoised image
    cv2.imwrite(output_image_path, denoised_image)

    # Display the original and denoised images
    cv2.imshow("Original Image", image)
    cv2.imshow("Denoised Image", denoised_image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get user input for image paths
    input_image = input("Enter the path to the input image : ")
    output_image = input("Enter the path to save the output image : ")

    # Optional user input for denoising parameters (with defaults)
    h = float(input("Enter the strength of noise reduction for luminance (default 10): ") or 10)
    hForColor = float(input("Enter the strength of noise reduction for color components (default 10): ") or 10)
    templateWindowSize = int(input("Enter the template window size (default 7): ") or 7)
    searchWindowSize = int(input("Enter the search window size (default 21): ") or 21)

    # Apply image denoising with user-defined parameters
    denoise_image(input_image, output_image, h, hForColor, templateWindowSize, searchWindowSize)
    print(f"Denoised image saved as {output_image}")
