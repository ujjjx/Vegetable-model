import cv2
import numpy as np

def equalize_histogram(channel):
    # Calculate histogram
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize the CDF to the range [0, 255]
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Fill the masked values with 0
    cdf = np.ma.filled(cdf, 0).astype('uint8')
    
    # Map the input pixel values to equalized values
    equalized_channel = cdf[channel]
    
    return equalized_channel

try:
    # Load the image
    original_image = cv2.imread('image1.jpg')
    if original_image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to the grayscale image
    equalized_image = equalize_histogram(gray_image)

    # Save or display the equalized grayscale image
    cv2.imwrite('equalized_image.jpg', equalized_image)
    cv2.imshow('Equalized Grayscale Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")