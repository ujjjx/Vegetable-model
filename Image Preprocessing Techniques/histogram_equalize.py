import cv2
import numpy as np

def equalize_histogram(channel):

    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    cdf = np.ma.filled(cdf, 0).astype('uint8')
    
    equalized_channel = cdf[channel]
    
    return equalized_channel

try:

    original_image = cv2.imread('image1.jpg')
    if original_image is None:
        raise ValueError("Image not found or unable to load.")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    equalized_image = equalize_histogram(gray_image)

    cv2.imwrite('equalized_image.jpg', equalized_image)
    cv2.imshow('Equalized Grayscale Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")
