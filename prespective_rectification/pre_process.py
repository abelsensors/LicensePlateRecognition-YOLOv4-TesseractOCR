import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label


def get_largest_connected_contours(segmentation_area):
    """
    Select the biggest region of the image that correspond to a connected contour
    """
    labels = label(1 - segmentation_area)
    assert (labels.max() != 0)  # assume at least 1 connected contour
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc * 255


def pre_process(image_read):
    """
    Pipeline to process and clean the image to a bynarized contour
    """
    # Convert to greyscale
    gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)

    # Adaptative Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    plt.imshow(equalized)
    plt.show()

    # Binaritzation
    ret, thresh_image = cv2.threshold(equalized, 120, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh_image)
    plt.show()

    # Clean small objects in the mask
    thresh = cv2.erode(thresh_image, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Use bitwise and with white mask to clean other areas
    square_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    plt.imshow(thresh)
    plt.show()

    # Get biggest contours area
    thresh = get_largest_connected_contours(thresh)
    thresh = thresh.astype('uint8')
    plt.imshow(thresh)
    plt.show()
    kernel = np.ones((5, 5), np.uint8)
    corners = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    plt.imshow(corners)
    plt.show()
    return thresh, corners


def get_masked_image(image, thresh):
    """
    Used to gather only the area of the mask where we want to work
    """
    result = cv2.bitwise_and(image, image, mask=thresh)
    return result