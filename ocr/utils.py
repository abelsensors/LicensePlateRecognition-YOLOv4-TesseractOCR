import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_aspect_ratio(gray_img, max_shape=1500):
    """
    Resize the image given the biggest size of any axis mantaining the aspect rate
    """
    x_shape, y_shape = gray_img.shape
    if x_shape >= y_shape:
        y_shape = int((1 - ((x_shape - max_shape) / x_shape)) * y_shape)
        x_shape = max_shape
    else:
        x_shape = int((1 - ((y_shape - max_shape) / y_shape)) * x_shape)
        y_shape = max_shape

    resized_img = cv2.resize(gray_img
                             , (y_shape, x_shape)
                             , interpolation=cv2.INTER_CUBIC)
    return resized_img


def remove_shades(image):
    """
    Remove shadowing of the lights of the image by applying normalization.
    TODO: this method needs to be improved, results seems to be in the lower side for some scenarios,
    TODO: might be interesting using it when the background extraction fails. The next paper should improve this
    TODO: scenario  https://arxiv.org/pdf/1710.05073.pdf
    """
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    plt.imshow(result)
    plt.show()
    result_norm = cv2.merge(result_norm_planes)
    plt.imshow(result_norm)
    plt.show()
    return result_norm
