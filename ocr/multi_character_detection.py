import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import clear_border


class PyImageSearchANPR:
    def __init__(self, debug=False):
        self.debug = debug

    @staticmethod
    def debug_imshow(title, image):
        plt.imshow(image)
        plt.title(title)
        plt.show()

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a blackhat morphological operation that will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)

        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)
        self.debug_imshow("Blackhat", blackhat)
        plt.imshow(blackhat)
        plt.title("Blackhat")
        plt.show()
        # next, find regions in the image that are light
        square_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kern)
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        self.debug_imshow("Light Regions", light)
        # compute the Scharr gradient representation of the blackhat
        # image in the x-direction and then scale the result back to
        # the range [0, 255]
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - minVal) / (maxVal - minVal))
        grad_x = grad_x.astype("uint8")
        self.debug_imshow("Scharr", grad_x)
        # blur the gradient representation, applying a closing
        # operation, and threshold the image using Otsu's method
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kern)
        thresh = cv2.threshold(grad_x, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)
        # perform a series of erosions and dilations to clean up the
        # thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)
        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, waitKey=True)
        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest
        # ones
        plt.imshow(thresh)
        plt.show()

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates, clear_border_flag=False):
        # initialize the license plate contour and ROI
        lp_cnt = None
        roi = None
        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # check to see if the aspect ratio is rectangular

            if self.minAR <= ar <= self.maxAR:
                # store the license plate contour and extract the
                # license plate from the grayscale image and then
                # threshold it
                lp_cnt = c
                license_plate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(license_plate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # check to see if we should clear any foreground
                # pixels touching the border of the image
                # (which typically, not but always, indicates noise)
                if clear_border_flag:
                    roi = clear_border(roi)
                # display any debugging information and then break
                # from the loop early since we have found the license
                # plate region
                self.debug_imshow("License Plate", license_plate)
                self.debug_imshow("ROI", roi, waitKey=True)
                break

        # return a 2-tuple of the license plate ROI and the contour
        # associated with it
        return roi, lp_cnt
