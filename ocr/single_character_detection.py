import re

from pytesseract import pytesseract

from ocr.utils import resize_aspect_ratio
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SingleCharacterRecognition:
    """
    Clean the image and extract individual characters in order to apply tesseract ocr treating the image as an
    individual character
    """
    @staticmethod
    def reduce_colors(img, kernel):
        """
        Cluster all the colors so during the cleaning with the k more prominent ones
        """
        img_z = img.reshape((-1, 3))

        # convert to np.float32
        img_z = np.float32(img_z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(img_z, kernel, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)

        return res2

    def clean_image(self, img):
        """
        Apply some morphological operations to clean the image for the purpose of extracting individual characters
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = resize_aspect_ratio(gray_img)
        resized_img = cv2.GaussianBlur(resized_img, (5, 5), 0)

        equalized_img = cv2.equalizeHist(resized_img)

        reduced = cv2.cvtColor(self.reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8),
                               cv2.COLOR_BGR2GRAY)

        ret, mask = cv2.threshold(reduced, 64, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    @staticmethod
    def extract_characters(img):
        """
        Select those areas where there is rectangles that matches the desired shape and extract it as a individual
        characters
        """
        bw_image = cv2.bitwise_not(img)

        contours, _ = cv2.findContours(bw_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        char_mask = np.zeros_like(img)
        bounding_boxes = []
        for contour in contours:
            float_contours = np.asarray(contour, dtype=np.float32)
            x, y, w, h = cv2.boundingRect(float_contours)
            area = w * h
            center = (x + w / 2, y + h / 2)
            if (area > 1000) and (area < 10000):
                x, y, w, h = x - 4, y - 4, w + 8, h + 8
                bounding_boxes.append((center, (x, y, w, h)))
                cv2.rectangle(char_mask, (x, y), (x + w, y + h), 255, -1)

        clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask=bw_image))

        bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

        characters = []
        for center, bbox in bounding_boxes:
            x, y, w, h = bbox
            char_image = clean[y:y + h, x:x + w]
            characters.append((bbox, char_image))

        return clean, characters

    @staticmethod
    def highlight_characters(img, chars):
        """
        Print a bounding box around the characters extracted
        """
        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for bbox, char_img in chars:
            x, y, w, h = bbox
            cv2.rectangle(output_img, (x, y), (x + w, y + h), 255, 1)

        return output_img

    def run(self, image):
        """
        Run the extraction of the characters individually and run pytesseract for each one
        """

        image = self.clean_image(image)
        clean_img, chars = self.extract_characters(image)

        output_img = self.highlight_characters(clean_img, chars)
        plate_chars = ""
        for _, char_img in chars:
            if char_img.any():
                text = pytesseract.image_to_string(char_img,
                                                   config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
                text = re.sub('[\W_]+', '', text)
                plate_chars += text
        plt.imshow(image)
        plt.title(plate_chars)
        plt.show()
        plt.imsave("results/result_" + plate_chars + ".png", output_img)
