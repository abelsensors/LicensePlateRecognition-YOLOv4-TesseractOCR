import os

from absl import flags, app
from absl.app import FLAGS
from matplotlib import cm
from skimage.segmentation import clear_border

import cv2
import matplotlib.pyplot as plt

from ocr.multi_character_detection import PyImageSearchANPR
from ocr.single_character_detection import SingleCharacterRecognition
from ocr.tesseract import TesseractOcr
from ocr.utils import resize_aspect_ratio

flags.DEFINE_string('algorithm', 'hough_lines', 'hough_lines or abrupt_changes')
flags.DEFINE_string('image_folder', "data/dataset/plates/generic", 'path to input image')


def main(_argv):
    """
    Page segmentation modes:
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR.
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text.
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
     10    Treat the image as a single character.
     11    Sparse text. Find as much text as possible in no particular order.
     12    Sparse text with OSD.
     13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
    """

    # Read image
    path = FLAGS.image_folder
    os.chdir(path)
    for file in os.listdir():
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = resize_aspect_ratio(gray)

        if FLAGS.algorithm == "background_substraction":
            roi = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            roi = clear_border(roi)
            tesseract = TesseractOcr(psm="7", oem="3")

        elif FLAGS.algorithm == "multi_character_detection":
            pyimage = PyImageSearchANPR()
            candidates = pyimage.locate_license_plate_candidates(gray)
            roi, _ = pyimage.locate_license_plate(gray, candidates)

            tesseract = TesseractOcr(psm="7", oem="3")

        elif FLAGS.algorithm == "single_character_detection":
            single = SingleCharacterRecognition()
            single.run(gray)
            break
        elif FLAGS.algorithm == "simple":
            roi = gray
            tesseract = TesseractOcr(psm="7", oem="3")

        else:
            raise "Algorithm not specified"

        if roi is not None:
            text = tesseract.run(roi)
            plt.imshow(roi, cmap=cm.bone)
            plt.title(text)
            plt.show()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
