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

flags.DEFINE_string('algorithm', 'background_substraction', 'background_substraction or multi_character_detection or '
                                                            'single_character_detection or simple')
flags.DEFINE_string('image_folder', "data/dataset/plates/belgium/croped", 'path to input folder')
flags.DEFINE_integer('max_image_size', 1500, 'maximum size of the resized image given any axis')


def main(_argv):
    # Read image
    path = FLAGS.image_folder
    os.chdir(path)
    for file in os.listdir():
        # Simple pre-process for any image and algorithm
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = resize_aspect_ratio(gray, FLAGS.max_image_size)

        # Algorithm running for any image
        if FLAGS.algorithm == "background_substraction":
            # Some background substraction and cleaning of the borders
            roi = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            roi = clear_border(roi)
            tesseract = TesseractOcr(psm="7", oem="3")

        elif FLAGS.algorithm == "multi_character_detection":
            # Select a region of interest within the license plate where the letters appear
            pyimage = PyImageSearchANPR()
            candidates = pyimage.locate_license_plate_candidates(gray)
            roi, _ = pyimage.locate_license_plate(gray, candidates)

            tesseract = TesseractOcr(psm="7", oem="3")

        elif FLAGS.algorithm == "single_character_detection":
            # Split any character of the license plate and run tesseract ocr for each character individually
            single = SingleCharacterRecognition()
            single.run(gray)
            break

        elif FLAGS.algorithm == "simple":
            # Run Tesseract ocr without any pre-process
            roi = gray
            tesseract = TesseractOcr(psm="7", oem="3")

        else:
            raise "Algorithm not specified"

        # given the region of interest run tesseract ocr
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
