import re

from pytesseract import pytesseract


class TesseractOcr:
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
    def __init__(self, psm=8, oem=3, cleanup=False):
        self.psm = psm
        self.oem = oem
        self.cleanup = cleanup
        self.options = self.build_tesseract_options()

    def build_tesseract_options(self):
        """
        Set the mode, the oem and the dictionary
        """
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(self.psm)
        # set the oem
        options += " --oem {}".format(self.oem)
        # return the built options string
        return options

    def run(self, lp):
        """
        Run tesseract ocr from pytesseract and clean the output
        """
        lp_text = pytesseract.image_to_string(lp, config=self.options)
        lp_text = re.sub('[\W_]+', '', lp_text)
        return lp_text
