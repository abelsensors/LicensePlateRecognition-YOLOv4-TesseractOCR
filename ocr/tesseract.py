import re

from pytesseract import pytesseract


class TesseractOcr:
    def __init__(self, psm=8, oem=3):
        self.psm = psm
        self.oem = oem
        self.options = self.build_tesseract_options()

    def build_tesseract_options(self):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(self.psm)
        # set the oem
        options += " --oem {}".format(self.oem)
        # return the built options string
        return options

    @staticmethod
    def cleanup_text(text):
        # strip out non-ASCII text, we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    def run(self, lp):
        lp_text = pytesseract.image_to_string(lp, config=self.options)
        lp_text = re.sub('[\W_]+', '', lp_text)
        lp_text = self.cleanup_text(lp_text)
        return lp_text
