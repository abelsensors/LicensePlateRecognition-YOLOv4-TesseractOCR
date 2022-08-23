import cv2
import numpy as np

from coordinates import Coordinates


class Perspective(object):
    """
    change the perspective look of an image
    """

    def __init__(self, source_element):
        self.shape = None
        self.destination = None
        self.source = source_element

    def set_destination(self, img):
        self.shape = img.shape
        width = self.shape[1]
        print(width)
        height = self.shape[0]
        print(height)
        self.destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        print(self.destination)

    def handle(self):
        """
        get the destinations and edges
        """
        img = self.source
        self.set_destination(img)
        edges = cv2.Canny(img, 100, 200, apertureSize=3)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        return edges

    @staticmethod
    def contour_method(edges_objects):
        """
        find the hull/ contour and the intersection points
        """
        hull = None
        contours, hierarchy = cv2.findContours(edges_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            if hierarchy[0, i, 3] == -1 and cv2.contourArea(cnt) > 5000:
                hull = cv2.convexHull(cnt, returnPoints=True)
                break

        length = len(hull)

        coord = Coordinates()
        for i in range(0, length):
            if (i + 3) < length:
                [x, y] = coord.intersection((hull[i][0][0], hull[i][0][1]), (hull[i + 1][0][0], hull[i + 1][0][1]),
                                            (hull[i + 2][0][0], hull[i + 2][0][1]),
                                            (hull[i + 3][0][0], hull[i + 3][0][1]))
                coord.append(x, y)
        return coord

    @staticmethod
    def get_coordinates_hough_p(edges_objects):
        constant = 100
        min_line_length = 10
        max_line_gap = 5
        coord = Coordinates()
        lines = cv2.HoughLinesP(edges_objects, 1, np.pi / 180, constant, min_line_length, max_line_gap)
        print(lines, type(lines))
        points = []
        for x1, y1, x2, y2 in lines[0]:
            points.append([x1, y1])
            points.append([x2, y2])

        [x, y] = coord.intersection(points[0], points[1], points[2], points[3])
        coord.append(x, y)
        return coord

    def transform(self, corners_points):
        """
        transform the points to the destination and return warped image and transformation_matrix
        """
        corners_points = np.float32(
            (corners_points[0][0], corners_points[1][0], corners_points[2][0], corners_points[3][0]))
        transformation_matrix = cv2.getPerspectiveTransform(corners_points, self.destination)
        min_val = np.min(self.destination[np.nonzero(self.destination)])
        print("minVal", min_val, "width", self.shape[0])
        max_val = np.max(self.destination[np.nonzero(self.destination)])
        print("max_val", max_val, "height", self.shape[1])
        warped_image = cv2.warpPerspective(self.source, transformation_matrix, (self.shape[1], self.shape[0]))
        return warped_image, transformation_matrix

    @staticmethod
    def show_sharpen(warped_image):
        """
        improve the image by sharpening it
        """
        cv2.imshow("image", warped_image)
        cv2.waitKey(0)
        blur = cv2.GaussianBlur(warped_image, (5, 5), 2)
        alpha = 1.5
        beta = 1 - alpha  # 1 - alpha
        gamma = 0
        sharpened = cv2.addWeighted(warped_image, alpha, blur, beta, gamma)
        cv2.imshow("sharpened", sharpened)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    source_main = cv2.imread("dataset/card.png", 0)
    pers_p = Perspective(source_main)
    edges_main = pers_p.handle()
    contour_coord_main = pers_p.contour_method(edges_main)
    if contour_coord_main.quad_check():
        contour_coord_main.calculate_centroid()
        corners_main = contour_coord_main.calculateTRTLBRBL()
        warpedImage, transformationMatrix = pers_p.transform(corners_main)
        pers_p.show_sharpen(warpedImage)
