import cv2
import numpy as np


class Coordinates(object):
    """
    coordinates of the number of intersections obtained
    """
    coord = []
    size = -1

    def __init__(self):
        self.size += 1
        self.centroid_x = 0
        self.centroid_y = 0
        self.sum_x = 0
        self.sum_y = 0
        self.corners = []
        self.quad = 4

    def centroid_xy(self, x, y):
        self.sum_x += x
        self.sum_y += y

    def append(self, x, y):
        """
        append the coordinates of intersections
        """
        self.centroid_xy(x, y)
        self.coord.append([int(x), int(y)])
        self.size += 1

    def quad_check(self):
        """
        check if the points make up a quadrilateral
        """
        self.coord = np.reshape(self.coord, (self.size, 1, 2))
        peri = cv2.arcLength(self.coord, True)
        approx = cv2.approxPolyDP(self.coord, 0.1 * peri, True)
        print(approx, "approx", len(approx))
        if len(approx) == self.quad:
            print("yes a quad")
            self.coord = approx.tolist()
            return True
        else:
            print("not a quad")
            self.coord = approx.tolist()
            return False

    def calculate_centroid(self):
        """
        find the centroid of the points of intersections found
        """
        self.centroid_x = self.sum_x / self.size
        self.centroid_y = self.sum_y / self.size
        print("centroids", self.centroid_x, self.centroid_y)

    @staticmethod
    def intersection(point_1, point_2, point_3, point_4):
        """
        find the intersection points of all the hull structures found
        """
        x1 = point_1[0]
        y1 = point_1[1]
        x2 = point_2[0]
        y2 = point_2[1]
        x3 = point_3[0]
        y3 = point_3[1]
        x4 = point_4[0]
        y4 = point_4[1]
        d = (((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
        if d:
            inter_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
            inter_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        else:
            inter_x, inter_y = None, None
        print("intersections", inter_x, inter_y)
        return [inter_x, inter_y]

    def calculateTRTLBRBL(self):
        """
        find the Top right, Top left, Bottom right and Bottom left points
        """
        top_points = []
        bottom_points = []
        for coord in self.coord:
            print(coord, type(coord))
            if coord[0][1] < self.centroid_y:
                top_points.append(coord)
            else:
                bottom_points.append(coord)

        top_left = min(top_points)
        top_right = max(top_points)
        bottom_right = max(bottom_points)
        bottom_left = min(bottom_points)

        self.corners.append(top_left)
        self.corners.append(top_right)
        self.corners.append(bottom_right)
        self.corners.append(bottom_left)
        print(self.corners)
        return self.corners
