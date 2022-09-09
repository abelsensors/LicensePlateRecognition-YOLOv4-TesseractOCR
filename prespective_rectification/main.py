import cv2
import matplotlib.pyplot as plt
import numpy as np

from prespective_rectification.abrupt_corners import abrupt_changes_algorithm
from prespective_rectification.hough_transform_intersections import hough_lines_main
from prespective_rectification.ploting import plot_points
from prespective_rectification.pre_process import pre_process
from prespective_rectification.utils import order_points, filter_intersections, get_intersection


def main():
    # Read image
    image_name = "dataset/plate.png"
    image = cv2.imread(image_name)
    height, width, _ = image.shape
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    masked_image, cnts = pre_process(image)

    algorithm = "hough_lines"
    if algorithm == "hough_lines":
        lines = hough_lines_main(image, cnts)
    elif algorithm == "abrupt_changes":
        lines = abrupt_changes_algorithm(image, masked_image)
    else:
        raise "Algorithm not specified"

    intersections = get_intersection(lines, "two_point")
    filtered_intersections = filter_intersections(image, intersections)
    ordered_intersections = order_points(destination, filtered_intersections)

    plot_points(filtered_intersections, image)

    corners_points = np.float32(ordered_intersections)
    transformation_matrix = cv2.getPerspectiveTransform(corners_points, destination)
    warped_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

    plt.imshow(warped_image)
    plt.show()


if __name__ == '__main__':
    main()
