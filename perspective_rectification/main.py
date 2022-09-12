import cv2
import matplotlib.pyplot as plt
import numpy as np

from absl import app, flags
from absl.flags import FLAGS

from perspective_rectification.abrupt_corners import abrupt_changes_algorithm
from perspective_rectification.hough_transform_intersections import hough_lines_main
from perspective_rectification.ploting import plot_points
from perspective_rectification.pre_process import pre_process
from perspective_rectification.utils import sort_points, filter_intersections, get_intersection

flags.DEFINE_string('algorithm', 'hough_lines', 'hough_lines or abrupt_changes')
flags.DEFINE_string('image', 'dataset/plate.png', 'path to input image')


def main(_argv):
    """
    Run the algorithm of automatic rectification. Steps:
    - Read and preprocess the image until we have a binaryzed central square object
    - Run one of the algorithm to extract 4 lines corresponding to each of the corners
    - Get the intersections between each pari of lines
    - Apply homography transform
    """
    # Read image
    image = cv2.imread(FLAGS.image)
    height, width, _ = image.shape
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    masked_image, contours = pre_process(image)

    if FLAGS.algorithm == "hough_lines":
        lines = hough_lines_main(image, contours)
    elif FLAGS.algorithm == "abrupt_changes":
        lines = abrupt_changes_algorithm(image, masked_image)
    else:
        raise "Algorithm not specified"

    intersections = get_intersection(lines, "two_point")
    filtered_intersections = filter_intersections(image, intersections)
    ordered_intersections = sort_points(destination, filtered_intersections)
    corners_points = np.float32(ordered_intersections)

    plot_points(filtered_intersections, image)

    # Extract perspective transform and wrap the image
    transformation_matrix = cv2.getPerspectiveTransform(corners_points, destination)
    warped_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

    plt.imshow(warped_image)
    plt.show()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
