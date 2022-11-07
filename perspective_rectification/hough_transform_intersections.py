import cv2
import numpy as np
from sklearn.cluster import KMeans

from perspective_rectification.ploting import draw_lines
from perspective_rectification.utils import convert_theta_to_two_points


def hough_lines_main(image, contours):
    """
    Extract hough lines probabilistic, gather the cluster parameters, initialize the points of the clustering
    fit the cluster, get the central points of the cluster and convert the x, y theta to 2d x y
    """
    height, width, _ = image.shape

    # Prepare and extract the lines
    min_line_length = 10
    max_line_gap = 10
    lines = cv2.HoughLinesP(contours, 1, np.pi / 180, 15, minLineLength=min_line_length, maxLineGap=max_line_gap)

    clustering_parameters_modified, clustering_parameters = get_parameters_clustering(lines)

    # Fit clustering
    start_pts = gather_initial_points(image, clustering_parameters_modified)
    kmeans = KMeans(n_clusters=4, init=start_pts, n_init=1)
    kmeans.fit(clustering_parameters_modified)
    averaged_lines = average_lines_clustered(clustering_parameters, kmeans.labels_)
    draw_lines(image, averaged_lines, type_line="theta")

    # Convert lines to 2d x y
    out_lines = [convert_theta_to_two_points(x, y, theta) for x, y, theta in averaged_lines]
    return out_lines


def get_parameters_clustering(lines):
    """
    Merge x and y and extract theta from the slope of the difference of x times y
    """
    clustering_parameters_modified = []
    clustering_parameters = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                theta = np.sign(y2 - y1) * np.pi / 2
            else:
                p = (y2 - y1) / (x2 - x1)
                theta = np.arctan(p)
            x = (x1 + x2) * 0.5
            y = (y1 + y2) * 0.5
            # TODO : Instead of using x and y use an intersection of the 2d points line to the 0 x axis or y axis and
            # TODO : instead of using Theta use the slope P
            clustering_parameters_modified.append([x, y, theta * 100])
            clustering_parameters.append([x, y, theta])
    return clustering_parameters_modified, clustering_parameters


def gather_initial_points(image, points):
    """
    Set initial points of the clustering in the corners of the images so it does not converge to the same point
    as easily as a randomized start
    """
    img_sz_x, img_sz_y, _ = image.shape
    numpy_points = np.array(points)  # your points
    half_x = img_sz_x // 2
    half_y = img_sz_y // 2
    # Set a point using the linear algebra function given each set of point and the offset to the direction of the
    # border
    top_i = np.argmin(np.linalg.norm(numpy_points[:, :2] - np.array([half_x, 0]), axis=1))  # top right corner
    left_i = np.argmin(np.linalg.norm(numpy_points[:, :2] - np.array([0, half_y]), axis=1))  # top left corner
    right_i = np.argmin(
        np.linalg.norm(numpy_points[:, :2] - np.array([img_sz_x, half_y]), axis=1))  # bottom right corner
    bottom_i = np.argmin(
        np.linalg.norm(numpy_points[:, :2] - np.array([half_x, img_sz_y]), axis=1))  # bottom left corner

    top = numpy_points[top_i]
    left = numpy_points[left_i]
    right = numpy_points[right_i]
    bottom = numpy_points[bottom_i]

    corner_points = np.stack((top, left, right, bottom))
    return corner_points


def average_lines_clustered(lines, labels):
    """
    Extract the centroid of each of the parameters of the clusters, for x and y use the median
    for theta use de mean
    """
    filtered_lines = [[] for _ in range(4)]
    out_line = [[] for _ in range(4)]
    for i, line in enumerate(lines):
        label = labels[i]
        filtered_lines[label].append(line)  # gather lines by each label
    for j in range(4):
        # Gather the middle point of the clustered data of x and y axis by using the median
        x_intercept = np.median(list(list(zip(*filtered_lines[j]))[0]))
        y_intercept = np.median(list(list(zip(*filtered_lines[j]))[1]))
        # TODO : we can try using median with theta too
        theta = np.average(list(list(zip(*filtered_lines[j]))[2]))
        out_line[j] = (x_intercept, y_intercept, theta)
        # TODO : it could be improved by differentiating horizontal and vertical lines
    return out_line
