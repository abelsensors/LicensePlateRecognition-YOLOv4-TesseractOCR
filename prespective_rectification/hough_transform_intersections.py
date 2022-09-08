import cv2
import numpy as np
from sklearn.cluster import KMeans

from prespective_rectification.ploting import draw_lines
from prespective_rectification.utils import convert_theta_to_two_points


def hough_lines(image, cnts):
    height, width, _ = image.shape

    min_line_length = 10
    max_line_gap = 10
    lines = cv2.HoughLinesP(cnts, 1, np.pi / 180, 15, minLineLength=min_line_length, maxLineGap=max_line_gap)
    clustering_parameters = []
    clustering_parameters2 = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                theta = np.sign(y2 - y1) * np.pi / 2
            else:
                p = (y2 - y1) / (x2 - x1)
                theta = np.arctan(p)
            x = (x1 + x2) * 0.5
            y = (y1 + y2) * 0.5
            clustering_parameters.append([x, y, theta * 100])
            clustering_parameters2.append([x, y, theta])

    startpts = gather_initial_points(image, clustering_parameters)
    kmeans = KMeans(n_clusters=4, init=startpts, n_init=1)
    kmeans.fit(clustering_parameters)
    averaged_lines = average_lines_clustered(clustering_parameters2, kmeans.labels_)
    draw_lines(image, averaged_lines, type_line="theta")
    out_lines = []
    for line in averaged_lines:
        x, y, theta = line
        out_lines.append(convert_theta_to_two_points(x, y, theta))

    return out_lines




def gather_initial_points(image, points):
    img_sz_x, img_sz_y, _ = image.shape
    rand_points = np.array(points)  # your points
    half_x = img_sz_x // 2
    half_y = img_sz_y // 2
    top_i = np.argmin(np.linalg.norm(rand_points[:, :2] - np.array([half_x, 0]), axis=1))  # top right corner
    left_i = np.argmin(np.linalg.norm(rand_points[:, :2] - np.array([0, half_y]), axis=1))  # top left corner
    right_i = np.argmin(
        np.linalg.norm(rand_points[:, :2] - np.array([img_sz_x, half_y]), axis=1))  # bottom right corner
    bottom_i = np.argmin(
        np.linalg.norm(rand_points[:, :2] - np.array([half_x, img_sz_y]), axis=1))  # bottom left corner

    top = rand_points[top_i]
    left = rand_points[left_i]
    right = rand_points[right_i]
    bottom = rand_points[bottom_i]

    corner_points = np.stack((top, left, right, bottom))
    return corner_points


def average_lines_clustered(lines, labels):
    filtered_lines = [[] for _ in range(4)]
    out_line = [[] for _ in range(4)]
    for i, line in enumerate(lines):
        label = labels[i]
        filtered_lines[label].append(line)
    for i in range(4):
        x = np.median(list(list(zip(*filtered_lines[i]))[0]))
        y = np.median(list(list(zip(*filtered_lines[i]))[1]))
        theta = np.average(list(list(zip(*filtered_lines[i]))[2]))
        out_line[i] = [x, y, theta]
        # it could be improved by differentiating horizontal and vertical lines
    return out_line

