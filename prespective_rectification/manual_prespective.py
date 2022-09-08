import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

import imutils
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.cluster import KMeans

from prespective_rectification.pre_process import pre_process


def convert_theta_to_two_points(x, y, theta):
    x2 = x + 1000 * np.cos(theta)
    y2 = y + 1000 * np.sin(theta)
    x1 = x - 1000 * np.cos(theta)
    y1 = y - 1000 * np.sin(theta)
    line = [x1, y1, x2, y2]
    return line


def convert_theta_rho_to_two_points(rho, theta):
    if rho < 0:
        rho *= -1
        theta -= np.pi
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    line = [x1, y1, x2, y2]
    return line


def draw_lines(img, lines, labels=None, type_line="two_points"):
    for i, line in enumerate(lines):
        red = 0
        green = 0
        blue = 0
        if labels is None:
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)
        else:
            if labels[i] == 0:
                red = 255
            elif labels[i] == 1:
                green = 255
            elif labels[i] == 2:
                blue = 255
        if type_line == "rho_theta":
            try:
                rho, theta = line
            except:
                line = line[0]
                rho, theta = line
            convert_theta_rho_to_two_points(rho, theta)
        elif type_line == "two_points":
            line = line[0]
        elif type_line == "theta":
            x, y, theta = line
            line = convert_theta_to_two_points(x, y, theta)
        try:
            cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [red, green, blue], 1)
        except:
            pass
    plt.imshow(img)
    plt.show()


def select_points_by_segment(point_start, point_end, len_points):
    if point_start > point_end:
        moved_initial_point = len_points - point_start
        mid_point = (moved_initial_point + point_end) // 2
        difference = int((point_end - point_start) * 0.25)
        marked_point_lower = mid_point - difference - moved_initial_point
        marked_point_upper = mid_point + difference - moved_initial_point
        if marked_point_lower < 0:
            marked_point_lower = len_points + marked_point_lower
    else:
        mid_point = (point_start + point_end) // 2
        difference = int((point_end - point_start) * 0.25)
        marked_point_lower = mid_point - difference
        marked_point_upper = mid_point + difference
    return marked_point_lower, marked_point_upper


def plot_points(points, image):
    for point in points:
        if type(point[0]) == list or type(point[0]) == np.ndarray:
            point = point[0]
        plt.plot(point[0], point[1], marker='o', color="red")
    plt.imshow(image)
    plt.show()


def get_intersection(polynomial_coeff_list, type_intersection="rho_theta"):
    # Get intersections with lines
    list_intersections = []
    for i, _ in enumerate(polynomial_coeff_list):
        for j, _ in enumerate(polynomial_coeff_list):
            if i != j:
                try:
                    x0, y0 = None, None
                    if type_intersection == "polynomial":
                        x0 = (polynomial_coeff_list[i] - polynomial_coeff_list[j]).roots()
                        y0 = polynomial_coeff_list[i](x0)
                    elif type_intersection == "rho_theta":
                        points = intersection_rho_theta(polynomial_coeff_list[i], polynomial_coeff_list[j])
                        if points is not None:
                            x0, y0 = points
                        else:
                            continue
                    elif type_intersection == "two_point":
                        x0, y0 = two_point_intersection(polynomial_coeff_list[i], polynomial_coeff_list[j])

                    list_intersections.append([x0, y0])
                except:
                    print("paralel")
    return list_intersections


def intersection_rho_theta(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    if abs(rho2 - rho1) > 30:
        a = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(a, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return x0, y0
    else:
        return None


def two_point_intersection(line1, line2):
    if len(line1) == 4:
        line1 = [[line1[0], line1[1]], [line1[2], line1[3]]]
        line2 = [[line2[0], line2[1]], [line2[2], line2[3]]]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_corners_abrupt_changes(cumulative_difference):
    flag_off = True
    total_cumulative = []
    for i, element in enumerate(cumulative_difference):

        if not element:

            if flag_off:
                current_cumulative = []
                counter = 0

                for series_contigous in cumulative_difference[i:]:
                    if series_contigous:
                        break
                    counter += 1

                    current_cumulative.append(i + counter)
                total_cumulative.append(current_cumulative)
                flag_off = False
        else:
            flag_off = True
    total_cumulative = sorted(total_cumulative, key=lambda l: (len(l), l))
    quadratic_changes = []

    for change in total_cumulative[-2:]:
        quadratic_changes.extend([change[0], change[len(change) - 1]])
    quadratic_changes.sort()

    return quadratic_changes


def alternative_abrutp_chages(time_series_abrupt_changes):
    points_abrupt = []
    flag_off = True
    for i, series in enumerate(time_series_abrupt_changes):
        if series:
            if flag_off:
                counter = 0
                for series_contigous in time_series_abrupt_changes[i:]:
                    if not series_contigous:
                        break
                    counter += 1

                points_abrupt.append(i + (counter // 2))
                flag_off = False
        else:
            flag_off = True
    points_abrupt.append(len(time_series_abrupt_changes) - 1)
    return points_abrupt


def abbrupt_changes_algorithm(image, masked_image):
    # gather and clean contours
    cnts = cv2.findContours(masked_image.astype("uint8"), cv2.RETR_EXTERNAL,
                            1)

    cnts = imutils.grab_contours(cnts)

    cleaned_cnts = []
    for point_element in cnts[0]:
        cleaned_cnts.append(list(point_element))
    plot_points(cleaned_cnts, image)

    # project the contours in a sum of diference list
    previous_element = None
    list_diference_x, list_diference_y = [], []
    diference_sum_x, diference_sum_y = 0, 0
    for element in cnts[0]:
        if previous_element is not None:
            diference_sum_x += (element[0][0] - previous_element[0][0])
            list_diference_x.append(diference_sum_x)
            diference_sum_y += (element[0][1] - previous_element[0][1])
            list_diference_y.append(diference_sum_y)
        previous_element = element
    y_range = range(len(list_diference_x))

    smoothed_difference_y = savgol_filter(list_diference_y, 13, 5)  # window size 13, polynomial order 5
    smoothed_difference_x = savgol_filter(list_diference_x, 13, 5)  # window size 13, polynomial order 5

    # plt.plot(y_range, smoothed_difference_y)
    # plt.show()

    plt.plot(y_range, smoothed_difference_x)
    plt.show()

    s = pd.Series(smoothed_difference_x)
    d = pd.Series(s.values[1:] - s.values[:-1], index=s.index[:-1]).abs()

    a = .7
    m = d.max()
    print(d > m * a)

    r = d.rolling(3, min_periods=1, win_type='parzen').sum()
    n = r.max()
    time_series_abrupt_changes = r > n * a

    points_abrupt = get_corners_abrupt_changes(time_series_abrupt_changes)

    # Get the lines of the borders
    if len(points_abrupt) == 4:
        initial_point = points_abrupt[-1]
    else:
        initial_point = 0
    two_points_list = []
    polynomial_coeff_list = []
    points_corners_lines = []
    for point in points_abrupt:
        # plot_points(cleaned_cnts[initial_point:point], image)

        index = select_points_by_segment(initial_point, point, len(time_series_abrupt_changes))
        lower_points = cnts[0][index[0]][0]
        upper_points = cnts[0][index[1]][0]
        two_points_list.append([lower_points, upper_points])
        points_corners_lines.append(lower_points)
        points_corners_lines.append(upper_points)
        x_points = [lower_points[0], lower_points[1]]
        y_points = [upper_points[0], upper_points[1]]

        polynomial_coeff = np.polynomial.Polynomial.fit(x_points, y_points, 1, domain=[-1, 1])
        polynomial_coeff_list.append(polynomial_coeff)
        initial_point = point
    plot_points(points_corners_lines, image)

    list_intersections = get_intersection(two_points_list, "two_point")
    plot_points(list_intersections, image)
    list_intersections = filter_intersections(image, list_intersections)
    return list_intersections


def get_strong_lines(lines):
    strong_lines = np.zeros([4, 1, 2])

    n2 = 0
    for n1 in range(0, len(lines)):
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                if rho < 0:
                    rho *= -1
                    theta -= np.pi
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=10)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
    return strong_lines


def filter_intersections(image, list_intersections):
    height, width, _ = image.shape
    filtered_intesersections = []
    for intersection in list_intersections:
        if intersection[0] > width or intersection[0] < 0 or intersection[1] > height or intersection[1] < 0 or \
                intersection in filtered_intesersections or len(filtered_intesersections) == 4:
            continue
        else:
            filtered_intesersections.append(intersection)
    return filtered_intesersections


def hough_implementations(image, cnts):
    height, width, _ = image.shape

    lines = cv2.HoughLines(cnts, 1, np.pi / 180, 40)
    draw_lines(image, lines)

    lines_reduced = []

    for line in lines:
        rho, theta = line[0]
        if rho < 0:
            rho *= -1
            theta -= np.pi
        lines_reduced.append([rho, theta])

    list_intersections = get_intersection(lines_reduced)
    list_intersections = filter_intersections(image, list_intersections)

    plot_points(list_intersections, image)
    startpts = np.array([[0.0, 0.0], [0.0, height], [width, 0.0], [width, height]], np.float64)
    kmeans = KMeans(n_clusters=4, init=startpts, n_init=1)
    kmeans.fit(list_intersections)

    plot_points(kmeans.cluster_centers_, image)

    lines = cv2.HoughLinesP(cnts, 1, np.pi / 180, 30, 10)

    list_polynomials, two_points_list = [], []
    for line in lines:
        line = line[0]
        x_points = line[0], line[2]
        y_points = line[1], line[3]
        two_points_list.append(line)
        polynomial_coeff = np.polynomial.Polynomial.fit(x_points, y_points, 1, domain=[-1, 1])
        list_polynomials.append(polynomial_coeff.coef)

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(list_polynomials)
    domain = np.array([-1.0, 1.0], dtype=np.float32)
    window = np.array([-1.0, 1.0], dtype=np.float32)
    centroid_coef = []
    xi = np.linspace(0, 200, 200)
    for centroid in kmeans.cluster_centers_:
        polynomial = np.polynomial.Polynomial(coef=centroid, domain=domain, window=window)
        centroid_coef.append(polynomial)
        plt.plot(xi, polynomial(xi))
    plt.imshow(image)
    plt.show()

    list_intersections = get_intersection(centroid_coef, "polynomial")
    plot_points(list_intersections, image)

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(two_points_list)

    list_intersections = get_intersection(kmeans.cluster_centers_, "two_points")
    plot_points(list_intersections, image)
    draw_lines(image, kmeans.cluster_centers_)


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


def hough_lines_nicolas(image, cnts):
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
    list_intersections = get_intersection(out_lines, "two_point")
    list_intersections = filter_intersections(image, list_intersections)

    return list_intersections


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


def order_points(destinations, predicted):
    ordered_predicted = []

    for destination in destinations:
        distances = []
        for prediction in predicted:
            distances.append(np.sqrt((destination[0] - prediction[0]) ** 2 + (destination[1] - prediction[1]) ** 2))
        index = distances.index(min(distances))
        ordered_predicted.append(predicted[index])
    return ordered_predicted


def main():
    # Read image
    image_name = "dataset/plate.png"
    image = cv2.imread(image_name)
    height, width, _ = image.shape
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    masked_image, cnts = pre_process(image)
    intersections = hough_lines_nicolas(image, cnts)

    # intersections = abbrupt_changes_algorithm(image, masked_image)
    # intersections = hough_implementations(image, cnts)

    intersections = order_points(destination, intersections)
    corners_points = np.float32(intersections)
    transformation_matrix = cv2.getPerspectiveTransform(corners_points, destination)
    min_val = np.min(destination[np.nonzero(destination)])
    max_val = np.max(destination[np.nonzero(destination)])
    warped_image = cv2.warpPerspective(image, transformation_matrix, (width, height))
    plt.imshow(warped_image)
    plt.show()


if __name__ == '__main__':
    main()
