import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

import imutils
from scipy.signal import savgol_filter
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans

from prespective_rectification.pre_process import pre_process


def draw_lines(img, lines, thickness=1):
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    for line in lines:
        try:
            rho, theta = line
        except:
            line = line[0]
            rho, theta = line

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
        try:
            cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [red, green, blue], thickness)
        except:
            pass
    plt.imshow(img)
    plt.show()


def select_points_by_segment(point_start, point_end):
    mid_point = (point_start + point_end) // 2
    difference = int((point_end - point_start) * 0.25)
    marked_point_lower = mid_point - difference
    marked_point_upper = mid_point + difference
    return marked_point_lower, marked_point_upper


def plot_points(points, image):
    for point in points:
        if type(point[0]) == list:
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
                    # x0 = -(left_line[1] - right_line[1]) / (left_line[0] - right_line[0])
                    # y0 = x0 * left_line[0] + left_line[1]
                    x0, y0 = None, None
                    if type_intersection == "polynomial":
                        x0 = (polynomial_coeff_list[i] - polynomial_coeff_list[j]).roots()
                        y0 = polynomial_coeff_list[i](x0)
                    elif type_intersection == "rho_theta":
                        x0, y0 = intersection_rho_theta(polynomial_coeff_list[i], polynomial_coeff_list[j])
                    elif type_intersection == "two_point":
                        x0, y0 = two_point_intersection(polynomial_coeff_list[i], polynomial_coeff_list[j])

                    list_intersections.append([x0, y0])
                except:
                    print("paralel")
    return list_intersections


def intersection_rho_theta(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(a, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


def two_point_intersection(line1, line2):
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


def old_script(image, masked_image):
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

    plt.plot(y_range, smoothed_difference_y)
    plt.show()

    plt.plot(y_range, smoothed_difference_x)
    plt.show()

    s = pd.Series(smoothed_difference_y)
    d = pd.Series(s.values[1:] - s.values[:-1], index=s.index[:-1]).abs()

    a = .6
    m = d.max()
    r = d.rolling(3, min_periods=1, win_type='parzen').sum()
    n = r.max()
    time_series_abrupt_changes = r > n * a
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

    # Get the lines of the borders
    initial_point = 0
    polynomial_coeff_list = []
    points_corners_lines = []
    for point in points_abrupt:
        plot_points(cleaned_cnts[initial_point:point], image)

        index = select_points_by_segment(initial_point, point)
        lower_points = cnts[0][index[0]][0]
        upper_points = cnts[0][index[1]][0]
        points_corners_lines.append(lower_points)
        points_corners_lines.append(upper_points)
        x_points = [lower_points[0], lower_points[1]]
        y_points = [upper_points[0], upper_points[1]]

        polynomial_coeff = np.polynomial.Polynomial.fit(x_points, y_points, 1, domain=[-1, 1])
        polynomial_coeff_list.append(polynomial_coeff)
        initial_point = point
    plot_points(points_corners_lines, image)

    list_intersections = get_intersection(polynomial_coeff_list)
    plot_points(list_intersections, image)


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
        if intersection[0] > height or intersection[0] < 0 or intersection[1] > width or intersection[1] < 0:
            continue
        else:
            filtered_intesersections.append(intersection)
    return filtered_intesersections


def main():
    # Read image
    image_name = "dataset/plate.png"
    image = cv2.imread(image_name)
    height, width, _ = image.shape
    masked_image, cnts = pre_process(image)
    # lines = cv2.HoughLinesP(cnts, 1, np.pi / 180, 30, 10)
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

    # plot_points(list_intersections, image)
    startpts = np.array([[0.0, 0.0], [0.0, height], [width, 0.0], [width, height]], np.float64)
    kmeans = KMeans(n_clusters=4, init=startpts, n_init=1)
    kmeans.fit(list_intersections)

    plot_points(kmeans.cluster_centers_, image)

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


if __name__ == '__main__':
    main()
