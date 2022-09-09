import imutils
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from prespective_rectification.ploting import plot_points


def gather_contours(edges):
    rows = edges.shape[1]
    cols = edges.shape[0]
    new_corners = []
    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            if edges[x, y]:
                new_corners.append(np.array([np.array([x, y])]))
    return np.array(new_corners)


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


def abrupt_changes_algorithm(image, masked_image):
    # gather and clean contours
    cnts = cv2.findContours(masked_image.astype("uint8"), cv2.RETR_EXTERNAL,
                            1)

    cnts = imutils.grab_contours(cnts)

    cleaned_cnts = []
    for point_element in cnts[0]:
        cleaned_cnts.append(list(point_element))
    plot_points(cleaned_cnts, image)

    # project the contours in a sum of difference list
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

    # Sliding window
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

    return two_points_list
