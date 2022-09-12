import imutils
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from perspective_rectification.ploting import plot_points


def gather_contours(edges):
    """
    Alternative way to gather the contours as numpy array
    """
    rows = edges.shape[1]
    cols = edges.shape[0]
    new_corners = []
    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            if edges[x, y]:
                new_corners.append(np.array([np.array([x, y])]))
    return np.array(new_corners)


def select_points_by_segment(point_start, point_end, len_points):
    """
    Select the two middle points from the abrupt changes
    @param point_start: corner where the first abrupt change is detected
    @param point_end: corner where the next abrupt change is detected related next to the point_start
    @param len_points: counter of total amount of corners to calculate the abrupt changes that connects the begging of
    the list with the end
    """
    if point_start > point_end:  # if the abrupt change is before ending the sequence we have to start from the end
        #  until the firs point in the beginning
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
    """
    Gather each sequence of corners divided by the abrupt changes and return a list of 4 lists each
    containing one side
    """
    # Sliding window
    series_smoothed = pd.Series(cumulative_difference)

    time_series_parsed = pd.Series(series_smoothed.values[1:] - series_smoothed.values[:-1],
                                   index=series_smoothed.index[:-1]).abs()

    threshold_window = .7
    max_time_series = time_series_parsed.max()
    print(time_series_parsed > max_time_series * threshold_window)

    rolling_windowed = time_series_parsed.rolling(3, min_periods=1, win_type='parzen').sum()
    max_rolling_windowed_series = rolling_windowed.max()
    abrupt_detections = rolling_windowed > max_rolling_windowed_series * threshold_window

    is_new_sequence = True
    total_cumulative = []
    for i, element in enumerate(abrupt_detections):

        if not element:

            if is_new_sequence:  # meanwhile there is non-abrupt changes keep gathering points
                current_cumulative = []
                counter = 0

                for series_contigous in abrupt_detections[i:]:
                    if series_contigous:
                        break
                    counter += 1

                    current_cumulative.append(i + counter)
                total_cumulative.append(current_cumulative)
                is_new_sequence = False
        else:
            # Once there is an abrupt change check for the next one
            is_new_sequence = True
    # TODO : Sort on the mean cumulative change of the points that meet the threshold
    total_cumulative = sorted(total_cumulative, key=lambda l: (len(l), l))
    quadratic_changes = []

    for change in total_cumulative[-2:]:
        quadratic_changes.extend([change[0], change[len(change) - 1]])
    quadratic_changes.sort()

    return quadratic_changes, abrupt_detections


def abrupt_changes_algorithm(image, masked_image):
    """
    Get the contours, extract the sum of difference between each point given one axis, indentify the abrupt changes
    in the time series, select two points for each abrupt change and gather the lines of the contours that traces
    this two lines
    """
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

    window_size = 13
    polynomial_order = 5
    smoothed_difference_y = savgol_filter(list_diference_y, window_size, polynomial_order)
    smoothed_difference_x = savgol_filter(list_diference_x, window_size, polynomial_order)

    plt.plot(y_range, smoothed_difference_x)
    plt.show()

    points_abrupt, abrupt_detections = get_corners_abrupt_changes(smoothed_difference_x)

    # Get the lines of the borders
    if len(points_abrupt) == 4:
        initial_point = points_abrupt[-1]
    else:
        initial_point = 0
    two_points_list = []
    polynomial_coeff_list = []
    points_corners_lines = []
    for point in points_abrupt:
        index = select_points_by_segment(initial_point, point, len(abrupt_detections))
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
