import numpy as np


def convert_theta_to_two_points(x, y, theta):
    """
    Conversion from theta x avg and y avg to two sets of 2d points projected outside the image

    """
    x2 = x + 1000 * np.cos(theta)
    y2 = y + 1000 * np.sin(theta)
    x1 = x - 1000 * np.cos(theta)
    y1 = y - 1000 * np.sin(theta)
    return [x1, y1, x2, y2]


def convert_theta_rho_to_two_points(rho, theta):
    """
    Conversion from theta rho to two sets of 2d points projected outside the image
    """
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
    return [x1, y1, x2, y2]


def get_intersection(lines, type_intersection="rho_theta"):
    """
    For each set of points gather each other intersection if possible given an algorithm for each
    coordinate representation
    """
    # Get intersections with lines
    list_intersections = []
    for i, _ in enumerate(lines):
        for j, _ in enumerate(lines):
            if i == j:
                continue
            try:
                x0, y0 = None, None
                if type_intersection == "polynomial":
                    x0 = (lines[i] - lines[j]).roots()
                    y0 = lines[i](x0)
                elif type_intersection == "rho_theta":
                    points = intersection_rho_theta(lines[i], lines[j])
                    if points is not None:
                        x0, y0 = points
                    else:
                        continue
                elif type_intersection == "two_point":
                    x0, y0 = two_point_intersection(lines[i], lines[j])

                list_intersections.append([x0, y0])
            except:
                print("Those lines are parallel they will never intersect")
    return list_intersections


def intersection_rho_theta(line1, line2):
    """
    Intersection using rho and theta circular representation
    """
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
    """
    Intersection using two sets of 2d points
    """
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


def sort_points(destinations, predicted):
    """
    Sort points by getting the minimum distance for each point to each corner of the image
    @param destinations: Four initial corners of the image (this can be improved by saving the A/R of the license plate
    and applying afterwards
    @param predicted: The corners of the license plate gathered by the intersection of the 4 lines
    """
    ordered_predicted = []

    for destination in destinations:
        distances = []
        for prediction in predicted:
            distances.append(np.sqrt((destination[0] - prediction[0]) ** 2 + (destination[1] - prediction[1]) ** 2))
        index = distances.index(min(distances))
        ordered_predicted.append(predicted[index])
    # TODO : calculate the closest corner for each prediction instead of the closest prediction for each corner.
    # TODO : Force each corner having only one prediction
    return ordered_predicted


def filter_intersections(image, list_intersections):
    """
    Remove those intersections that take part outside the image so we only have 4
    """
    height, width, _ = image.shape
    filtered_intesersections = []
    for intersection in list_intersections:
        if intersection[0] > width or intersection[0] < 0 or intersection[1] > height or intersection[1] < 0 or \
                intersection in filtered_intesersections or len(filtered_intesersections) == 4:
            # Optional, maybe we should give more space to intersect outside the image to make sure that we gather 4
            # points
            continue
        else:
            filtered_intesersections.append(intersection)
    return filtered_intesersections
