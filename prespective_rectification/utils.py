import numpy as np


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


def order_points(destinations, predicted):
    ordered_predicted = []

    for destination in destinations:
        distances = []
        for prediction in predicted:
            distances.append(np.sqrt((destination[0] - prediction[0]) ** 2 + (destination[1] - prediction[1]) ** 2))
        index = distances.index(min(distances))
        ordered_predicted.append(predicted[index])
    return ordered_predicted


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
