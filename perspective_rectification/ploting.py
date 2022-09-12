import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from perspective_rectification.utils import convert_theta_to_two_points, convert_theta_rho_to_two_points


def plot_points(points, image):
    for point in points:
        if type(point[0]) == list or type(point[0]) == np.ndarray:
            point = point[0]
        plt.plot(point[0], point[1], marker='o', color="red")
    plt.imshow(image)
    plt.show()


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
