import numpy as np
import csv
import matplotlib.pyplot as plt

# naca4412 param
m = 0.04  # maximum camber
p = 0.4  # location of maximum camber
t = 0.12  # maximum thickness
chord_length = 1.0  # chord length


def naca4412_points(nPts, length=1.0, thickness=0.12):
    x = np.linspace(0, 1, nPts) * length
    yt = 5 * thickness * (0.2969 * np.sqrt(x / length) - 0.1260 * (x / length)
                          - 0.3516 * (x / length) ** 2 + 0.2843 * (x / length) ** 3 - 0.1015 * (x / length) ** 4)

    yc = np.where(x <= p * length, m / p ** 2 * (2 * p * x / length - (x / length) ** 2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x / length - (x / length) ** 2))
    dyc_dx = np.where(x <= p * length, 2 * m / p ** 2 * (p - x / length),
                      2 * m / (1 - p) ** 2 * (p - x / length))
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    lower_surface = [(xl[i], yl[i]) for i in range(len(x))]
    upper_surface = np.array([(xu[i], yu[i]) for i in range(len(x))])
    upper_surface = np.flipud(upper_surface)  # Reverse the upper surface order

    points = list(upper_surface) + lower_surface

    return points


def save_to_csv(points, filename='airfoil_points.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for point in points:
            writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}"])
    print(f"Points saved to {filename}")


def remove_close_points(points, min_distance):
    new_points = [points[0]]  # Start with the second non-zero point
    for i in range(1, len(points)):
        distance = np.sqrt((points[i][0] - new_points[-1][0]) ** 2 + (points[i][1] - new_points[-1][1]) ** 2)
        if distance >= min_distance:
            new_points.append(points[i])
    return new_points


def plot_from_csv(filename):
    x_coords = []
    y_coords = []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            x_coords.append(float(row[0]))
            y_coords.append(float(row[1]))

    plt.plot(x_coords, y_coords, label="Airfoil")
    plt.title("naca4412 airfoil")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


airfoil_points = naca4412_points(200, length=1.0, thickness=0.12)

filtered_points = remove_close_points(airfoil_points, 0.00001)

save_to_csv(filtered_points, 'naca4412_airfoil_points.csv')

plot_from_csv('naca4412_airfoil_points.csv')
