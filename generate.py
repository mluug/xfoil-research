import numpy as np
import matplotlib.pyplot as plt
import random


def generate_coordinates(radius, rim_length, rim_thickness, points_count_factor,
                         tire_start_angle, tire_end_angle, tire_x_factor, slope0, slope1):
    # tire larger than rim, to be improved
    tire_radius = radius

    # tire
    theta = np.linspace(tire_start_angle, tire_end_angle, points_count_factor // 4)
    x_tire = tire_x_factor * tire_radius * np.cos(theta) - (tire_x_factor * tire_radius * np.cos(tire_start_angle))
    y_tire = tire_radius * np.sin(theta)

    # numpy array for start/end coords
    start_xy = np.array([0, np.sin(tire_start_angle) * tire_radius])
    end_xy = np.array([rim_length, rim_thickness / 2])

    # fit cubic spline for sidewall
    coeff = cubic_spline(start_xy, end_xy, start_slope=slope0, end_slope=slope1)
    x_rim_upper = np.linspace(end_xy[0], start_xy[0], points_count_factor)
    y_rim_upper = (
            coeff[0] * x_rim_upper ** 3
            + coeff[1] * x_rim_upper ** 2
            + coeff[2] * x_rim_upper ** 1
            + coeff[3] * x_rim_upper ** 0
    )

    # rim back, 1/4 circle
    omega = np.linspace(0, np.pi / 2, points_count_factor // 15)
    x_rim_back = rim_thickness / 2 * np.cos(omega) + rim_length
    y_rim_back = rim_thickness / 2 * np.sin(omega)

    # concat
    x = np.concatenate((x_rim_back, x_rim_upper, x_tire))
    y = np.concatenate((y_rim_back, y_rim_upper, y_tire))

    # mirror
    x = np.concatenate((x, np.flip(x)))
    y = np.concatenate((y, -np.flip(y)))

    # remove close points for xfoil
    x, y = remove_close_points(x, y, min_distance=0.0001)

    # save, tuple
    coords = list(zip(x, y))
    coords = list(dict.fromkeys(coords))  # Remove duplicates
    return coords


def cubic_spline(start_xy, end_xy, start_slope, end_slope):
    # fit a cubic curve
    A = np.array(
        [
            [start_xy[0] ** 3, start_xy[0] ** 2, start_xy[0] ** 1, start_xy[0] ** 0],
            [end_xy[0] ** 3, end_xy[0] ** 2, end_xy[0] ** 1, end_xy[0] ** 0], # points
            [3 * start_xy[0] ** 2, 2 * start_xy[0] ** 1, 1 * start_xy[0] ** 0, 0],
            [3 * end_xy[0] ** 2, 2 * end_xy[0] ** 1, 1 * end_xy[0] ** 0, 0], # slopes, differentiated
        ]
    )

    b = np.array( # constraints
        [
            [start_xy[1]],
            [end_xy[1]],
            [start_slope],
            [end_slope],
        ]
    )
    x = np.linalg.inv(A) @ b
    return np.reshape(x, (4,))


def remove_close_points(x, y, min_distance):
    new_x, new_y = [x[0]], [y[0]]  # Start with the first point
    for i in range(1, len(x)):
        distance = np.sqrt((x[i] - new_x[-1]) ** 2 + (y[i] - new_y[-1]) ** 2)
        if distance >= min_distance:
            new_x.append(x[i])
            new_y.append(y[i])
    return np.array(new_x), np.array(new_y)

def save_as_csv(coordinates, filename="bike.csv"):
    with open(filename, 'w') as f:
        for i, (x, y) in enumerate (coordinates):
            #if i == 0:
            #    continue
            #else:
            f.write(f"{x:.6f},{y:.6f}\n")


def plot(coordinates):
    x_coords, y_coords = zip(*coordinates)
    plt.figure(figsize=(16, 6))
    plt.plot(x_coords, y_coords, marker='o')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def generate_random_parameters():
    radius = random.uniform(0.8, 1.2)
    rim_length = random.uniform(6, 8)
    rim_thickness = random.uniform(0.4, 0.6)
    points_count_factor = random.randint(50, 100)
    tire_start_angle = random.uniform(np.pi / 4, np.pi / 3)
    tire_end_angle = random.uniform(np.pi, 5 * np.pi / 4)
    tire_x_factor = random.uniform(0.9, 1.1)
    slope0 = random.uniform(0, 0.2)
    slope1 = random.uniform(-0.5, -0.1)
    return [radius, rim_length, rim_thickness, points_count_factor, tire_start_angle, tire_end_angle, tire_x_factor,
            slope0, slope1]


def main():
    c = generate_coordinates(1, 7, 0.5, 60, np.pi / 3, np.pi, 1, 0.1, -0.3)
    plot(c)
    save_as_csv(c)


if __name__ == "__main__":
    main()