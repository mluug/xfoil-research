import matplotlib.pyplot as plt
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="XFoil Simulation")
    parser.add_argument("--tire-width", type=float)
    parser.add_argument("--rim-width", type=float)

    args = parser.parse_args()
    tire = args.tire_width
    rim = args.rim_width
    print(tire, rim)

    start_xy = np.array([0, 0.5])
    end_xy = np.array([5, 0])
    plt.figure(figsize=(10, 6))
    plt.plot(start_xy[0], start_xy[1], "b*")
    plt.plot(end_xy[0], end_xy[1], "b*")
    plt.axis("equal")

    coeff = cubic_spline(start_xy, end_xy, start_slope=0, end_slope=0)
    rim_x = np.arange(start_xy[0] - 1, end_xy[0] + 1, 0.1)
    rim_y = (
            coeff[0] * rim_x ** 3
            + coeff[1] * rim_x ** 2
            + coeff[2] * rim_x ** 1
            + coeff[3] * rim_x ** 0
    )
    plt.plot(rim_x, rim_y, "r-")
    plt.show()

    return


def cubic_spline(start_xy, end_xy, start_slope, end_slope):
    A = np.array(
        [
            [start_xy[0] ** 3, start_xy[0] ** 2, start_xy[0] ** 1, start_xy[0] ** 0],
            [end_xy[0] ** 3, end_xy[0] ** 2, end_xy[0] ** 1, end_xy[0] ** 0],
            [3 * start_xy[0] ** 2, 2 * start_xy[0] ** 1, 1 * start_xy[0] ** 0, 0], # differentiated, for slope
            [3 * end_xy[0] ** 2, 2 * end_xy[0] ** 1, 1 * end_xy[0] ** 0, 0],
        ]
    )

    b = np.array(
        [
            [start_xy[1]],
            [end_xy[1]],
            [start_slope],
            [end_slope],
        ]
    )
    x = np.linalg.inv(A) @ b
    return np.reshape(x, (4,))


if __name__ == "__main__":
    main()

