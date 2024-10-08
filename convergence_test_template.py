from src.xfoil.xfoil import XFoil
import numpy as np
import matplotlib.pyplot as plt
from xfoilpython.main import load_airfoil
from naca_generator import generate


def get_gradient(x):
    delta = 1e-3
    upper = xfoil_drag_result(x + delta)
    lower = xfoil_drag_result(x - delta)
    gradient = (upper - lower) / (2 * delta)
    return gradient


def xfoil_drag_result(x):
    generate(150, 1.0, x)
    xf = XFoil()
    xf.airfoil = load_airfoil('naca4412_airfoil_points.csv')
    xf.Re = 1e6
    xf.max_iter = 150
    xf.n_crit = 9
    a, cl, cd, cm, cp = xf.aseq(-5, 5, 0.5)

    return np.nanmean(cd)


def main():
    airfoil_thickness = 0.12
    grad = np.inf
    learning_rate = 1e-2
    grad_values = []
    cost_values = []
    x_values = []
    results_list = []

    i = 0
    while abs(grad) > 1e-3:
        grad = get_gradient(airfoil_thickness)
        airfoil_thickness -= learning_rate * grad
        cd = xfoil_drag_result(airfoil_thickness)
        t = airfoil_thickness

        results_list.append((cd, t))
        for result in results_list:
            print(f"{result[0]:.5f}, {result[1]:.5f}")
        _ = input()
        grad_values.append(airfoil_thickness)
        cost_values.append(xfoil_drag_result(airfoil_thickness))
        x_values.append(airfoil_thickness)
        i += 1
        if i >= 1000:
            break
    plt.figure()
    plt.plot(cost_values, label="grad")
    plt.plot(grad_values, label="cost")
    plt.plot(x_values, label="x values")
    plt.xlabel("number of iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
