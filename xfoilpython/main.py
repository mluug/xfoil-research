from argparse import ArgumentParser
from src.xfoil.xfoil import XFoil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from src.xfoil.model import Airfoil
from settings import FIG_SIZE, REYNOLDS_NUMBERS, N_CRIT, MAX_ITER


def load_airfoil(filename: str) -> Airfoil:
    """
    Load airfoil data from .dat file.

    Args:
    filename (str): File path of airfoil data.

    Returns: Airfoil object

    """
    coords = pd.read_csv(filename, sep=",", header=None).to_numpy()
    x, y = coords[:, 0], coords[:, 1]

    return Airfoil(x, y)


def plot_airfoil(airfoil: Airfoil, ax) -> None:
    x, y = airfoil.x, airfoil.y

    plt.title("Airfoil")
    ax.plot(x, y, color="black")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_aspect("equal")
    ax.grid()
    return None


def plot_simulation_results(results: tuple, ax) -> None:
    """
    Args:
        results: xfoil simulation results (alpha, cl, cd, cm)
        ax: plt.gax figure

    Returns: None
    """
    alpha, cl, cd, cm, _ = results

    plt.title("Results")

    ax.grid()
    ax.plot(alpha, cl, label="Lift Coefficient (CL)", color="green")
    ax.spines["left"].set_color("green")

    ax2 = ax.twinx()
    ax2.plot(alpha, cd, label="Drag Coefficient (CD)", color="red")
    ax2.spines["right"].set_color("red")

    ax.set_xlabel("Angle of Attack (degrees)")
    ax.set_ylabel("Lift Coefficient")
    ax2.set_ylabel("Drag Coefficient")

    return None


def main() -> None:
    parser = ArgumentParser(
        prog="Python XFoil for Mac",
        description="This program is used to run an XFoil simulation on an airfoil.",
    )
    parser.add_argument(
        "airfoil",
        type=str,
        help="Path to airfoil x,y coordinate file (.csv)",
    )
    parser.add_argument(
        "alpha_max",
        type=float,
        help="Maximum angle of attack in degrees",
    )
    parser.add_argument(
        "alpha_min",
        type=float,
        help="Minimum angle of attack in degrees",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=MAX_ITER,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Angle of attack step size in degrees",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot airfoil",
        default=True,
    )
    args = parser.parse_args()
    airfoil = load_airfoil(args.airfoil)

    # run XFoil simulation
    xf = XFoil()
    xf.airfoil = airfoil
    xf.Re = REYNOLDS_NUMBERS
    xf.n_crit = N_CRIT
    xf.max_iter = args.max_iter
    step = args.step_size
    results = xf.aseq(args.alpha_min, args.alpha_max + step, step)

    # plot results
    fig = plt.figure(figsize=FIG_SIZE)
    spec = gridspec.GridSpec(
        ncols=1, nrows=2, wspace=0.5, hspace=0.5, height_ratios=[1, 4]
    )

    ax0 = fig.add_subplot(spec[0])
    plot_airfoil(airfoil, ax=ax0)

    ax1 = fig.add_subplot(spec[1])
    plot_simulation_results(results, ax=ax1)

    plt.suptitle("XFoil Simulation")

    plt.show()
    return None


if __name__ == "__main__":
    main()
