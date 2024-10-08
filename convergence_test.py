from src.xfoil.xfoil import XFoil
import csv
import numpy as np
import matplotlib.pyplot as plt
from xfoilpython.main import load_airfoil



xf = XFoil()
xf.airfoil = load_airfoil('naca4412_airfoil_points.csv')
xf.Re = 1e6
xf.max_iter = 150
xf.n_crit = 9
a, cl, cd, cm, cp = xf.aseq(-5, 5, 0.5)

print()
print(np.nanmean(cd))