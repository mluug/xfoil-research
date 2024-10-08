from src.xfoil.xfoil import XFoil


def csv_to_numpy(filename):
    points = []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            points.append([float(row[0]), float(row[1])])

    return np.array(points)

xf = XFoil()
xf.airfoil =
xf.Re = 1e6
xf.max_iter = 150
xf.n_crit = 9
a, cl, cd, cm, cp = xf.aseq(-20, 20, 0.5)

