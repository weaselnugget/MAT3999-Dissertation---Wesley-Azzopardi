import numpy as np
from scipy.stats import linregress

def box_counting_dimension(points, epsilons):
    counts = []

    # normalize to unit square (important)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    norm_points = (points - mins) / (maxs - mins)

    for eps in epsilons:
        bins = np.floor(norm_points / eps)
        unique_boxes = np.unique(bins, axis=0)
        counts.append(len(unique_boxes))

    return np.array(counts)

def henon_map(a,b,x,y):
    return 1-a*x**2 + y, b*x

def henon(x0 = 0.1, y0 = 0.1):
    a = 1.4
    b = 0.3
    N = 1000000
    xs = []
    ys = []
    x,y = x0,y0
    for _ in range(N):
        x,y = henon_map(a,b,x,y)
        xs.append(x)
        ys.append(y)
    return xs, ys

x, y = henon()
points = np.column_stack((x, y))
epsilons = np.logspace(-2, -1, 10)
counts = box_counting_dimension(points, epsilons)

log_eps = np.log(1 / epsilons)
log_counts = np.log(counts)
slope, _, _, _, _ = linregress(log_eps, log_counts)
print("Hénon Map: Box-Counting dimension:", slope)