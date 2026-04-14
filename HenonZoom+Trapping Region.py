from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

def henon_map(a,b,x,y): # Function
    return 1-a*x**2 + y, b*x

def henon(x0 = 0.1, y0 = 0.1):
    a = 1.4
    b = 0.3
    N = 10000000
    xs = []
    ys = []
    x,y = x0,y0
    for _ in range(N):
        x,y = henon_map(a,b,x,y)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(4, 6))
    plt.xlim(-2, 2)
    plt.scatter(xs, ys, s=0.0001, color="#EB5E00", alpha=1, label="x")
       
    box = Polygon([(0.54,0.15), (0.54, 0.21), (0.72, 0.21), (0.72, 0.15)]) # Define boxes and zoom into the boundaries each time
    x_tr, y_tr = box.exterior.xy
    plt.plot(x_tr, y_tr, color="cyan", linewidth=0.5)
    x_tr, y_tr = box.exterior.xy
    plt.plot(x_tr, y_tr, color="black", linewidth=0.5)
    plt.show()
    
    plt.figure(figsize=(6,6))
    box = Polygon([(0.62,0.185), (0.62, 0.191), (0.64, 0.191), (0.64, 0.185)])    
    plt.xlim(0.54, 0.72)
    plt.ylim(0.15, 0.21)
    plt.scatter(xs, ys, s=0.1, color="#EB5E00", alpha=1, label="x")
    x_tr, y_tr = box.exterior.xy
    plt.scatter(xs, ys, s=0.0001, color="#EB5E00", alpha=1, label="x")
    plt.plot(x_tr, y_tr, color="black", linewidth=0.5)
    plt.show
    
    plt.figure(figsize=(6,6))
    box = Polygon([(0.6303,0.1889), (0.6303, 0.1895), (0.6325, 0.1895), (0.6325, 0.1889)])    
    plt.xlim(0.62, 0.64)
    plt.ylim(0.185, 0.191)
    plt.scatter(xs, ys, s=0.1, color="#EB5E00", alpha=1, label="x")
    x_tr, y_tr = box.exterior.xy
    plt.scatter(xs, ys, s=1, color="#EB5E00", alpha=1, label="x")
    plt.plot(x_tr, y_tr, color="black", linewidth=0.5)
    plt.show      

henon()