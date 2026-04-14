from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

def ZS_map(a,b,x,y): # Function
    return -(a*x)/(1+y**2), x+b*y

def ZS(x0 = 0.1, y0 = 0.1):
    a = 3.8
    b = 0.6
    N = 10000000
    xs = []
    ys = []
    x,y = x0,y0
    for _ in range(N):
        x,y = ZS_map(a,b,x,y)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, s=0.0001, color="#EB5E00", alpha=1, label="x")
    plt.xlabel("x")
    plt.ylabel("y")
    
    box = Polygon([(-0.5,-1.05), (-0.5, -0.75), (-0.2, -0.75), (-0.2, -1.05)]) # Define boxes and zoom into the boundaries each time

    x_tr, y_tr = box.exterior.xy
    plt.plot(x_tr, y_tr, color="black", linewidth=0.5)
    plt.show()
    
    plt.figure(figsize=(6,6))
    box = Polygon([(-0.28,-0.93), (-0.28, -0.87), (-0.34, -0.87), (-0.34, -0.93)])    
    plt.xlim(-0.5, -0.2)
    plt.ylim(-1.05, -0.75)
    plt.scatter(xs, ys, s=0.5, color="#EB5E00", alpha=1, label="x")
    x_tr, y_tr = box.exterior.xy
    plt.plot(x_tr, y_tr, color="black", linewidth=0.5)
    plt.show()
    
    plt.figure(figsize=(6,6))   
    box = Polygon([(-0.31,-0.91), (-0.31, -0.89), (-0.3, -0.89), (-0.3, -0.91)])  
    plt.xlim(-0.34, -0.28)
    plt.ylim(-0.93, -0.87)
    plt.scatter(xs, ys, s=0.5, color="#EB5E00", alpha=1, label="x")
    x_tr, y_tr = box.exterior.xy
    plt.plot(x_tr, y_tr, color="black", linewidth=0.5)
    plt.show()

ZS()