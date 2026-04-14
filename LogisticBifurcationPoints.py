import numpy as np
from scipy.optimize import brentq

def logistic(x,r):
    return r*x*(1 - x)

def iterate_map_points(r, x0=0.5, n=1000, np=1):
    x = x0
    for _ in range(n):
        x = logistic(x,r)
    orbit = []
    for _ in range(np):
        x = logistic(x,r)
        orbit.append(x)
    return orbit

def derivative_n(r,n):
    orbit = iterate_map_points(r, n=2000, np=n)
    prod = 1.0
    for x in orbit:
        prod *= r * (1 - 2*x)
    return abs(prod) - 1   # Extract the root of the absolute derivative < 1 (Stability Test of kth period order)

def find_bifurcation(n, r_min, r_max):
    return brentq(lambda r: derivative_n(r,n), r_min, r_max) # Using Brendt's Hybrid Root-Finder
    
if __name__ == "__main__":
    r1 = find_bifurcation(2**0, 2.9, 3.1)   
    r2 = find_bifurcation(2**1, 3.4, 3.5)
    r3 = find_bifurcation(2**2, 3.53, 3.56) 
    r4 = find_bifurcation(2**3, 3.55, 3.57) # Narrowing the intervals to prevent error
    #...
    rinfty = find_bifurcation(2**8, 3.56, 3.57)

    print("r1 =", r1)
    print("r2 =", r2)
    print("r3 =", r3)
    print("r4 =", r4)
    print("...")
    print("r∞ =", rinfty)