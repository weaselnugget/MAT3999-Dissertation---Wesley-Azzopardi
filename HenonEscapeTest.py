import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

a = 1.6
b = 0.3

def henon_map(x,y,a,b):
    return 1 - a*x**2 + y, b*x
    
def henon_escape(a, b, n=100000, R=1000):
    x, y = 0.1, 0.1
    for _ in range(n):
        x, y = henon_map(x, y, a, b)
        if abs(x) > R or abs(y) > R:
            return True   # Escaped
    return False # Bounded

print(henon_escape(1.425, 0.3))
print(henon_escape(1.4, 0.314)) # Estimated Limits
