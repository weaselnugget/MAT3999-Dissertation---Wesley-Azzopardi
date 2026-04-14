import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

def vary_a(): # Fixing b, Varying a
    b = 0.6
    a_values = np.linspace(-6, 6, 5000)

    xs = []
    ys = []
    as_plot = []

    for a in a_values:
        x = 0.1
        y = 0.1
        for _ in range(1200): # Burn-in Period
            x, y = -(a*x)/(1+y**2), x+b*y
        
        for _ in range(200): # Collect Points
            x, y = -(a*x)/(1+y**2), x+b*y
            xs.append(x)
            ys.append(y)
            as_plot.append(a)

    plt.figure(figsize=(12, 8))
    plt.scatter(as_plot, xs, s=0.0001, color="#EB5E00", alpha=1, label="x")
    plt.scatter(as_plot, ys, s=0.0001, color="blue", alpha=1, label="y")
    plt.title(f"ZS Bifurcation Plot for a, b={b:.2f}")
    plt.axvline(0.9111, color='g', linestyle='--') # Empirically identified Period-Doubling Points
    plt.axvline(0.36295, color='g', linestyle='--')
    plt.axvline(1.0238, color='g', linestyle='--')
    plt.axvline(1.05131, color='g', linestyle='--')
    plt.xlabel("a")
    plt.show()

    lyaps = []

    for a in a_values:
        x, y = 0.1, 0.1
        for _ in range(1200):
            x, y = -(a*x)/(1+y**2), x+b*y

        v = [1,1]
        lyap_sum = 0
        n_iter = 200
        for _ in range(n_iter):
            L = np.array([
                [-a/(1+y**2), (2*a*x*y)/(1+y**2)**2], [1, b]]) # Jacobian
            v = L @ v # Pass initial vector
            norm_v = np.linalg.norm(v)
            lyap_sum += np.log(norm_v)
            v = v / norm_v
            x, y = -(a*x)/(1+y**2), x+b*y

        lyaps.append(lyap_sum / n_iter)

    plt.figure(figsize=(12, 8))
    plt.scatter(a_values, lyaps, s=0.5, color='#EB5E00')
    plt.axhline(0, color='blue', linestyle='--')
    plt.xlabel("a")
    plt.show()

def vary_b(): # Fixing a, Varying b
    a = 3.8
    b_values = np.linspace(-1.2, 1.2, 5000)

    xs = []
    ys = []
    bs_plot = []

    for b in b_values:
        x = 0.1
        y = 0.1
        for _ in range(1200): # Burn-in Period
            x, y = -(a*x)/(1+y**2), x+b*y
        
        for _ in range(200): # Collect Points
            x, y = -(a*x)/(1+y**2), x+b*y
            xs.append(x)
            ys.append(y)
            bs_plot.append(b)

    plt.figure(figsize=(12, 8))
    plt.scatter(bs_plot, xs, s=0.001, color="#EB5E00", alpha=1, label="x")
    plt.scatter(bs_plot, ys, s=0.001, color="blue", alpha=1, label="y")
    plt.ylim(-10, 10)
    plt.title(f"ZS Bifurcation Plot for b, a={a:.2f}")
    plt.axvline(-0.3761, color='g', linestyle='--') # Empirically identified Period-Doubling Points
    plt.axvline(-0.0944, color='g', linestyle='--')
    plt.axvline(-0.0219, color='g', linestyle='--')
    plt.axvline(-0.0044, color='g', linestyle='--')
    plt.xlabel("b")
    plt.show()

    lyaps = []

    for b in b_values:
        x, y = 0.1, 0.1
        for _ in range(1200):
            x, y = -(a*x)/(1+y**2), x+b*y

        v = [1,1]
        lyap_sum = 0
        n_iter = 200
        for _ in range(n_iter):
            L = np.array([
                [-a/(1+y**2), (2*a*x*y)/(1+y**2)**2], [1, b]]) # Jacobian
            v = L @ v # Pass initial vector
            norm_v = np.linalg.norm(v)
            lyap_sum += np.log(norm_v)
            v = v / norm_v
            x, y = -(a*x)/(1+y**2), x+b*y

        lyaps.append(lyap_sum / n_iter)

    plt.figure(figsize=(12, 8))
    plt.scatter(b_values, lyaps, s=0.5, color='#EB5E00')
    plt.axhline(0, color='blue', linestyle='--')
    plt.xlabel("b")
    plt.show()
    
vary_a() # Seperate using Functions to prevent faulty plots (overused same variables)
vary_b()