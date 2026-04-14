import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

a_values = np.linspace(0, 2, 15000)
plt.figure(figsize=(12, 6))

for a in a_values:
    x = np.random.random()
    for i in range(1000): # Burn-in Period
        if x < 0.5:
            x = a * x
        else:
            x = a * (1 - x)
    X, Y = [], []
    for i in range(100):
        if x < 0.5:
            x = a * x
        else:
            x = a * (1 - x)
        X.append(a)
        Y.append(x)
    plt.plot(X, Y, ",", color='#EB5E00', alpha=0.25)

plt.xlabel("a")
plt.ylabel("x")
plt.show()

plt.figure(figsize=(12, 6))
for a in a_values:
    x = np.random.random()
    for _ in range(1000):
       if x < 0.5:
           x = a * x
       else:
           x = a * (1 - x)
    lyap_sum = 0
    for _ in range(100):
        if x < 0.5:
            x = a * x
        else:
            x = a * (1 - x)
        lyap_sum += np.log(abs(a))
    lyap = lyap_sum / 100
    plt.plot(a, lyap, ".", color='#EB5E00', alpha=1)

plt.axhline(0, color="blue", linestyle="--", linewidth=0.5)
plt.show()


a_values = np.linspace(1, 1.25, 15000)
plt.figure(figsize=(12, 6))

for a in a_values:
    x = np.random.random()
    for i in range(1000): # Burn-in Period
        if x < 0.5:
            x = a * x
        else:
            x = a * (1 - x)
    X, Y = [], []
    for i in range(100):
        if x < 0.5:
            x = a * x
        else:
            x = a * (1 - x)
        X.append(a)
        Y.append(x)
    plt.plot(X, Y, ",", color='#EB5E00', alpha=0.25)

plt.xlabel("a")
plt.ylabel("x")
plt.ylim(0.4, 0.6)
plt.show()