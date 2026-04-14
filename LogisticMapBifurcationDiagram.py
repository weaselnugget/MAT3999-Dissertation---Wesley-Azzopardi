import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

# ---- Full View ----
r_values = np.linspace(-2, 4, 15000) # Parameters
plt.figure(figsize=(12, 6))

for r in r_values:
    x = np.random.random()
    for i in range(1000): # Burn-in Period
        x = r * x * (1 - x)
    X, Y = [], []
    for i in range(100):
        x = r * x * (1 - x)
        X.append(r)
        Y.append(x)
    plt.plot(X, Y, ",", color='#EB5E00', alpha = 0.25)
    
ro = 1+np.sqrt(5)
ro2 = 1-np.sqrt(5)
plt.axvline(2, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro2, color="blue", linestyle="--", linewidth=0.8)
plt.axhline(0.5, color="blue", linestyle="--", linewidth=0.8) # Locations of Superstable Points
plt.xlabel("r")
plt.ylabel("x")
plt.show()    

plt.figure(figsize=(12, 6))
for r in r_values:
    x = np.random.random()
    for _ in range(1000):
        x = r*x*(1 - x)

    lyap_sum = 0
    for _ in range(100):
        x = r*x*(1 - x)
        lyap_sum += np.log(abs(r*(1 - 2*x)))

    lyap = lyap_sum / 100
    plt.plot(r, lyap, ",", color='#EB5E00', alpha=1)

plt.axhline(0, color="blue", linestyle="--", linewidth=0.5)
plt.axvline(3.56994, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(2, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro2, color="blue", linestyle="--", linewidth=0.8)

# ---- Full View on Positive Plane ----

r_values = np.linspace(0, 4, 15000) # Parameters
plt.figure(figsize=(12, 6))

for r in r_values:
    x = np.random.random()
    for i in range(1000): # Burn-in Period
        x = r * x * (1 - x)
    X, Y = [], []
    for i in range(100):
        x = r * x * (1 - x)
        X.append(r)
        Y.append(x)
    plt.plot(X, Y, ",", color='#EB5E00', alpha = 0.25)
    
ro = 1+np.sqrt(5)
plt.axvline(2, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro, color="blue", linestyle="--", linewidth=0.8)
plt.axhline(0.5, color="blue", linestyle="--", linewidth=0.8) # Locations of Superstable Points
plt.xlabel("r")
plt.ylabel("x")
plt.show()    

plt.figure(figsize=(12, 6))
for r in r_values:
    x = np.random.random()
    for _ in range(1000):
        x = r*x*(1 - x)

    lyap_sum = 0
    for _ in range(100):
        x = r*x*(1 - x)
        lyap_sum += np.log(abs(r*(1 - 2*x)))

    lyap = lyap_sum / 100
    plt.plot(r, lyap, ",", color='#EB5E00', alpha=1)

plt.axhline(0, color="blue", linestyle="--", linewidth=0.5)
plt.axvline(3.56994, color="black", linestyle="--", linewidth=0.8)
plt.axvline(2, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro, color="blue", linestyle="--", linewidth=0.8)

# ---- Zooming into Chaotic Region ----

P = np.linspace(2.95, 3.7, 10000)
X = []
Y = []

for x in P:
    r = 0.5  # Initial value for r
    for n in range(1000): # Burn-in Period
        r = (x * r) * (1 - r)
    for l in range(100):
        r = (x * r) * (1 - r)
        X.append(x)
        Y.append(r)

plt.figure(figsize=(10, 6))
plt.plot(X, Y, ',', color='#EB5E00', alpha = 0.25) 
plt.axhline(0.5, color="gray", linewidth=0.8)

R1, R2, R3, R4 = 3, 1+np.sqrt(6), 3.544, 3.564 # Empirically found
plt.axvline(R1, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(R2, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(R3, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(R4, color="gray", linestyle="--", linewidth=0.8)

plt.text(R1, 0.93, r"$r_1$", ha="center", va="bottom")
plt.text(R2, 0.93, r"$r_2$", ha="center", va="bottom")
plt.text(R3, 0.93, r"$r_3$", ha="center", va="bottom")
plt.text(R4, 0.93, r"$r_4$", ha="center", va="bottom")

plt.xlabel("r")
plt.ylabel("x")
plt.ylim(0.3, 0.95)
plt.xlim(2.95, 3.7)
plt.show()

plt.figure(figsize=(12, 6))
for r in r_values:
    x = np.random.random()
    for _ in range(1000):
        x = r*x*(1 - x)

    lyap_sum = 0
    for _ in range(100):
        x = r*x*(1 - x)
        lyap_sum += np.log(abs(r*(1 - 2*x)))

    lyap = lyap_sum / 100
    plt.plot(r, lyap, ",", color='#EB5E00', alpha=1)

plt.axhline(0, color="blue", linestyle="--", linewidth=0.5)
plt.axvline(3.56994, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro, color="blue", linestyle="--", linewidth=0.8)
plt.axvline(ro2, color="blue", linestyle="--", linewidth=0.8)

# ---- Period-Three and the Link to Sharkovsky's Theorem

r_values = np.linspace(2.95, 4, 15000) # Parameters
plt.figure(figsize=(12, 6))

for r in r_values:
    x = np.random.random()
    for i in range(1000): # Burn-in Period
        x = r*x*(1-x)
    X, Y = [], []
    for i in range(100):
        x = r * x * (1 - x)
        X.append(r)
        Y.append(x)
    plt.plot(X, Y, ",", color='#EB5E00', alpha = 0.25)

plt.axvline(3.56994, color="blue", linestyle="--", linewidth=0.8)
plt.xlabel("r")
plt.ylabel("x")
plt.ylim(0, 1)
plt.xlim(2.95, 4)
plt.show()

P = np.linspace(3.82, 3.86, 10000)  # Further Zoom
X = []
Y = []

for x in P:
    r = 0.5
    for n in range(1000): # Burn-in Period
        r = (x * r) * (1 - r)
    for l in range(100):
        r = (x * r) * (1 - r)
        X.append(x)
        Y.append(r)

plt.figure(figsize=(4, 6))
plt.plot(X, Y, ',', color='#EB5E00', alpha = 0.25) 

plt.xlabel("r")
plt.ylabel("x")
plt.ylim(0.1, 1)
plt.xlim(3.82, 3.86)
plt.show()

P = np.linspace(3.84, 3.856, 10000)  # Further Zoom
X = []
Y = []

for x in P:
    r = 0.5
    for n in range(1000): # Burn-in Period
        r = (x * r) * (1 - r)
    for l in range(100):
        r = (x * r) * (1 - r)
        X.append(x)
        Y.append(r)

plt.figure(figsize=(10, 6))
plt.plot(X, Y, ',', color='#EB5E00', alpha = 0.25) 
plt.xlabel("r")
plt.ylabel("x")
plt.ylim(0.44, 0.56)
plt.xlim(3.84, 3.856)
plt.show()