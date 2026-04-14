import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

# ----- Lyapunov Stability -----

g = 9.806685  # Gravity
L = 2   # Length of Pendulum
omega0 = np.sqrt(g / L) # Angular Frequency

def pendulum(t, z): # ODE for Simple Pendulum
    x, y = z
    dxdt = y
    dydt = -np.square(omega0) * np.sin(x)
    return [dxdt, dydt]

x_vals = np.linspace(-np.pi, np.pi, 40) # Grid
y_vals = np.linspace(-5, 5, 40)
X, Y = np.meshgrid(x_vals, y_vals)
U = Y
V = -np.square(omega0) * np.sin(X)

initial_conditions = [[0.5, -0.5]] # Initial Conditions near origin (x=0)

t_span = [0, 10] # Length of x-axis
t_eval = np.linspace(*t_span, 1000)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].streamplot(X, Y, U, V, density=1, color='black') # Phase-Portrait
for ic in initial_conditions:
    sol = solve_ivp(pendulum, t_span, ic, t_eval=t_eval)
    axs[0].plot(sol.y[0], sol.y[1], color = 'red')
axs[0].set_xlabel('θ', fontsize=20)
axs[0].set_ylabel("θ'", fontsize=20)
axs[0].legend()
axs[0].grid(True)

for ic in initial_conditions: # Time Simulation
    sol = solve_ivp(pendulum, t_span, ic, t_eval=t_eval)
    axs[1].plot(t_eval, sol.y[0], color = 'red')
axs[1].set_xlabel('t', fontsize=20)
axs[1].set_ylabel('θ',fontsize=20)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# ---- Asymptotic Stability -----

drag = 0.5

def pendulum(t, z): # ODE for Damped Pendulum
    x, y = z
    dxdt = y
    dydt = -drag*y -np.square(omega0) * np.sin(x)
    return [dxdt, dydt]

x_vals = np.linspace(-np.pi, np.pi, 40) # Grid
y_vals = np.linspace(-5, 5, 40)
X, Y = np.meshgrid(x_vals, y_vals)
U = Y
V = -drag*Y -np.square(omega0) * np.sin(X)

initial_conditions = [[0.5, -0.5]] # Initial Conditions near origin (x=0)

t_span = [0, 50] # Length of x-axis
t_eval = np.linspace(*t_span, 1000)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].streamplot(X, Y, U, V, density=1, color='black') # Phase-Portrait
for ic in initial_conditions:
    sol = solve_ivp(pendulum, t_span, ic, t_eval=t_eval)
    axs[0].plot(sol.y[0], sol.y[1], color = 'red')
axs[0].set_xlabel('θ')
axs[0].set_ylabel("θ'")
axs[0].legend()
axs[0].grid(True)

for ic in initial_conditions: # Time Simulation
    sol = solve_ivp(pendulum, t_span, ic, t_eval=t_eval)
    axs[1].plot(t_eval, sol.y[0], color = 'red')
axs[1].set_xlabel('t')
axs[1].set_ylabel('θ')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()