import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import warnings
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize # 3D Required Packages

warnings.filterwarnings('ignore')

seed = 67 # Best chosen seed
rng = np.random.default_rng(seed)

sigma_Q = 0.5  # controls variability of Q_k
def ZS_map(x, y, a, b):
    return -(a*x)/(1+y**2), x+b*y

def ukf_ZS_estimation(a_true, a_guess, b_true, b_guess, rng, n_steps=100, Q_error=1e-5, x0_true=0.1, x0_hat=0.1, y0_true = 0.1, y0_hat = 0.1, meas_noise=None, alpha=1e-3, beta=2, kappa=0):
    t0 = time.perf_counter()
    n = 4
    lam = alpha**2 * (n + kappa) - n # Lambda substitution
    c = n + lam # Denominator of w
    Wm = np.full(2*n + 1, 1/(2*c)) # For Mean
    Wc = np.full(2*n + 1, 1/(2*c)) # For Variance
    Wm[0] = lam/c
    Wc[0] = (-alpha**2 + beta) + lam/c
    
    if meas_noise is None:
        Q_seq = Q_error * np.exp(sigma_Q * rng.normal(size = n_steps)) 
        x_true = np.zeros(n_steps)
        y_true = np.zeros(n_steps)
        x_meas = np.zeros(n_steps)
        y_meas = np.zeros(n_steps)
        x_true[0] = x0_true
        y_true[0] = y0_true
        for k in range(1, n_steps):
            x_true[k], y_true[k] = ZS_map(x_true[k-1], y_true[k-1], a_true, b_true)
            x_meas[k] = x_true[k] + rng.normal(0, np.sqrt(Q_seq[k]))
            y_meas[k] = y_true[k] + rng.normal(0, np.sqrt(Q_seq[k]))
    else:
        x_true = None # If data does already exist, just use that
        y_true = None

    x_hat = np.zeros(n_steps) # UKF Setup
    y_hat = np.zeros(n_steps)
    a_hat = np.zeros(n_steps)
    b_hat = np.zeros(n_steps)
    x_hat[0] = x0_hat
    y_hat[0] = y0_hat
    a_hat[0] = a_guess    
    b_hat[0] = b_guess
    P = np.eye(n)

    def sigma_points(mu, P):
        U, s, _ = np.linalg.svd(c * P, full_matrices=False)
        s = np.maximum(s, 1e-12)
        A = U @ np.diag(np.sqrt(s))
        SP = np.zeros((2*n + 1, n))
        SP[0] = mu
        
        for i in range(n):
            SP[i + 1] = mu + A[:, i]
            SP[n + i + 1] = mu - A[:, i]
        return SP

    
    for k in range(1, n_steps): # UKF Loop
        mu = np.array([x_hat[k-1], y_hat[k-1], a_hat[k-1], b_hat[k-1]])

        X = sigma_points(mu, P) # Form Sigma Points

        X_pred = np.zeros_like(X) # Propagate SP through Phi
        for i in range(2 * n + 1):
            x_i, y_i, a_i, b_i = X[i]
            x_next, y_next = ZS_map(x_i, y_i, a_i, b_i)
            a_next, b_next = a_i, b_i  # Assume constant a,b (only info on x,y)
            X_pred[i] = np.array([x_next, y_next, a_next, b_next])

        mu_pred = np.sum(Wm[:, None] * X_pred, axis=0) # Predict Mean
        
        P_pred = np.zeros((n, n)) # Predict Covariance
        for i in range(2 * n + 1):
            d = (X_pred[i] - mu_pred).reshape(-1, 1)
            P_pred += Wc[i] * (d @ d.T)
        P_pred += 1e-6 * np.eye(n)

        m = 2 # Update Stage
        Y_sigma = np.zeros((2*n + 1, m))
        for i in range(2*n + 1):
            Y_sigma[i] = X_pred[i, 0:2]

        y_pred = np.sum(Wm[:, None] * Y_sigma, axis=0) # Measurment for Mean

        S = np.zeros((m, m)) # Measurement for Variance Covariance
        for i in range(2*n + 1):
            dy = (Y_sigma[i] - y_pred).reshape(-1, 1)
            S += Wc[i] * (dy @ dy.T)
        S += Q_seq[k] # Constant Measurement Noise

        P_xy = np.zeros((n, m)) # Cross-Covariance between State & Measurement
        for i in range(2 * n + 1):
            dx = (X_pred[i] - mu_pred).reshape(-1, 1)
            dy = (Y_sigma[i] - y_pred)
            P_xy += Wc[i] * (dx*dy)

        K = P_xy @ np.linalg.inv(S)
        z_k = np.array([x_meas[k], y_meas[k]])
        innovation = z_k - y_pred
        mu_upd = mu_pred + K @ innovation
        P_upd = P_pred - K @ S @ K.T
        x_hat[k], y_hat[k], a_hat[k], b_hat[k] = mu_upd[0], mu_upd[1], mu_upd[2], mu_upd[3]
        P = 0.5 * (P_upd + P_upd.T)
        P[np.isnan(P)] = 0.0
        P[np.isinf(P)] = 0.0
    t1 = time.perf_counter()        
    runtime = t1-t0
    return a_hat[-1], b_hat[-1], runtime

a_true_list = [-6, -3.6, -1.2, 1.2, 3.6, 6] # Run Experiment
b_true_list = [-1.2, -0.72, -0.24, 0.24, 0.72, 1.2]
Q_list = np.logspace(-7, -2, 11, base = 10)
a_guess, b_guess = 1, 0.1
n_steps = 100

err_a = np.zeros((len(a_true_list), len(b_true_list), len(Q_list)))
err_b = np.zeros((len(a_true_list), len(b_true_list), len(Q_list)))
times = np.zeros((len(a_true_list), len(b_true_list), len(Q_list)))

for m, a_true in enumerate(a_true_list): # Create Grid
    for n, b_true in enumerate(b_true_list):
        for o, Q_error in enumerate(Q_list):
            a_final, b_final, runtime = ukf_ZS_estimation(a_true=a_true, b_true=b_true, a_guess=a_guess, b_guess=b_guess, rng=rng, n_steps=n_steps, Q_error=Q_error)
            times[m, n, o] = runtime
            err_a[m, n, o] = abs(a_true - a_final)
            err_b[m, n, o] = abs(b_true - b_final)

Na = len(a_true_list)
Nb = len(b_true_list)
Nq = len(Q_list)
Q_log = np.log10(Q_list)
Zc = np.zeros(Nq + 1)
Zc[1:] = Q_log
filled = np.ones((Na, Nb, Nq), dtype=bool)
X, Y, Z = np.meshgrid(
    np.arange(Na + 1),
    np.arange(Nb + 1),
    Zc,
    indexing="ij")
exponents = np.linspace(-7, -2, len(Q_list))

a_vals = np.array(a_true_list)
b_vals = np.array(b_true_list)
Q_log = np.log10(Q_list)
A, B = np.meshgrid(a_vals, b_vals, indexing="ij")

E_a = err_a / np.nanmax(err_a) # For Parameter a

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(projection="3d")
norm = Normalize(vmin=E_a.min(), vmax=E_a.max())
sm = ScalarMappable(norm=norm, cmap='OrRd')
sm.set_array([]) # For Color Bar

ax.set_box_aspect((3, 3, 4))  # Elongate Z for better view
norm = Normalize(vmin=E_a.min(), vmax=E_a.max())

for k, q in enumerate(Q_log):
    Z = k * np.ones_like(A)
    ax.plot_surface(
        A, B, Z,
        facecolors=cm.OrRd(E_a[:, :, k]),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=True,
        linewidth=0.4,
        edgecolor="k")

ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{q:g}' for q in Q_log])

ax.set_xlabel("a", fontsize=20, labelpad=12)
ax.set_ylabel("b", fontsize=20, labelpad=12)
ax.set_zlabel("log10(Q)", fontsize=20, labelpad=12)

ax.set_xticks(a_vals)
ax.set_yticks(b_vals)
ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{e:g}' for e in exponents])

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='z', which='major', labelsize=18)
fig.colorbar(sm, ax=ax)
ax.set_title('UKF Accuracy a vs Q', fontsize = 40)
ax.view_init(elev=18, azim=45)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 14)) # Rotated
ax = fig.add_subplot(projection="3d")
ax.set_box_aspect((3, 3, 4))  # Elongate Z for better view

for k, q in enumerate(Q_log):
    Z = k * np.ones_like(A)
    ax.plot_surface(
        A, B, Z,
        facecolors=cm.OrRd(E_a[:, :, k]),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=True,
        linewidth=0.4,
        edgecolor="k")

ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{q:g}' for q in Q_log])

ax.set_xlabel("a", fontsize=20, labelpad=12)
ax.set_ylabel("b", fontsize=20, labelpad=12)
ax.set_zlabel("log10(Q)", fontsize=20, labelpad=12)

ax.set_xticks(a_vals)
ax.set_yticks(b_vals)
ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{e:g}' for e in exponents])

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='z', which='major', labelsize=18)
fig.colorbar(sm, ax=ax)
ax.set_title('UKF Accuracy a vs Q', fontsize = 40)
ax.view_init(elev=18, azim=315)
plt.tight_layout()
plt.show()

E_b = err_b / np.nanmax(err_b) # For Parameter b

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(projection="3d")
norm = Normalize(vmin=E_b.min(), vmax=E_b.max())
sm = ScalarMappable(norm=norm, cmap='OrRd')
sm.set_array([]) # For Color Bar

ax.set_box_aspect((3, 3, 4))  # Elongate Z for better view
for k, q in enumerate(Q_log):
    Z = k * np.ones_like(B)
    ax.plot_surface(
        A, B, Z,
        facecolors=cm.OrRd(E_b[:, :, k]),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=True,
        linewidth=0.4,
        edgecolor="k")

ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{q:g}' for q in Q_log])

ax.set_xlabel("a", fontsize=20, labelpad=12)
ax.set_ylabel("b", fontsize=20, labelpad=12)
ax.set_zlabel("log10(Q)", fontsize=20, labelpad=12)

ax.set_xticks(a_vals)
ax.set_yticks(b_vals)
ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{e:g}' for e in exponents])

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='z', which='major', labelsize=18)
fig.colorbar(sm, ax=ax)
ax.set_title('UKF Accuracy b vs Q', fontsize = 40)
ax.view_init(elev=18, azim=45)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 14)) # Rotated
ax = fig.add_subplot(projection="3d")
ax.set_box_aspect((3, 3, 4))  # Elongate Z for better view

for k, q in enumerate(Q_log):
    Z = k * np.ones_like(B)
    ax.plot_surface(
        A, B, Z,
        facecolors=cm.OrRd(E_b[:, :, k]),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=True,
        linewidth=0.4,
        edgecolor="k")

ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{q:g}' for q in Q_log])

ax.set_xlabel("a", fontsize=20, labelpad=12)
ax.set_ylabel("b", fontsize=20, labelpad=12)
ax.set_zlabel("log10(Q)", fontsize=20, labelpad=12)

ax.set_xticks(a_vals)
ax.set_yticks(b_vals)
ax.set_zticks(np.arange(len(Q_list)))
ax.set_zticklabels([f'{e:g}' for e in exponents])

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='z', which='major', labelsize=18)
fig.colorbar(sm, ax=ax)
ax.set_title('UKF Accuracy b vs Q', fontsize = 40)
ax.view_init(elev=18, azim=315)
plt.tight_layout()
plt.show()