import numpy as np
import matplotlib.pyplot as plt
import time

seed = 3132004 # Best chosen seed
rng = np.random.default_rng(seed)

plt.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16})

def tent_map(x, a):
    return a * np.minimum(x, 1 - x)

sigma_Q = 0.5  # controls variability of Q_k
def ukf_tent_a_estimation(a_true, a_guess, rng, n_steps=100, Q=1e-5, x0_true=0.7, x0_hat=0.7, alpha=1e-3, beta=2, kappa=0):
    t0 = time.perf_counter()
    Q_seq = Q * np.exp(sigma_Q * rng.normal(size = n_steps))
    n = 2
    lam = alpha**2 * (n + kappa) - n # Lambda substitution
    c = n + lam # Denominator of w
    Wm = np.full(2*n + 1, 1/(2*c)) # For Mean
    Wc = np.full(2*n + 1, 1/(2*c)) # For Variance
    Wm[0] = lam/c
    Wc[0] = (-alpha**2 + beta) + lam/c

    x_true = np.zeros(n_steps)
    y_meas = np.zeros(n_steps)
    x_true[0] = x0_true
    y_meas[0] = x_true[0] + rng.normal(0, np.sqrt(Q_seq[0]))
    for k in range(1, n_steps):
        x_true[k] = tent_map(x_true[k-1], a_true)
        y_meas[k] = x_true[k] + rng.normal(0, np.sqrt(Q_seq[k]))

    x_hat = np.zeros(n_steps)
    a_hat = np.zeros(n_steps)
    x_hat[0] = x0_hat
    a_hat[0] = a_guess
    P = np.eye(n)# Initial Covariance

    def sigma_points(mu, P): # Sigma Point Generator
        A = np.linalg.cholesky(c * P)
        SP = np.zeros((2*n + 1, n))
        SP[0] = mu.copy()
        for i in range(n):
            SP[i + 1] = mu + A[:, i]
            SP[n + i + 1] = mu - A[:, i]
        return SP

    for k in range(1, n_steps): # UKF Loop
        mu = np.array([x_hat[k-1], a_hat[k-1]])
        
        X = sigma_points(mu, P) # Form Sigma Points

        X_pred = np.zeros_like(X) # Propagate SP through Phi
        for i in range(2 * n + 1):
            x_i, a_i = X[i]
            x_next = tent_map(x_i, a_i)
            a_next = a_i  # Assume constant a
            X_pred[i] = np.array([x_next, a_next])

        mu_pred = np.sum(Wm[:, None] * X_pred, axis=0) # Predicted Mean

        P_pred = np.zeros((n, n)) # Predicted Covariance
        for i in range(2 * n + 1):
            d = (X_pred[i] - mu_pred).reshape(-1, 1)
            P_pred += Wc[i] * (d @ d.T)

        Y_sigma = np.zeros(2*n + 1) # Update Stage
        for i in range(2*n + 1):
            Y_sigma[i] = X_pred[i, 0]

        y_pred = np.sum(Wm * Y_sigma) # Measurement for Mean

        S = 0 # Measurement for Variance Covariance
        for i in range(2*n + 1):
            dy = Y_sigma[i] - y_pred
            S += Wc[i] * (dy*dy)
        S += Q_seq[k] # Constant Measurement Noise

        P_xy = np.zeros((n, 1)) # Cross-Covariance between State & Measurement
        for i in range(2 * n + 1):
            dx = (X_pred[i] - mu_pred).reshape(-1, 1)
            dy = (Y_sigma[i] - y_pred)
            P_xy += Wc[i] * (dx*dy)

        K = P_xy/S
        innovation = y_meas[k] - y_pred
        mu_upd = mu_pred + (K.flatten() * innovation)
        P_upd = P_pred - (K @ K.T) * S

        x_hat[k] = mu_upd[0]
        a_hat[k] = mu_upd[1]
        P = P_upd
        t1 = time.perf_counter()
        runtime = t1 - t0
    return a_hat[-1], runtime

a_true_list = [0.5, 0.8, 1.1, 1.4, 1.7, 2-1e-8] # Run Experiment
Q_list = np.logspace(-7, -2, 11, base = 10)
a_guess = 1
n_steps = 100
heat = np.zeros((len(a_true_list), len(Q_list)))
times = np.zeros_like(heat)

for m, a_true in enumerate(a_true_list): # Create Grid
    for j, Q in enumerate(Q_list):
        a_final, runtime = ukf_tent_a_estimation(a_true, a_guess, rng, n_steps, Q)
        times[m, j] = runtime
        heat[m, j] = abs(a_true - a_final)

heat_normalized = (heat / np.max(heat)) # Normalize Metrics
times_normalized = times / np.max(times)
times_normalized = (times_normalized - np.min(times_normalized))
times_normalized /= np.max(times_normalized)
times_normalized = 0.2 + 0.8 * times_normalized

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(heat_normalized, cmap='OrRd', origin='lower')
exponents = np.linspace(-7, -2, len(Q_list))

ax.set_xticks(np.arange(len(Q_list)))
ax.set_xticklabels([f'{e:g}' for e in exponents])
ax.set_yticks(np.arange(len(a_true_list)))
ax.set_yticklabels([f'{r:.2f}' for r in a_true_list])
ax.tick_params(axis='both', which='major', labelsize=26)
ax.set_xlabel('log10(Q) - Measurement Noise', fontsize=26)
ax.set_ylabel('True a', fontsize=26)
ax.set_title('UKF Accuracy a vs Q', fontsize=26)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 6))
for i, a_true in enumerate(a_true_list):
    ax2.plot(Q_list, times[i], marker='o', label=f"a={a_true:.1f}")

ax2.set_xscale('log')
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_xlabel("Q - Measurement Noise", fontsize=18)
ax2.set_ylabel("Runtime (s)", fontsize=18)
ax2.set_title("UKF Runtime vs Q", fontsize=18)
ax2.legend(fontsize=18)
plt.tight_layout()
plt.show()