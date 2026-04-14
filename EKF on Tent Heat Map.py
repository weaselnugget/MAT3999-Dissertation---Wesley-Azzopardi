import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
seed = 3132004 # Best chosen seed
rng = np.random.default_rng(seed)

def tent_map(x, a): # Function for Tent
    return a * np.minimum(x, 1 - x)

sigma_Q = 0.5  # Controls Variability of Q_k
def ekf_tent_a_estimation(a_true, a_guess, rng, n_steps=100, Q=1e-5, x0_true=0.7, x0_hat=0.7):
    t0 = time.perf_counter()
    Q_seq = Q * np.exp(sigma_Q * rng.normal(size = n_steps)) 
    x_true = np.zeros(n_steps)
    y_meas = np.zeros(n_steps)
    x_true[0] = x0_true
    y_meas[0] = x_true[0] + rng.normal(0, np.sqrt(Q_seq[0]))
    for k in range(1, n_steps):
        x_true[k] = tent_map(x_true[k-1], a_true)
        y_meas[k] = x_true[k] + rng.normal(0, np.sqrt(Q_seq[k]))
    
    x_hat = np.zeros(n_steps)
    a_hat = np.zeros(n_steps)
    x_hat[0], a_hat[0] = x0_hat, a_guess
    P = np.eye(2)
    C = np.array([[1, 0]])

    for k in range(1, n_steps):
        x_prev, a_prev = x_hat[k-1], a_hat[k-1]
        x_pred = tent_map(x_prev, a_prev)
        a_pred = a_prev
        s_pred = np.array([[x_pred], [a_pred]])

        if x_prev < 0.5:
            dphidx = a_prev
            dphida = x_prev 
        elif x_prev > 0.5:
            dphidx = -a_prev
            dphida = 1 - x_prev
        else:
            dphidx = 0
            dphida = 0.5

        Phi = np.array([[dphidx, dphida],
                        [0,   1]])
        P_pred = Phi @ P @ Phi.T

        y_pred = (C @ s_pred)[0, 0]
        innovation = y_meas[k] - y_pred
        S = (C @ P_pred @ C.T)[0, 0] + Q_seq[k]

        K = (P_pred @ C.T) / S
        s_upd = s_pred + K * innovation
        P = (np.eye(2) - K @ C) @ P_pred

        s_upd = np.asarray(s_upd).reshape(-1)
        x_hat[k], a_hat[k] = s_upd[0], s_upd[1]
    
    t1 = time.perf_counter()
    runtime = t1 - t0
    return a_hat[-1], runtime

a_true_list = [0.5, 0.8, 1.1, 1.4, 1.7, 2-1e-8] # Run Experiment
Q_list = np.logspace(-7, -2, 11, base = 10)
a_guess = 1
n_steps = 100
heat = np.zeros((len(a_true_list), len(Q_list)))
times = np.zeros_like(heat)

for i, a_true in enumerate(a_true_list): # Create Grid
    for j, Q in enumerate(Q_list):
        a_final, runtime = ekf_tent_a_estimation(a_true, a_guess, rng, n_steps, Q)
        times[i, j] = runtime
        heat[i, j] = abs(a_true - a_final)

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
ax.set_title('EKF Accuracy a vs Q', fontsize=26)
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
ax2.set_title("EKF Runtime vs Q", fontsize=18)
ax2.legend(fontsize=18)

plt.tight_layout()
plt.show()