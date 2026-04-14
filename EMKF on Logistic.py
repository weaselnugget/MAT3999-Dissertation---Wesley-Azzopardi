import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

plt.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 24
})

# -------------------------------------------------
# SYNTHETIC LOGISTIC DATA
# -------------------------------------------------

def logistic_true(r, x):
    return r*x*(1-x)

def logistic_map(theta1, theta2, x):
    return theta1*x + theta2*x**2


n_steps = 100
true_r = 4

x = np.zeros(n_steps+1)
x[0] = 0.4

for t in range(n_steps):
    x[t+1] = logistic_true(true_r, x[t])

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

split = int(0.8 * len(x))

# -------------------------------------------------
# TRAIN-ONLY STANDARDISATION
# -------------------------------------------------

x_train = x[:split+1]

xmin = x_train.min()
xmax = x_train.max()
scale = xmax - xmin

x_std = (x - xmin) / scale

# -------------------------------------------------
# BUILD STATE SPACE MODEL
# -------------------------------------------------

y = x_std[1:]
x_prev = x_std[:-1]

C = np.zeros((len(y),1,2))
C[:,0,0] = x_prev
C[:,0,1] = x_prev**2

y_train = y[:split-1]
y_test  = y[split-1:]

C_train = C[:split-1]
C_test  = C[split-1:]

# -------------------------------------------------
# INITIALISE KF + EM
# -------------------------------------------------

kf = KalmanFilter(
    transition_matrices=np.eye(2),
    observation_matrices=C_train,
    transition_covariance=1e-4*np.eye(2),
    observation_covariance=1e-4,
    initial_state_mean=[1,1],
    initial_state_covariance=np.eye(2)
)

kf = kf.em(
    y_train,
    n_iter=300,
    em_vars=[
        'transition_matrices',
        'transition_covariance',
        'observation_covariance'
    ]
)

# -------------------------------------------------
# SMOOTH TRAINING
# -------------------------------------------------

xi_train_s, P_train_s = kf.smooth(y_train)

std_s = np.sqrt([np.diag(P) for P in P_train_s])

# -------------------------------------------------
# FILTER FULL SAMPLE
# -------------------------------------------------

kf_full = KalmanFilter(
    transition_matrices=kf.transition_matrices,
    transition_covariance=kf.transition_covariance,
    observation_covariance=kf.observation_covariance,
    observation_matrices=C,
    initial_state_mean=xi_train_s[0],
    initial_state_covariance=P_train_s[0]
)

xi_full, P_full = kf_full.filter(y)

# -------------------------------------------------
# FULL RECONSTRUCTION
# -------------------------------------------------

x_kf = np.zeros(len(x_std))
x_low = np.zeros(split)
x_high = np.zeros(split)

x_kf[0] = x_std[0]
x_low[0] = x_std[0]
x_high[0] = x_std[0]

for t in range(len(x_std)-1):

    r = xi_full[t]

    x_kf[t+1] = logistic_map(
        r[0], r[1], x_kf[t]
    )

# de-standardise

x_kf_real = scale*x_kf + xmin






steps = 5 # 5-Step Forecasting

x_fore = np.zeros(steps+1)


x_forecast = np.zeros((steps, 2))
P_forecast = np.zeros((steps, 2, 2))

x_forecast[0] = xi_train_s[-1]
P_forecast[0] = P_train_s[-1]

for t in range(1, steps):
    x_prev = x_forecast[t-1]
    x_forecast[t] = kf.transition_matrices @ x_prev



x_model_fore = np.zeros(steps)
x_model_fore[0] = x_kf[-1]   # last standardized KF reconstruction


for t in range(steps-1):
    th1_t, th2_t = x_forecast[t]
    x_model_fore[t+1] = th1_t*x_model_fore[t] + th2_t*x_model_fore[t]**2


x_true_future = np.zeros(steps) # True future for comparison
x_true_future[0] = x[-1] # Pick up from last point
for i in range(steps-1):
    x_true_future[i+1] = logistic_true(true_r, x_true_future[i])


# -------------------------------------------------
# DIAGNOSTICS
# -------------------------------------------------

err = x - x_kf_real

MSE = np.mean(err**2)
RMSE = np.sqrt(MSE)

SS_res = np.sum(err**2)
SS_tot = np.sum((x - np.mean(x))**2)

R2 = 1 - SS_res/SS_tot

print("\nReconstruction Diagnostics")
print("RMSE:", RMSE)
print("R^2:", R2)

# -------------------------------------------------
# PLOTS
# -------------------------------------------------

t = np.arange(len(x))

plt.figure(figsize=(15,5))

plt.plot(t, x, color="blue")
plt.plot(t, x_kf_real, "--", color="orange")

plt.axvline(split+1, linestyle=':', lw=5, color="purple")
plt.axvline(n_steps, linestyle=':', lw=5, color="red")

plt.plot(np.arange(n_steps, n_steps+steps), x_model_fore, '--', color = "orange")
plt.plot(np.arange(n_steps, n_steps+steps), x_true_future, ':', color = "blue")