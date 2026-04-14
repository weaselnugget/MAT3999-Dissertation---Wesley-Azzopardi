import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import plotly.graph_objects as go
import sys

plt.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 24,
})

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

file_type = input("Crypto or Commodity? ")
column_name_1 = input("Price Asset #1: ")
column_name_2 = input("Price Asset #2: ")
max_steps = int(input("Number of N-Steps Ahead (max 7) "))
if (max_steps > 7 or max_steps < 0):
    sys.exit("Exiting... Please enter a value between 1 and 7!")
    
max_steps = max_steps+1 # Must include also the endpoint

file = pd.read_csv("E:/Thesis/Codes/Application/" + str.lower(file_type) + "_prices.csv")
file['Date'] = pd.to_datetime(file['Date'])

file_raw = file[file['Date'] <= "2025-12-31"]
file_ahead = file[file['Date'] > "2025-12-31"]

x_1_raw = pd.to_numeric(file_raw[column_name_1]).values
x_2_raw = pd.to_numeric(file_raw[column_name_2]).values
mask = np.isfinite(x_1_raw) & np.isfinite(x_2_raw) # Joint Mask
x_1_raw = x_1_raw[mask]
x_2_raw = x_2_raw[mask]

# -------------------------------------------------
# TRAIN / TEST SPLIT (80%)
# -------------------------------------------------

split = int(0.8 * len(x_1_raw))

# -------------------------------------------------
# STANDARDISE USING TRAINING ONLY
# -------------------------------------------------

mu_1, mu_2 = x_1_raw[:split].mean(), x_2_raw[:split].mean()
sigma_1, sigma_2 = x_1_raw[:split].std(),x_2_raw[:split].std()

x_1_std = (x_1_raw - mu_1) / sigma_1
x_2_std = (x_2_raw - mu_2) / sigma_2

# -------------------------------------------------
# BUILD STATE SPACE MODEL
# -------------------------------------------------

def henon_map(xi, x, y):
    a, b = xi
    xn = 1 - a*x**2 + y
    yn = b*x
    return xn, yn


x_1_prev = x_1_std[:-1]
x_2_prev = x_2_std[:-1]

x_1_next = x_1_std[1:]
x_2_next = x_2_std[1:]

Y = np.column_stack([x_1_next, x_2_next]) # Store all measurements under each other.


T = len(x_1_prev)
C = np.zeros((T, 2, 2))

C[:,0,0] = x_1_prev**2
C[:,1,1] = x_1_prev

Y_adj = Y.copy()
Y_adj[:,0] = Y_adj[:,0] - x_2_prev # y_k is just a scaled linear transform of x_k


Y_train, Y_test = Y_adj[:split], Y_adj[split:]
C_train, C_test = C[:split], C[split:]






Y_train, Y_test = Y[:split], Y[split:]
C_train, C_test = C[:split], C[split:]


# -------------------------------------------------
# INITIALISE + EM
# -------------------------------------------------

kf = KalmanFilter( # Initial guesses
    transition_matrices=np.eye(2),
    observation_matrices=C_train,
    transition_covariance=1e-4*np.eye(2),
    observation_covariance=1e-4*np.eye(2),
    initial_state_mean=[1,1],
    initial_state_covariance=np.eye(2))


kf = kf.em( # EM Algorithm to learn parameters
    Y_train,
    n_iter=10,
    em_vars=[
        'observation_covariance', 'transition_covariance']) # No need to write down the optimal KF parameters (too long)

# -------------------------------------------------
# SMOOTH TRAINING
# -------------------------------------------------

xi_train, P_train = kf.smooth(Y_train)
std_s = np.sqrt([np.diag(P) for P in P_train])
lower = xi_train - 1.96*std_s
upper = xi_train + 1.96*std_s

# -------------------------------------------------
# FILTER FULL SAMPLE (FOR RECONSTRUCTION)
# -------------------------------------------------

kf_test = KalmanFilter(
    transition_matrices=kf.transition_matrices,
    transition_covariance=kf.transition_covariance,
    observation_covariance=kf.observation_covariance,
    observation_matrices=C_test,
    initial_state_mean=xi_train[-1],
    initial_state_covariance=P_train[-1]
)

xi_test, P_test = kf_test.filter(Y_test)

xi_full = np.vstack([xi_train, xi_test]) # Xi_1 and Xi_2

# -------------------------------------------------
# FULL RECONSTRUCTION
# -------------------------------------------------

x_1_kf, x_2_kf = np.zeros(len(x_1_std)), np.zeros(len(x_2_std))
x_1_low, x_2_low = np.zeros(split), np.zeros(split)
x_1_high, x_2_high = np.zeros(split), np.zeros(split)


x_1_kf[0] = x_1_std[0]
x_1_low[0], x_2_low[0] = x_1_std[0], x_2_std[0]
x_1_high[0], x_2_high[0] = x_1_std[0], x_2_std[0]
    
for t in range(split-1):
    x_1_low[t+1], x_2_low[t+1] = np.clip(henon_map(lower[t], x_1_kf[t], x_2_kf[t]), -5, 5)
    x_1_high[t+1], x_2_high[t+1] = np.clip(henon_map(upper[t], x_1_kf[t], x_2_kf[t]), -5, 5)
  
for t in range(split-1):
    x_1_kf[t+1], x_2_kf[t+1] = henon_map(xi_full[t], x_1_kf[t], x_2_kf[t])

x_1_kf_real, x_2_kf_real = sigma_1 * x_1_kf + mu_1, sigma_2 * x_2_kf + mu_2
x_1_low_real, x_2_low_real = sigma_1 * x_1_low + mu_1, sigma_2 * x_2_low + mu_2
x_1_high_real, x_2_high_real = sigma_1 * x_1_high + mu_1, sigma_2 * x_2_high + mu_2
x_1_real, x_2_real = x_1_raw, x_2_raw

# -------------------------------------------------
# PREPARE FUTURE OBSERVATIONS
# -------------------------------------------------

x_1_ahead = pd.to_numeric(file_ahead[column_name_1]).values
x_2_ahead = pd.to_numeric(file_ahead[column_name_2]).values
mask_1 = np.isfinite(x_1_ahead)
x_1_ahead = x_1_ahead[mask_1]
mask_2 = np.isfinite(x_2_ahead)
x_2_ahead = x_2_ahead[mask_2]

x_1_std_ahead = (x_1_ahead - mu_1) / sigma_1
x_2_std_ahead = (x_2_ahead - mu_2) / sigma_2

y_1_full = np.concatenate([x_1_next, x_1_std_ahead[:max_steps]])
y_2_full = np.concatenate([x_2_next, x_2_std_ahead[:max_steps]])

x_1_prev_full = np.concatenate([x_1_std[:-1], x_1_std[-1:], x_1_std_ahead[:max_steps-1]])
x_2_prev_full = np.concatenate([x_2_std[:-1], x_2_std[-1:], x_2_std_ahead[:max_steps-1]])

Phi_full = -x_1_prev_full / (1 + x_2_prev_full**2)

C_full = np.zeros((len(Phi_full), 2, 3))

C_full[:,0,0] = 1
C_full[:,0,1] = Phi_full
C_full[:,0,2] = 0 # x equation: c + a*Phi

C_full[:,1,0] = 0 # y equation: b*y
C_full[:,1,1] = 0
C_full[:,1,2] = x_2_prev_full



T = len(x_1_prev_full)
C_full = np.zeros((T, 2, 2))

C_full[:,0,0] = x_1_prev_full**2
C_full[:,1,1] = x_1_prev_full

dates_raw = file_raw['Date']
dates = np.concatenate([dates_raw, file_ahead['Date'].values[:max_steps]])

values_window_1 = x_1_real.max() - x_1_real.min()
values_window_2 = x_2_real.max() - x_2_real.min() # To measure errors at the end as a percentage

# -------------------------------------------------
# ROLLING FORECASTS 1 → max_steps
# -------------------------------------------------

forecast_1_dict = {}
forecast_2_dict = {}
rmse_dict = {}

A = kf.transition_matrices
Q = kf.transition_covariance

# One-Step ahead (normal filter)

x_1_fore_std = np.full(len(x_1_std), np.nan)
x_2_fore_std = np.full(len(x_2_std), np.nan)

xi_current = xi_train[-1]
P_current  = P_train[-1]
x_1_state = x_1_std[split-1]
x_2_state = x_2_std[split-1]

for t in range(split, len(y_1_full) - max_steps):

    # ---- Parameter prediction
    xi_pred = A @ xi_current
    P_pred  = A @ P_current @ A.T + Q

    # ---- Nonlinear 1-step propagation
    x_1_temp, x_2_temp = henon_map(xi_pred, x_1_state, x_2_state)

    x_1_fore_std[t+1] = x_1_temp
    x_2_fore_std[t+1] = x_2_temp

    # ---- Update with observation
    xi_current, P_current = kf.filter_update(
        xi_current,
        P_current,
        observation=np.array([y_1_full[t], y_2_full[t]]),
        observation_matrix=C_full[t]
    )

    x_1_state = y_1_full[t]
    x_2_state = y_2_full[t]

x_1_fore_real = sigma_1 * x_1_fore_std + mu_1
x_2_fore_real = sigma_2 * x_2_fore_std + mu_2

x_1_fore_real_aligned, x_2_fore_real_aligned = x_1_fore_real[split+1:], x_2_fore_real[split+1:]
x_1_actual_aligned, x_2_actual_aligned = x_1_real[split:], x_2_real[split:]

min_len_1 = min(len(x_1_fore_real_aligned), len(x_1_actual_aligned))
min_len_2 = min(len(x_2_fore_real_aligned), len(x_2_actual_aligned))

forecast_1 = x_1_fore_real_aligned[:min_len_1]
actual_1   = x_1_actual_aligned[:min_len_1]
rmse_1 = np.sqrt(np.mean((actual_1 - forecast_1)**2))
error_at_last_1 = abs(actual_1[-1] - forecast_1[-1])/values_window_1*100 # Percentage

forecast_2 = x_2_fore_real_aligned[:min_len_2]
actual_2   = x_2_actual_aligned[:min_len_2]
rmse_2 = np.sqrt(np.mean((actual_2 - forecast_2)**2))
error_at_last_2 = abs(actual_2[-2] - forecast_2[-2])/values_window_2*100 # Percentage


forecast_1_dict[1] = forecast_1
forecast_2_dict[1] = forecast_2

print(f"{1}-step ahead RMSE of " + column_name_1 + ":", rmse_1, ", Relative Error at last step:", error_at_last_1, "%\n")
print(f"{1}-step ahead RMSE of " + column_name_2 + ":", rmse_2, ", Relative Error at last step:", error_at_last_2, "%\n\n")

if (max_steps > 1):
    for horizon in range(2, max_steps):

        x_1_fore_std = np.full(len(x_1_std) + horizon, np.nan)
        x_2_fore_std = np.full(len(x_2_std) + horizon, np.nan)

        xi_current = xi_train[-1]
        P_current  = P_train[-1]

        x_1_state = x_1_std[split-1]
        x_2_state = x_2_std[split-1]

        for t in range(split-(horizon-1), len(y_1_full[:-(max_steps-horizon)])+1-horizon):
            
            # ---- propagate parameters h times
            xi_pred = xi_current.copy()
            P_pred  = P_current.copy()
            
            for _ in range(horizon):
                xi_pred = A @ xi_pred
                P_pred  = A @ P_pred @ A.T + Q
                
                # ---- propagate nonlinear system h times
                x_1_temp = x_1_state
                x_2_temp = x_2_state
                
            for _ in range(horizon):
                x_1_temp, x_2_temp = henon_map(xi_pred, x_1_temp, x_2_temp)
                    
            x_1_fore_std[t + horizon] = x_1_temp
            x_2_fore_std[t + horizon] = x_2_temp
                    
            # ---- update filter
            xi_current, P_current = kf.filter_update(
                xi_current,
                P_current,
                observation=np.array([y_1_full[t], y_2_full[t]]),
                observation_matrix=C_full[t]
                )
                    
            x_1_state = y_1_full[t]
            x_2_state = y_2_full[t]

        x_1_fore_real = sigma_1 * x_1_fore_std + mu_1
        x_2_fore_real = sigma_2 * x_2_fore_std + mu_2
    
        x_1_fore_real_aligned, x_2_fore_real_aligned = x_1_fore_real[split+1:], x_2_fore_real[split+1:]
        x_1_actual_aligned, x_2_actual_aligned = x_1_real[split:], x_2_real[split:]
        
        min_len_1 = min(len(x_1_fore_real_aligned), len(x_1_actual_aligned))
        min_len_2 = min(len(x_2_fore_real_aligned), len(x_2_actual_aligned))
        
        forecast_1 = x_1_fore_real_aligned[:min_len_1]
        actual_1   = x_1_actual_aligned[:min_len_1]
        rmse_1 = np.sqrt(np.mean((actual_1 - forecast_1)**2))
        error_at_last_1 = abs(actual_1[-1] - forecast_1[-1])/values_window_1*100 # Percentage
        
        forecast_2 = x_2_fore_real_aligned[:min_len_2]
        actual_2   = x_2_actual_aligned[:min_len_2]
        rmse_2 = np.sqrt(np.mean((actual_2 - forecast_2)**2))
        error_at_last_2 = abs(actual_2[-2] - forecast_2[-2])/values_window_2*100 # Percentage
        
        
        forecast_1_dict[horizon] = forecast_1
        forecast_2_dict[horizon] = forecast_2
        
        print(f"{horizon}-step ahead RMSE of " + column_name_1 + ":", rmse_1, ", Relative Error at last step:", error_at_last_1, "%\n")
        print(f"{horizon}-step ahead RMSE of " + column_name_2 + ":", rmse_2, ", Relative Error at last step:", error_at_last_2, "%\n\n")


# -------------------------------------------------
# PLOT 1 — TRAINING REGION
# -------------------------------------------------

plt.figure(figsize=(15,5))

plt.plot(dates[:split], x_1_real[:split], color="blue")
plt.plot(dates[:split], x_1_kf_real[:split], "--", color="orange")
plt.fill_between(dates[:split], x_1_low_real, x_1_high_real,
                 alpha=0.3, color="sandybrown")
plt.title("henon EMKF Reconstruction on Training Phase of " + column_name_1)
plt.ylim(0, x_1_real.max())
plt.gcf().autofmt_xdate()


plt.figure(figsize=(15,5))

plt.plot(dates[:split], x_2_real[:split], color="blue")
plt.plot(dates[:split], x_2_kf_real[:split], "--", color="orange")
plt.fill_between(dates[:split], x_2_low_real, x_2_high_real,
                 alpha=0.3, color="sandybrown")
plt.title("henon EMKF Reconstruction on Training Phase of " + column_name_2)
plt.ylim(0, x_2_real.max())
plt.gcf().autofmt_xdate()

# -------------------------------------------------
# PLOT 2 — TEST REGION (ALL HORIZONS)
# -------------------------------------------------

plt.figure(figsize=(15,5))

colors = plt.cm.YlOrRd_r(np.linspace(0,1,max_steps))

for i, horizon in enumerate(range(1, max_steps)):

    forecast = forecast_1_dict[horizon]
    date_slice = dates[split+horizon:split+horizon+len(forecast)]
    plt.plot(date_slice,
             forecast,
             '--',
             color=colors[i],
             label=f"{horizon}-step")

x_1_real_all = np.concatenate([x_1_real, x_1_ahead[:(max_steps-1)]]) # Extend to the number of steps blindly taken
plt.plot(dates[split:split+len(x_1_real[split:])+(max_steps-1)], # Same length of x_real
         x_1_real_all[split:],
         color='blue',
         label="Actual")
plt.title("Prediction on Testing Phase of " + column_name_1 + f" with {max_steps} steps ahead")
length = len(dates[split:split+len(x_1_real[split:])+(max_steps-1)])

x_1_start = dates[split + len(x_1_real[split:])]
x_1_end   = dates[split + len(x_1_real[split:]) + (max_steps - 1)]

plt.axvspan(x_1_start, x_1_end, alpha=0.3, color="skyblue")

plt.legend()
plt.gcf().autofmt_xdate()
plt.show()




plt.figure(figsize=(15,5))

colors = plt.cm.YlOrRd_r(np.linspace(0,1,max_steps))

for i, horizon in enumerate(range(1, max_steps)):

    forecast = forecast_2_dict[horizon]   # for asset 2
    date_slice = dates[split+horizon:split+horizon+len(forecast)]
    plt.plot(date_slice,
             forecast,
             '--',
             color=colors[i],
             label=f"{horizon}-step")

x_2_real_all = np.concatenate([x_2_real, x_2_ahead[:(max_steps-1)]]) # Extend to the number of steps blindly taken
plt.plot(dates[split:split+len(x_2_real[split:])+(max_steps-1)], # Same length of x_real
         x_2_real_all[split:],
         color='blue',
         label="Actual")
plt.title("Prediction on Testing Phase of " + column_name_2 + f" with {max_steps} steps ahead")
length = len(dates[split:split+len(x_2_real[split:])+(max_steps-1)])

x_2_start = dates[split + len(x_2_real[split:])]
x_2_end   = dates[split + len(x_2_real[split:]) + (max_steps - 1)]

plt.axvspan(x_2_start, x_2_end, alpha=0.3, color="skyblue")

plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

# -------------------------------------------------
# 3D Plots
# -------------------------------------------------


# fig_rec = go.Figure()

# fig_rec.add_trace(go.Scatter3d( # True Trajectory
#     x=dates,
#     y=x_1_real_all,
#     z=x_2_real_all,
#     mode='lines',
#     name='True Trajectory'))

# fig_rec.add_trace(go.Scatter3d( # KF Reconstruction
#     x=dates[:split],
#     y=x_1_kf_real[:split],
#     z=x_2_kf_real[:split],
#     mode='lines',
#     name='KF Reconstruction'))

# for i, horizon in enumerate(range(1, max_steps)):

#     forecast_1 = forecast_1_dict[horizon]   # for asset 1
#     forecast_2 = forecast_2_dict[horizon]   # for asset 2
#     date_slice = dates[split+horizon:split+horizon+len(forecast)]
#     fig_rec.add_trace(go.Scatter3d( # N-Step Forecast
#         x=date_slice,
#         y=forecast_1,
#         z=forecast_2,
#         mode='lines',
#         name=f"{horizon}-Step Forecast"))




# fig_rec.update_layout(
#     title="3D Joint Trajectory — KF Reconstruction vs True",
#     scene=dict(
#         xaxis_title='Time',
#         yaxis_title='Oil',
#         zaxis_title='Gasoline',
#         xaxis=dict(
#             type='date',
#             dtick="M12",
#             tickformat="%Y-%m-%d")))

# fig_rec.write_html("henon_KF_Reconstruction_" + column_name_1 + "_x_" + column_name_2 + ".html")
# fig_rec.show()