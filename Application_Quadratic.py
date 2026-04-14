import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import sys
plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 24,})

file_type = input("Crypto or Commodity? ") # Load Data
column_name = input("Price Asset? ")
max_steps = int(input("Number of N-Steps Ahead (max 7) "))
if (max_steps > 7 or max_steps < 0): # Direct Questions to User
    sys.exit("Exiting... Please enter a value between 1 and 7!")
    
max_steps = max_steps+1 # Must include also the endpoint
file = pd.read_csv("E:/Thesis/Codes/Application/" + str.lower(file_type) + "_prices_full.csv")
file['Date'] = pd.to_datetime(file['Date'])
file_raw = file[file['Date'] <= "2026-02-28"]
file_ahead = file[file['Date'] > "2026-02-28"]
x_raw = pd.to_numeric(file_raw[column_name]).values
mask = np.isfinite(x_raw)
x_raw = x_raw[mask]
split = int(0.8 * len(x_raw)) # Train-Test Approach

mu = x_raw[:split].mean() # Standardise using Training only
sigma = x_raw[:split].std()
x_std = (x_raw - mu) / sigma

def quadratic_map(a, b, c, x): # Function for Quadratic
    return a + b*x + c*x**2

y = x_std[1:]
x_prev = x_std[:-1]
C = np.zeros((len(y), 1, 3))
C[:, 0, 0] = 1
C[:, 0, 1] = x_prev
C[:, 0, 2] = x_prev**2
y_train = y[:split-1]
y_test  = y[split-1:]
C_train = C[:split-1]
C_test  = C[split-1:]

kf = KalmanFilter(
    transition_matrices=np.eye(3),
    observation_matrices=C_train,
    transition_covariance=1e-4*np.eye(3),
    observation_covariance=1e-4,
    initial_state_mean=[1,1,1],
    initial_state_covariance=np.eye(3))

kf = kf.em( # EMKF
    y_train,
    n_iter=10,
    em_vars=[
        'transition_matrices',
        'transition_covariance',
        'observation_covariance'])

xi_train, P_train = kf.smooth(y_train) # Smoothing
x_pred_train = np.zeros(len(y_train)) # Training RMSE
x_state = x_std[0]

for t in range(len(y_train)):
    a, b, c = xi_train[t]
    x_pred_train[t] = quadratic_map(a, b, c, x_state)
    x_state = y_train[t]

x_pred_train_real = sigma * x_pred_train + mu
y_train_real = sigma * y_train + mu # Convert to real from Standardised
rmse_train = np.sqrt(np.mean((y_train_real - x_pred_train_real)**2))
print("Training RMSE:", rmse_train)

kf_test = KalmanFilter( # Testing Reconstruction
    transition_matrices=kf.transition_matrices,
    transition_covariance=kf.transition_covariance,
    observation_covariance=kf.observation_covariance,
    observation_matrices=C,
    initial_state_mean=xi_train[-1],
    initial_state_covariance=P_train[-1])

xi_test, P_test = kf_test.filter(y_test)
xi_full = np.vstack([xi_train, xi_test])

x_kf = np.zeros(len(x_std)) # Full Reconstruction
x_kf[0] = x_std[0]

for t in range(len(x_std)-1):
    a, b, c = xi_full[t]
    x_kf[t+1] = quadratic_map(a, b, c, x_kf[t])

x_kf_real = sigma * x_kf + mu
x_real = x_raw

x_ahead = pd.to_numeric(file_ahead[column_name]).values # Prepare future observations
mask_2 = np.isfinite(x_ahead)
x_ahead = x_ahead[mask_2]

x_std_ahead = (x_ahead - mu) / sigma
y_full = np.concatenate([y, x_std_ahead[:max_steps]])
x_prev_full = np.concatenate([x_std[:-1], x_std[-1:], x_std_ahead[:max_steps-1]])

C_full = np.zeros((len(y_full), 1, 3))
C_full[:, 0, 0] = 1
C_full[:, 0, 1] = x_prev_full
C_full[:, 0, 2] = x_prev_full**2

dates_raw = file_raw['Date'].values
dates_raw = dates_raw[mask]
dates = np.concatenate([dates_raw, file_ahead['Date'].values[:max_steps]])
values_window = x_real.max() - x_real.min() # To measure errors at the end as a percentage

A = kf.transition_matrices # Test RMSE (1-step ahead)
Q = kf.transition_covariance
x_pred_test = np.zeros(len(y_test))
xi_current = xi_train[-1]
P_current  = P_train[-1]
x_state = x_std[split-1]

for t in range(len(y_test)):
    xi_pred = A @ xi_current
    a, b, c = xi_pred
    x_pred_test[t] = quadratic_map(a, b, c, x_state)
    xi_current, P_current = kf.filter_update(
        xi_current,
        P_current,
        observation=y_test[t],
        observation_matrix=C_test[t])
    x_state = y_test[t]

x_pred_test_real = sigma * x_pred_test + mu
y_test_real = sigma * y_test + mu
rmse_test = np.sqrt(np.mean((y_test_real - x_pred_test_real)**2))
print("Test RMSE:", rmse_test)

forecast_dict = {} # R-rolling Forecasts
x_fore_std = np.full(len(x_std), np.nan)
xi_current = xi_train[-1]
P_current  = P_train[-1]
x_state = x_std[split-1]

for t in range(split, len(y_full)-(max_steps)):
    xi_pred = xi_current.copy()
    P_pred  = P_current.copy()
    xi_pred = A @ xi_pred
    P_pred  = A @ P_pred @ A.T + Q
    a, b, c = xi_pred
    x_temp = x_state
    x_temp = quadratic_map(a, b, c, x_temp)
    x_fore_std[t+1] = x_temp
    xi_current, P_current = kf.filter_update(
        xi_current,
        P_current,
        observation=y_full[t],
        observation_matrix=C_full[t])
    x_state = y_full[t]

x_fore_real = sigma * x_fore_std + mu
x_fore_real_aligned = x_fore_real[split+1:]
x_actual_aligned = x_real[split:]
min_len = min(len(x_fore_real_aligned), len(x_actual_aligned))
forecast = x_fore_real_aligned[:min_len]
actual   = x_actual_aligned[:min_len]
rmse = np.sqrt(np.mean((actual - forecast)**2))
error_at_last = abs(actual[-1] - forecast[-1])/values_window*100 # Percentage

forecast_dict[1] = forecast
print(f"{1}-step ahead RMSE:", rmse, ", Relative Error at last step:", error_at_last, "%")
if (max_steps > 1): # Rest of the Steps (Blind-Search)
    for horizon in range(2, max_steps):
        x_fore_std = np.full(len(x_std) + horizon, np.nan)
        xi_current = xi_train[-1]
        P_current  = P_train[-1]
        x_state = x_std[split-1]

        for t in range(split-(horizon-1), len(y_full[:-(max_steps-horizon)])+1-horizon):
            xi_pred = xi_current.copy()
            P_pred  = P_current.copy()
            for _ in range(horizon):
                xi_pred = A @ xi_pred
                P_pred  = A @ P_pred @ A.T + Q
            a, b, c = xi_pred
            x_temp = x_state
            for _ in range(horizon):
                x_temp = quadratic_map(a, b, c, x_temp)
            x_fore_std[t + horizon] = x_temp
            xi_current, P_current = kf.filter_update(
                xi_current,
                P_current,
                observation=y_full[t],
                observation_matrix=C_full[t])
            x_state = y_full[t]

        x_fore_real = sigma * x_fore_std + mu
        x_fore_real_aligned = x_fore_real[split+1:]
        x_real_all = np.concatenate([x_real, x_ahead[:(horizon-1)]]) # Extend to the number of steps blindly taken
        
        x_actual_aligned = x_real[split:]
        min_len = min(len(x_fore_real_aligned), len(x_actual_aligned))
        forecast = x_fore_real_aligned[:min_len]
        actual   = x_actual_aligned[:min_len]

        rmse = np.sqrt(np.mean((actual - forecast)**2))
        error_at_last = abs(x_real_all[-1] - x_fore_real[-1])/values_window*100 # Percentage
        forecast_dict[horizon] = forecast
        print(f"{horizon}-step ahead RMSE:", rmse, ", Relative Error at last step:", error_at_last, "%")  

plt.figure(figsize=(15,5)) # Plot Training Region
plt.plot(dates[:split], x_real[:split], color="blue")
plt.plot(dates[:split], x_kf_real[:split], "--", color="orange")
plt.title("Quadratic EMKF Reconstruction on Training Phase of " + column_name)
plt.gcf().autofmt_xdate()

plt.figure(figsize=(15,5)) # Plot Testing Region
colors = plt.cm.YlOrRd_r(np.linspace(0,1,max_steps))
for i, horizon in enumerate(range(1, max_steps)):
    forecast = forecast_dict[horizon]
    date_slice = dates[split+horizon:split+horizon+len(forecast)]
    plt.plot(date_slice,
             forecast,
             '--',
             color=colors[i],
             label=f"{horizon}-step")
x_real_all = np.concatenate([x_real, x_ahead[:(max_steps-1)]]) # Extend to the number of steps blindly taken
plt.plot(dates[split:split+len(x_real[split:])+(max_steps-1)], # Same length of x_real
         x_real_all[split:],
         color='blue',
         label="Actual")
plt.title("Prediction on Testing Phase of " + column_name + f" with {max_steps} steps ahead")
length = len(dates[split:len(dates)])
x_start = dates[split + len(x_real[split:])]
x_end   = dates[split + len(x_real[split:]) + (max_steps - 1)]
plt.axvspan(x_start, x_end, alpha=0.3, color="skyblue")
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()