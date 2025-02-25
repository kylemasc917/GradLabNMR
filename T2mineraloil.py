import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
from scipy.optimize import curve_fit  # For curve fitting
from scipy.stats import sem  # For standard error calculation

# Given Data
time = np.array([3.1, 9.3, 15.5, 21.7, 27.9, 34.1, 40.3, 46.5, 52.7, 58.9, 65.1, 71.3, 77.5, 83.7])
amplitude = np.array([6.20, 6.00, 5.76, 5.48, 5.24, 5.04, 4.80, 4.68, 4.44, 4.32, 4.16, 3.96, 3.80, 3.72])

# Define fitting function for T2 decay
def f(t, a, tau, c):  
    return a * np.exp(-t / tau) + c

# Initial guess for curve fitting
initial_values = np.array([3.6, 1.0, 0.3])  

# Fit the function to data
popt, pcov = curve_fit(f, time, amplitude, p0=initial_values)

# Extract fitted parameters and their uncertainties
a, tau, c = popt
p_sigma = np.sqrt(np.diag(pcov))  # Standard deviation (errors) from covariance matrix
tau_error = round(p_sigma[1], 3)  # Extract error in T2 and round to 3 significant figures

# Compute R² value for goodness of fit
y_pred = f(time, *popt)
ss_res = np.sum((amplitude - y_pred) ** 2)
ss_tot = np.sum((amplitude - np.mean(amplitude)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Compute standard error of amplitude
amplitude_error = round(sem(amplitude), 3)

# Generate smooth curve for fitted function
x_fitted = np.linspace(np.min(time), np.max(time), 1000)
y_fitted = f(x_fitted, *popt)

# Plot data and fitted curve
plt.scatter(time, amplitude, c='r', label='Data')
plt.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
plt.ylim(0, 7)
plt.title('T2 0.005M CuSO4 Sample')
plt.xlabel('Delay Time (ms)')
plt.ylabel('Amplitude (V)')
plt.legend()

# Display calculated values on the plot
plt.text(0.2, 2.7, f'$R^2$ = {r2:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.2, 2.2, f'$T_2$ = {tau:.3f} ± {tau_error} ms', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.0, -2.0, f'Figure 11.7: Plot of maximum amplitude of spin echo vs delay time in CuSO4 0.005M\n'
                    f'                  sample to measure $T_2$ using the Meibloom-Gill method.\n'
                    f'Amplitude error: ± 0.12 V, Delay time error: ± 0.1 ms',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Save and display plot
plt.savefig('T2_CuSO4_0005M.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final values with errors
print("Plot saved.")
print(f"T₂ = {tau:.3f} ± {tau_error} ms")
print(f"Amplitude SEM = ± {amplitude_error} V")
