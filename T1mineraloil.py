import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
from scipy.optimize import curve_fit  # For curve fitting
from scipy.stats import sem  # For standard error calculation

# Given data
time = np.array([10, 50, 100, 110, 150, 200, 250, 300, 350, 400])
amplitude = np.array([-4.04, -2.32, -0.12, 0.00, 1.36, 2.32, 2.96, 3.44, 3.80, 4.04])

# Shift amplitude values to positive domain
amplitude = amplitude + np.abs(amplitude[0])

# Define fitting function
def f(t, a, tau, c):
    return a * (1 - np.exp(-t / tau)) + c

# Initial guess for curve fitting
initial_values = np.array([3.5, 0.7, 0.0])

# Fit the function
popt, pcov = curve_fit(f, time, amplitude, p0=initial_values)

# Extract fitted parameters and their uncertainties
a, tau, c = popt
p_sigma = np.sqrt(np.diag(pcov))  # Standard deviation (errors) from covariance matrix
tau_error = round(p_sigma[1], 3)  # Extract error in T1 and round to 3 significant figures

# Compute R² value for goodness of fit
y_pred = f(time, *popt)
ss_res = np.sum((amplitude - y_pred) ** 2)
ss_tot = np.sum((amplitude - np.mean(amplitude)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Compute standard error of amplitude (optional but adds more error estimation)
amplitude_error = round(sem(amplitude), 3)

# Generate smooth curve for fitted function
x_fitted = np.linspace(np.min(time), np.max(time), 1000)
y_fitted = f(x_fitted, *popt)

# Plot data and fitted curve
plt.scatter(time, amplitude, c='r', label='Data')
plt.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
plt.ylim(0, 12)
plt.title('T1 0.005M CuSO4 Sample')
plt.xlabel('Delay Time (ms)')
plt.ylabel('Amplitude (V)')
plt.legend()

# Display R² and T1 values with errors
plt.text(0.2, 9.5, f'$R^2$ = {r2:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.2, 8.5, f'$T_1$ = {tau:.3f} ± {tau_error} ms', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.0, -3.2, f'Figure 10.7: Plot of maximum amplitude of FID signal vs delay time in\n' 
                    f'                      0.005M CuSO4 sample to measure $T_1$.\n'
                    f'Amplitude error: ± 0.04 V, Delay time error: ± 10 ms',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))


# Save and display plot
plt.savefig('T1_CuSO4_0005M.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved.")
print(f"T1 = {tau:.3f} ± {tau_error} ms")
print(f"Amplitude SEM = ± {amplitude_error} V")
