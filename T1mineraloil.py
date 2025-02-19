import matplotlib.pyplot as plt # matplotlib is the plotting package
import csv # This is for reading .csv files
import numpy as np # For large arrays and functions that operate on them
from scipy.optimize import curve_fit # For various special functions, optimization, linear algebra, integration

time = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0]
amplitude = [-2.84, -2.52, -2.04, -1.56, -1.12, -0.56, 0.00, 0.16, 0.64, 1.04, 1.36, 1.56, 1.76, 1.96, 2.08, 2.64, 2.96, 3.20]

amplitude = [val + np.abs(amplitude[0]) for val in amplitude]

def f(t, a, tau, c): # Here we define the function used in the fit
    return a * (1-np.exp(-t/tau)) + c

initial_values = np.array([-3.5, 1.0, -3.0]) # We create an array from expected initial values

popt, pcov = curve_fit(f, time, amplitude, p0 = initial_values) # Fit the function
a = popt[0] # popt is a list of the optimized values for the fit parameters
tau = popt[1]
c = popt[2]
p_sigma = np.sqrt(np.diag(pcov)) # Error (standard deviation) is the square root of variance in the diagonals of pcov matrix
print(popt) # Prints the values for the fit parameters
print(p_sigma) # Prints the errors for the fit parameters

# Calculate R2 value
y_pred = f(np.array(time), *popt)
y_true = np.array(amplitude)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)

x_fitted = np.linspace(np.min(time), np.max(time), 1000) # Create an array from the time limits with 1000 items
y_fitted = a * (1 - np.exp(-x_fitted/tau)) + c # Create the fitted curve
plt.scatter(time, amplitude, c = 'r', label = 'Data') # Makes the plot
plt.plot(x_fitted, y_fitted, 'k', label = 'Fitted curve') # Plots the fit curve
plt.ylim(0, 8) # Set y-axis limits
plt.xlabel('Time (ms)') # The x-axis label
plt.ylabel('Amplitude (V)') # The y-axis label
plt.legend() # Show legend

# Display R2 value on the plot
plt.text(1.0, 2.5, f'$R^2$ = {r2:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(1.0, 3.0, f'$T_1$ = {tau:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.savefig('T1_CuSO4_1M.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved.")
