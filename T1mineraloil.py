import matplotlib.pyplot as plt # matplotlib is the plotting package
import csv # This is for reading .csv files
import numpy as np # For large arrays and functions that operate on them
from scipy.optimize import curve_fit # For various special functions, optimization, linear algebra, integration

time = [10, 50, 100, 110, 150, 200, 250, 300, 350, 400]
amplitude = [-4.04, -2.32, -0.12, 0.00, 1.36, 2.32, 2.96, 3.44, 3.80, 4.04]

amplitude = [val + np.abs(amplitude[0]) for val in amplitude]

def f(t, a, tau, c): # Here we define the function used in the fit
    return a * (1-np.exp(-t/tau)) + c

initial_values = np.array([4.2, 120, 0]) # We create an array from expected initial values

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
plt.ylim(0, 10) # Set y-axis limits
plt.title('T1 0.005 M CuSO4 sample.')
plt.xlabel('Time (ms)') # The x-axis label
plt.ylabel('Amplitude (V)') # The y-axis label
plt.legend() # Show legend

# Display R2 value on the plot
plt.text(1.0, 8.0, f'$R^2$ = {r2:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(1.0, 6.0, f'$T_1$ = {tau:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.savefig('T1_CuSO4_0005M.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved.")
