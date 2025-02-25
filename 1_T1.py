import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
from scipy.stats import linregress, sem  # For linear regression & standard error calculation

# Given Data (T1 relaxation time and corresponding molarity)
T1 = np.array([1.022, 0.662, 0.278, 0.091, 0.048, 0.010, 0.007 ])
molarity = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005])

# Perform linear regression (fit a straight line to the data)
slope, intercept, r_value, p_value, std_err = linregress(T1, molarity)

# Generate fitted line for plotting
x_fitted = np.linspace(np.min(T1), np.max(T1), 1000)
y_fitted = slope * x_fitted + intercept

# Compute standard error of molarity
molarity_error = round(sem(molarity), 3)

# Plot data and fitted line
plt.scatter(T1, molarity, c='r', label='Data')
plt.plot(x_fitted, y_fitted, 'k', label='Linear Fit')
plt.ylim(0, 2)
plt.title('Linear Fit of Molarity vs. T1 in CuSO4 Solution')
plt.xlabel('1/T1 (ms^-1)')
plt.ylabel('Molarity (M)')
plt.legend()

# Display calculated values on the plot
plt.text(0.1, 1.5, f'$R^2$ = {r_value**2:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.1, 1.3, f'Slope = {slope:.4f} ± {std_err:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.0, -0.6, f'Figure 12.1: Plot of Molarity in moles vs 1/T1 in inverse ms in\n' 
                    f'                      CuSO4 solutions to test linear relation.',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
# Save and display plot
plt.savefig('1_T1_vs_Molarity_LinearFit.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final values with errors
print("Plot saved.")
print(f"Slope = {slope:.4f} ± {std_err:.4f}")
print(f"Intercept = {intercept:.4f}")
print(f"R² = {r_value**2:.4f}")
