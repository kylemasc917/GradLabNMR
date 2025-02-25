import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from scipy.stats import sem  # Standard error of the mean

# Given frequency measurements
f_float = np.array([15.41279, 15.41117, 15.40971, 15.40791, 15.40670, 15.40682, 15.40768, 15.40902,
                     15.41026, 15.41189, 15.41138, 15.41062, 15.40876, 15.40758, 15.40560, 15.40544,
                     15.40673, 15.40794, 15.40928, 15.4110, 15.41077, 15.40898, 15.40759, 15.40600,
                     15.40492, 15.41946, 15.41846, 15.41846, 15.4186, 15.41856, 15.41846])

# Corresponding x and y positions
x = np.array([-0.4, -0.2, 0.0, 0.2, 0.4, 0.4, 0.0, 0.0, -0.2, -0.4, -0.4, -0.2, 0.0, 0.2, 0.4, 0.4, 0.2, 0.0,
              -0.2, -0.4, -0.4, -0.2, 0.0, 0.2, 0.4, -0.4, -0.5, -0.4, -0.3, -0.4, -0.4])

y = np.array([11.4, 11.4, 11.4, 11.4, 11.4, 11.2, 11.2, 11.2, 11.2, 11.2, 11.0, 11.0, 11.0, 11.0, 11.0, 10.8,
              10.8, 10.8, 10.8, 10.8, 10.6, 10.6, 10.6, 10.6, 10.6, 11.0, 11.4, 11.4, 11.4, 11.5, 11.3])

# Constants
gamma = 2.675  # Gyromagnetic ratio in (10^8 rad/s/T)

# Compute magnetic field strength (B-field)
z = 2 * np.pi * f_float / (100 * gamma)  

# Compute error using standard error of the mean (SEM)
z_error = round(sem(z), 4)  # SEM provides a more statistical approach to error estimation

# Use triangulation for contour plotting
triang = tri.Triangulation(x, y)
plt.tricontourf(triang, z)  # Use tricontourf for unstructured data

# Add labels, colorbar, and error information
plt.colorbar(label="Magnetic field (T)", format="%.4f")
plt.title('Magnetic Field Uniformity Plot')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.text(0.0, 10.4, f'Figure 7: Plot of magnetic field intensity as a function of position.\n'
                    f'X and Y error: ± 0.08 cm, Magnetic Field error (SEM): ± {z_error} T',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Save and display plot
plt.savefig('magnetgraph.png', dpi=300, bbox_inches='tight')
plt.show()
