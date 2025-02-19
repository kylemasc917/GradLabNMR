import matplotlib.pyplot as plt # matplotlib is the plotting package
import csv # This is for reading .csv files
import numpy as np # For large arrays and functions that operate on them
import matplotlib.tri as tri  # For triangulation


f_float = [15.41279,15.41117,15.40971,15.40791,15.40670,15.40682,15.40768,15.40902,15.41026,15.41189,15.41138,15.41062,15.40876,15.40758,15.40560,15.40544,15.40673,15.40794,15.40928,15.4110,15.41077,15.40898,15.40759,15.40600,15.40492,15.41946,15.41846,15.41846,15.4186,15.41856,15.41846]
x = [-0.4,-0.2,0.0,0.2,0.4,0.4,0.,0.0,-0.2,-0.4,-0.4,-0.2,0.0,0.2,0.4,0.4,0.2,0.0,-0.2,-0.4,-0.4,-0.2,0.0,0.2,0.4,-0.4,-0.5,-0.4,-0.3,-0.4,-0.4]
y = [11.4,11.4,11.4,11.4,11.4,11.2,11.2,11.2,11.2,11.2,11.0,11.0,11.0,11.0,11.0,10.8,10.8,10.8,10.8,10.8,10.6,10.6,10.6,10.6,10.6,11.0,11.4,11.4,11.4,11.5,11.3]
z = 2*np.pi*np.array(f_float)/(100*2.675) # Calculate b-field values

# Use triangulation for contour plotting
triang = tri.Triangulation(x, y)
plt.tricontourf(triang, z)  # Use tricontourf for unstructured data

# Add labels, colorbar, and save the plot
plt.colorbar(label="Magnetic field (T)", format="%.4f")
plt.title('Magnetic Field Uniformity Plot')
plt.xlabel('X (a.u.)')
plt.ylabel('Y (a.u.)')
plt.savefig('magnetgraph.png', dpi=300, bbox_inches='tight')
plt.show()
