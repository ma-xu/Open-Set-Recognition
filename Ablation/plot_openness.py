from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
Known = np.arange(0.0, 100.0, 1.0)
Unknown = np.arange(0.0, 100.0, 1.0)
for i in range(len(Known)):
    for j in range(len(Unknown)):
        print(f"{Known[i]}\t{Unknown[j]}\t{(Known[i])/(Known[i]+Unknown[j]+1e-7)}")
Known, Unknown = np.meshgrid(Known, Unknown)
Openness = (Unknown/(Known+Unknown+1e-7))


# Plot the surface.
surf = ax.plot_surface(Known, Unknown, Openness, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0.0, 1.0)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
