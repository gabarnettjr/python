import numpy as np
import matplotlib.pyplot as plt


# Load the integer which is the number of layers of nodes:
with open('../../julia/packages/disk/layers.txt') as f:
    for line in f:
        layers = np.int(line)

# Load the integer which is the total number of radial nodes
with open('../../julia/packages/disk/nPts.txt') as f:
    for line in f:
        nPts = np.int(line)

# Load the x-coordinates of the points
x = np.zeros((nPts,))
k = 0
with open('../../julia/packages/disk/x.txt') as f:
    for line in f:
        x[k] = np.float(line)
        k = k+1

# Load the y-coordinates of the points
y = np.zeros((nPts,))
k = 0
with open('../../julia/packages/disk/y.txt') as f:
    for line in f:
        y[k] = np.float(line)
        k = k + 1

# Get the boundary nodes so they can be connected in a line and plotted
r = np.sqrt(x**2 + y**2)
ind = (abs(r - 1.) <= 1e-12)
xb = x[ind]
xb = np.hstack((xb, xb[0]))
yb = y[ind]
yb = np.hstack((yb, yb[0]))

# Plot the nodes and the boundary
plt.clf()
plt.plot(x, y, '.', xb, yb, '-')
plt.axis('equal')
plt.title('{0:01d} Total Points, h=1/{1:01d}'.format(nPts, layers-1))
plt.show()
