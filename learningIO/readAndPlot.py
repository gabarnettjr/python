import numpy as np
import matplotlib.pyplot as plt

with open('../../julia/packages/nodes/nPts.txt') as f:
    for line in f:
        nPts = np.int(line)

x = np.zeros((nPts,))
k = 0
with open('../../julia/packages/nodes/x.txt') as f:
    for line in f:
        x[k] = np.float(line)
        k = k+1

y = np.zeros((nPts,))
k = 0
with open('../../julia/packages/nodes/y.txt') as f:
    for line in f:
        y[k] = np.float(line)
        k = k + 1

plt.clf()
plt.plot(x, y, '.')
plt.axis('equal')
plt.show()
