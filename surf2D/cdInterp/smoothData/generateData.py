
import numpy as np

wdir = "C:\\Users\\gabarne\\cdInterpData\\smoothData\\"

################################################################################

x = np.array([])
with open(wdir + "x.txt") as f:
    for line in f:
        ell = line.strip()
        x = np.hstack((x, np.float64(ell)))

y = np.array([])
with open(wdir + "y.txt") as f:
    for line in f:
        ell = line.strip()
        y = np.hstack((y, np.float64(ell)))

xe = np.array([])
with open(wdir + "xe.txt") as f:
    for line in f:
        ell = line.strip()
        xe = np.hstack((xe, np.float64(ell)))

ye = np.array([])
with open(wdir + "ye.txt") as f:
    for line in f:
        ell = line.strip()
        ye = np.hstack((ye, np.float64(ell)))

################################################################################

def eff(x,y):
    return np.cos(2*np.pi*x/25000) * np.sin(2*np.pi*y/25000)

def gee(x,y):
    return np.cos(2*np.pi*x/25000) * np.cos(2*np.pi*y/25000) \
    + np.sin(2*np.pi*x/25000) * np.sin(2*np.pi*y/25000)

################################################################################

errX = eff(x,y)
with open(wdir + "errX.txt", "w") as f:
    for i in range(len(errX)):
        f.write('{0:17.14f}\n'.format(errX[i]))

errY = gee(x,y)
with open(wdir + "errY.txt", "w") as f:
    for i in range(len(errY)):
        f.write('{0:17.14f}\n'.format(errY[i]))

errXe = eff(xe,ye)
with open(wdir + "errXe.txt", "w") as f:
    for i in range(len(errXe)):
        f.write('{0:17.14f}\n'.format(errXe[i]))

errYe = gee(xe,ye)
with open(wdir + "errYe.txt", "w") as f:
    for i in range(len(errYe)):
        f.write('{0:17.14f}\n'.format(errYe[i]))

