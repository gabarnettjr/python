import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sys

#####################################################################

# USER INPUT

if len(sys.argv) > 1:
    whatToPlot = sys.argv[1]
else:
    whatToPlot = "rho"

# Define the base string which is where the files are located
st = '../../julia/waveDiskPhsfd/results/'

# Set the contour levels
if whatToPlot == "rho":
    clevels = np.linspace(-.5,.5,41)
elif (whatToPlot == "u") or (whatToPlot == "v"):
    clevels = np.linspace(-1/3, 1/3, 41)

#####################################################################

# Define the string used to load the variable you want to plot
varst = st + whatToPlot

# Load the x-coordinates of the points
x = np.array([])
with open(st + 'x.txt') as f:
    for line in f:
        x = np.hstack((x, np.float(line)))

# Load the y-coordinates of the points
y = np.array([])
with open(st + 'y.txt') as f:
    for line in f:
        y = np.hstack((y, np.float(line)))

# Get the triangular mesh for plotting the contours
triang = mtri.Triangulation(x, y)

# Initialize the array to store the variable you want to plot
var = np.zeros(np.shape(x))

# Initialize the frame number
frame = 0

# The main loop that plots and saves figures

while True:

    try:

        i = 0
        with open(varst + "_{0:04d}.txt".format(frame)) as f:
            for line in f:
                var[i] = np.float(line)
                i = i + 1

    except:

        break

    else:

        fig = plt.figure(figsize = (12, 10))
        ax = fig.add_subplot(111)
        cs = ax.tricontourf(triang, var, levels = clevels)
        fig.colorbar(cs)
        plt.axis('equal')
        plt.axis([-1,1,-1,1])
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_trisurf(tri, rho)
        # ax.set_xlim3d(-1,1)
        # ax.set_ylim3d(-1,1)
        # ax.set_zlim3d(-1,1)
        # plt.show()
        
        fig.savefig('{0:04d}'.format(frame) + '.png', bbox_inches = 'tight')
        
        plt.close()

        frame = frame + 1







