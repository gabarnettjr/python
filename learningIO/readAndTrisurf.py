import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
# from mpl_toolkits.mplot3d import Axes3D

# Define the base string which is where the files are located
st = '../../julia/waveDiskPhsfd/results/'

# Load the integer which is the total number of nodes
with open(st + 'npts.txt') as f:
    for line in f:
        npts = np.int(line)

# Load the integer which is the number of saves
with open(st + 'numsaves.txt') as f:
    for line in f:
        numsaves = np.int(line)

# Load the x-coordinates of the points
x = np.zeros((npts,))
k = 0
with open(st + 'x.txt') as f:
    for line in f:
        x[k] = np.float(line)
        k = k+1

# Load the y-coordinates of the points
y = np.zeros((npts,))
k = 0
with open(st + 'y.txt') as f:
    for line in f:
        y[k] = np.float(line)
        k = k+1

# # Load the array of all times
# t = np.array([])
# with open(st + 't.txt') as f:
#     for line in f:
#         t = np.hstack((t, np.float(line)))

triang = mtri.Triangulation(x, y)

rho = np.zeros((npts,))

for i in range(numsaves):

    k = 0
    with open(st + "rho_{0:04d}.txt".format(i+1)) as f:
        for line in f:
            rho[k] = np.float(line)
            k = k + 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tricontourf(triang, rho, levels=np.linspace(-.5,.5,21))
    # plt.colorbar()
    plt.axis('equal')
    plt.axis([-1,1,-1,1])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(tri, rho)
    # ax.set_xlim3d(-1,1)
    # ax.set_ylim3d(-1,1)
    # ax.set_zlim3d(-1,1)
    # plt.show()

    fig.savefig( '{0:04d}'.format(np.int(np.round(i)+1e-12)) \
    + '.png', bbox_inches = 'tight' )

    plt.close()






