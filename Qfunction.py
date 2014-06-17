from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def qfunc3d(x,y,bins=4):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #x, y = np.random.rand(2, 100) * 4 # original x and y
    hist, xedges, yedges = np.histogram2d(x, y, bins)

    elements = (len(xedges) - 1) * (len(yedges) - 1)
    xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)

    barsize = xedges[1] - xedges[0]

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = barsize * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    return fig

def qfuncimage(array,bins=10,dolog=False,scaling=1.0):
    x = scaling*np.imag(array) # x is first dim. so imshow has it vertical
    y = scaling*np.real(array) # y is second dim. so imshow has it horizontal

    H, xe, ye = np.histogram2d(x,y,bins)
    extent = [ye[0], ye[-1], xe[0], xe[-1]] # flipped axes since original
    #print extent
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    # dolog takes the log of the histogram to show subtle values
    if dolog:
        H = np.log(H+0.1)

    plt.imshow(H, origin="lower", extent=extent, interpolation='nearest', cmap='jet')
    plt.colorbar()
    plt.xticks((ye[-1],0,ye[0]))
    plt.yticks((xe[0],0,xe[-1]))
    plt.xlabel(r'Real($ \alpha $)')
    plt.ylabel(r'Imag($ \alpha $)')
    plt.title("Q function")

    return fig
