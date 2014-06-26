from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import rc

def qsurf_publish(x,y,bins=30):
    fig = plt.figure(figsize=(3.38,2))  # PRL default width
    ax = fig.add_subplot(121, projection='3d')
    plt.subplots_adjust(left=0, right=0.9, top=1, bottom=0)
    
    hist, xedges, yedges = np.histogram2d(x, y, bins)

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    Z = hist
    Z = Z/Z.sum() # normalize Z

    #ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    surf = ax.plot_surface(X,Y,Z,cstride=1,rstride=1,color="white",shade=False)

    contourz = (Z.min()-Z.max())*0.6  # where to put the contours

    cset = ax.contour(X,Y,Z.reshape(X.shape),zdir='z',offset=contourz)
    
    ax.set_xlabel(r'$x_p$')
    ##ax.set_xlim(-40, 40)
    ax.set_ylabel(r'$y_p$')
    ##ax.set_ylim(-40, 40)
    ax.set_zlabel(r'$Q$')
    ax.set_zlim(contourz,Z.max()*1.1)

    #ax.zaxis.set_major_locator(LinearLocator(3))
    #ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.view_init(elev=10., azim=-45)
    #ax.locator_params(tight=True)
    ax.tick_params(labelsize=10)
    #ax.set_xticks([min(x), max(x), 0.0])
    #ax.set_yticks([min(y), max(y), 0.0])
    #ax.set_zticks([min(Z), max(Z), 0.0])

    return fig

def qfuncimage(array,bins=30,dolog=False,scaling=1.0):
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

def qsurf(x,y,bins=30):
    """Create a surface plot after calculating a kernel estimate"""
    X,Y,Z = kernel_estimate(x,y,bins)
    
    #font = {'size':18}
    #rc('font', **font)

    fig = plt.figure(figsize=(3.38,4))  # PRL default width
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.subplots_adjust(left=0, right=0.9, top=1, bottom=0)

    surf = ax.plot_surface(X,Y,Z.reshape(X.shape),cstride=2,rstride=2,color="white",shade=False)

    contourz = (Z.min()-Z.max())*1.2  # where to put the contours

    cset = ax.contour(X,Y,Z.reshape(X.shape),zdir='z',offset=contourz)
    
    ax.set_xlabel(r'$x_p$')
    ##ax.set_xlim(-40, 40)
    ax.set_ylabel(r'$y_p$')
    ##ax.set_ylim(-40, 40)
    ax.set_zlabel(r'$Q$')
    ax.set_zlim(contourz,Z.max()*1.1)

    #ax.zaxis.set_major_locator(LinearLocator(3))
    #ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.view_init(elev=20., azim=-45)
    #ax.locator_params(tight=True)
    #ax.tick_params(labelsize=20)
    #ax.set_xticks([min(x), max(x), 0.0])
    #ax.set_yticks([min(y), max(y), 0.0])
    #ax.set_zticks([min(Z), max(Z), 0.0])

    return fig

def kernel_estimate(x,y,bins=30):
    """Use the x and y data sets to create a PDF over the x,y range.
    Returns X,Y,Z where Z is the estimated PDF over X,Y"""
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    X, Y = np.mgrid[xmin:xmax:bins*2j, ymin:ymax:bins*2j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    Z = Z/Z.sum() # normalize Z
    return X,Y,Z

def avg_n(X,Y,Z):
    return (Z*0.5*(X**2 + Y**2)).sum() - 1

def est_avg_n(x,y):
    return np.average(x**2+y**2)*0.5

def std_n(X,Y,Z):
    nsquared = (Z * (0.25*X**4 + 0.5*(X**2 * Y**2) + 0.25*Y**4 - 1.5*X**2 - 1.5*Y**2 + 1)).sum()
    avgn = avg_n(X,Y,Z)
    return np.sqrt(nsquared - avgn**2)

if __name__ == '__main__':
    import numpy as np
    import sys
    filename = sys.argv[1]
    #TODO - need to run vaccuum correction
    data = np.load(filename)
    output = data[171,:,:].flatten()*np.sqrt(2.0)/13074
    x = np.real(output)
    y = np.imag(output)

    X,Y,Z = kernel_estimate(x,y)
    print "naive Avg n = ", est_avg_n(x,y)
    print "Avg n = ", avg_n(X,Y,Z)
    print "StDev n = ", std_n(X,Y,Z)

    fig = qsurf_publish(x,y)
    plt.show()


