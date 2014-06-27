from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import rc
import numpy as np
import sys

rc('font',size=28)
rc('font',family='serif')
rc('axes',labelsize=32)

x = np.random.normal(size=2000)
y = np.random.normal(size=2000)

bins=30
fig = plt.figure(figsize=(13.5,8))  # PRL default width

ax1 = fig.add_subplot(121, projection='3d')
plt.subplots_adjust(left=0, right=0.90, top=1, bottom=0, wspace=0.22)

hist, xedges, yedges = np.histogram2d(x, y, bins)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
Z = hist
Z = Z/Z.sum() # normalize Z
surf = ax1.plot_surface(X,Y,Z,cstride=1,rstride=1,color="white",shade=False)
contourz = (Z.min()-Z.max())*0.4  # where to put the contours
cset = ax1.contour(X,Y,Z.reshape(X.shape),zdir='z',offset=contourz)

ax2 = fig.add_subplot(122, projection='3d')
hist, xedges, yedges = np.histogram2d(x, y, bins)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
Z = hist
Z = Z/Z.sum() # normalize Z
surf2 = ax2.plot_surface(X,Y,Z,cstride=1,rstride=1,color="white",shade=False)
contourz2 = (Z.min()-Z.max())*0.4  # where to put the contours
cset2 = ax2.contour(X,Y,Z.reshape(X.shape),zdir='z',offset=contourz2)

ax1.set_xlabel(r'$x_p$')
ax1.set_ylabel(r'$y_p$')
#ax1.set_zlabel(r'$Q(x_p,y_p)$')
ax1.text(X.min()*1.1, Y.min()*1.1, Z.max()*1.1, r'$Q(x_p,y_p)$')
ax1.xaxis.set_rotate_label(False)
ax1.yaxis.set_rotate_label(False)
ax1.zaxis.set_rotate_label(False)
ax1.set_zlim(contourz,Z.max()*1.1)
ax1.view_init(elev=10, azim=135)
ax1.grid(False)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.yaxis.set_major_locator(MultipleLocator(2))
ax1.zaxis.set_major_locator(MultipleLocator(0.01))
[t.set_va('center') for t in ax1.get_yticklabels()]
[t.set_ha('left') for t in ax1.get_yticklabels()]
[t.set_va('center') for t in ax1.get_xticklabels()]
[t.set_ha('right') for t in ax1.get_xticklabels()]
[t.set_va('center') for t in ax1.get_zticklabels()]
[t.set_ha('left') for t in ax1.get_zticklabels()]
ax1.xaxis._axinfo['tick']['inward_factor'] = 0
ax1.xaxis._axinfo['tick']['outward_factor'] = 0.4
ax1.yaxis._axinfo['tick']['inward_factor'] = 0
ax1.yaxis._axinfo['tick']['outward_factor'] = 0.4
ax1.zaxis._axinfo['tick']['inward_factor'] = 0
ax1.zaxis._axinfo['tick']['outward_factor'] = 0.4
ax1.zaxis._axinfo['tick']['outward_factor'] = 0.4

ax2.set_xlabel(r'$x_p$',fontsize=32)
ax2.set_ylabel(r'$y_p$')
#ax2.set_zlabel(r'$Q(x_p,y_p)$')
ax2.text(X.min()*1.1, Y.min()*1.1, Z.max()*1.1, r'$Q(x_p,y_p)$')
ax2.xaxis.set_rotate_label(False)
ax2.yaxis.set_rotate_label(False)
ax2.zaxis.set_rotate_label(False)
ax2.set_zlim(contourz,Z.max()*1.1)
ax2.view_init(elev=10, azim=135)
ax2.grid(False)
ax2.xaxis.pane.set_edgecolor('black')
ax2.yaxis.pane.set_edgecolor('black')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.set_major_locator(MultipleLocator(2))
ax2.yaxis.set_major_locator(MultipleLocator(2))
ax2.zaxis.set_major_locator(MultipleLocator(0.01))
[t.set_va('center') for t in ax2.get_yticklabels()]
[t.set_ha('left') for t in ax2.get_yticklabels()]
[t.set_va('center') for t in ax2.get_xticklabels()]
[t.set_ha('right') for t in ax2.get_xticklabels()]
[t.set_va('center') for t in ax2.get_zticklabels()]
[t.set_ha('left') for t in ax2.get_zticklabels()]
ax2.xaxis._axinfo['tick']['inward_factor'] = 0
ax2.xaxis._axinfo['tick']['outward_factor'] = 0.4
ax2.yaxis._axinfo['tick']['inward_factor'] = 0
ax2.yaxis._axinfo['tick']['outward_factor'] = 0.4
ax2.zaxis._axinfo['tick']['inward_factor'] = 0
ax2.zaxis._axinfo['tick']['outward_factor'] = 0.4

#ax.set_xlim(-40, 40)
#ax.set_ylim(-40, 40)
#ax.zaxis.set_major_locator(LinearLocator(3))
#ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
#ax.locator_params(tight=True)
#ax.tick_params(labelsize=30)
#ax.set_xticks([min(x), max(x), 0.0])
#ax.set_yticks([min(y), max(y), 0.0])
#ax.set_zticks([min(Z), max(Z), 0.0])

fig.savefig("test.pdf")
