import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
plt.rcParams['font.size']='14'
plt.rcParams['font.family']='sans-serif'
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'
plt.rcParams['legend.fontsize']='12'

CMAP = {'k':'#2c3e50','gy':'#7f8c8d','r':'#c0392b','b':'#2980b9','g':'#27ae60',
		'aq':'#16a085','o':'#d35400','y':'#f39c12','p':'#8e44ad','w':'#bdc3c7'}

xy_points = [(-9.75,9.75),(0.0,9.75),(9.75,9.75),(-9.75,0.0),(0.0,0.0),(9.75,0.0),(-9.75,-9.75),(0.0,-9.75),(-9.75,9.75)]
flux_data = [3.47,7.55,5.33,5.43,13.5,8.68,3.53,6.33,5.51]

def quadratic_fit(xy,F_0,a,x_0,y_0):
	return np.array([F_0 + a*(x[0]-x_0)**2 + a*(x[1]-y_0)**2 for x in xy])

fit,unc = curve_fit(quadratic_fit,xy_points,flux_data,p0=[13.5,-0.1,0.0,0.0])
print 'x-offset = ',round(fit[2],2),' +/- ',round(np.sqrt(unc[2][2]),2),' (mm)'
print 'y-offset = ',round(fit[3],2),' +/- ',round(np.sqrt(unc[3][3]),2),' (mm)'
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
ax.scatter([i[0] for i in xy_points],[i[1] for i in xy_points],flux_data,c=CMAP['r'],marker='o',label='Measured Flux')
xy_interp = [(x,y) for y in np.arange(-10,10.1,0.5) for x in np.arange(-10,10.1,0.5)]
Z = quadratic_fit(xy_interp,*fit)
ax.plot_trisurf([i[0] for i in xy_interp],[i[1] for i in xy_interp],Z,color=CMAP['gy'],alpha=0.8)
ax.set_xlabel('\n\nX Position (mm)')
ax.set_ylabel('\n\nY Position (mm)')
ax.set_zlabel('\n\n'+r'Neutron Flux (10$^6$ n/cm$^2$/s)')
ax.legend()
plt.show()