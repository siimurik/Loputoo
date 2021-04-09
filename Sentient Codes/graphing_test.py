import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ay = fig.add_subplot(121, projection='3d')

x =[1,2,3,4,5,6,7,8,9,10]
y =[5,6,2,3,13,4,1,2,4,8]
z =[2,3,3,3,5,7,9,11,9,10]

#ax.scatter(x, y, z, c='r', marker='o')
#ax.plot(x, y, z, c='b')

#def fucked_up(x,y,z):
#    f = ((x-11.337)^2 + (y-2.348)^2 + (z-2.090)^2 - (3.157256/2)^2)^2 + (-0.068848*(x-9.741)  - -1.220278*(y-2.172) +  2.638612*(z-2.350))^2
#    return f

#a =  np.arange(-5,5,0.1)
#b =  np.arange(-5,5,0.1)
#c =  np.arange(-5,5,0.1)
a = [7.807, 7.502, 7.580, 7.508, 7.589]
b = [7.672, 7.397, 6.164, 6.294, 7.480]
c = [0.320, 2.302, 2.094, 1.545, 1.703]

#ay.plot(a,b,c, c='r')
andmed=pd.read_csv("lipp_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

eelandmed = pd.concat([andmed['X'],andmed['Y'],andmed['Z']], axis=1)
#print('\nI dunno man, seems kinda gay to me\n', eelandmed.head())
data = eelandmed.to_numpy()
#print('\nkatse:\n', data)
max_x = max(andmed['X']) + 1.0
min_x = min(andmed['X']) - 1.0
max_y = max(andmed['Y']) + 1.0
min_y = min(andmed['Y']) - 1.0
max_z = max(andmed['Z']) + 1.0
min_z = min(andmed['Z']) - 1.0
#print('max z =', max_z)    #kaamera 9311 kõrgus oli vale
                            #nüüd parandatud :)
X, Y, Z = zip(*data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z)

# Cylinder
x=np.linspace(-1, 1, 100)
z=np.linspace(-2, 2, 100)
Xc, Zc=np.meshgrid(x, z)
Yc = np.sqrt(1-Xc**2)
Yo = np.sqrt(1-Xc**2+Zc**2)

# Draw parameters
rstride = 20
cstride = 10
ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
ax.plot_surface(Xc, Yo, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
ax.plot_surface(Xc, -Yo, Zc, alpha=0.2, rstride=rstride, cstride=cstride)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
