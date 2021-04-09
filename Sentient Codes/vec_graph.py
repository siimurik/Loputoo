import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from csv import reader

def angle_to_pos(EL, AZ):
    d =  np.pi/180
    x =  np.cos(AZ*d) * np.cos(EL*d)
    y =  np.sin(AZ*d) * np.cos(EL*d)
    z =  np.sin(EL*d)
    vec =  np.array([x, y, z])
    return vec

andmed=pd.read_csv("FINAL_CALC_test_new_pos_to_ang.csv",header=0,
                   names=['ID','X','Y','Z','EL','AZ'])
#print('andmed\n',andmed.head())

vec_suund = []
#print(len(andmed))
for i in np.arange(0, len(andmed), 1):
    vec_suund += [angle_to_pos(andmed['EL'][i], andmed['AZ'][i])]
#print("vec_suund\n", vec_suund) #list of arrays

suunad = pd.DataFrame.from_records(vec_suund, columns = list('UVW'))
#print("\nVektorite suunad:\n",suunad.head())

eelandmed = pd.concat([andmed['X'],andmed['Y'],andmed['Z'], suunad], axis=1)
print('\nAndmed töötlemiseks:\n[(X,Y,Z) määravad asukoha]',
      '\n[(U,V,W) määravad vektori suuna]\n', eelandmed.head())
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
X, Y, Z, U, V, W = zip(*data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W) #x, y, z määravad vektori alguspunti
                            #u, v, w määravad vektori suuna

# read csv file as a list of lists
#with open("FINAL_CALC_test_2.csv", "r") as f:
#    tabeli_read = f.readlines()
#del tabeli_read[0]
#for i in range(0,5):
#    print(tabeli_read[0:5])


ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])
plt.show()
