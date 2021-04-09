## Code for modifying cameras' position and orientation
import pandas as pd
import numpy as np

andmed=pd.read_csv("coord_orien_all.csv",header=0, names=['ID','X','Y','Z','RX','RY','RZ'])
#print('andmed',andmed.head())
rz = andmed.loc[:, ['RZ']]
#print('Raw RZ data:\n',rz.head())

def orien(z):
    if [(z < -180) & (z < 90)]:
        nurk = 90 + z
        return nurk
    else:
        nurk = z - 270
        return nurk

rx = andmed.loc[:, ['RX']]
#print('Raw RX data:\n',rx.head())

def elev(x):
    angle = x - 90
    return angle

def angle_to_pos(EL, AZ):
    d =  np.pi/180
    x =  np.cos(AZ*d) * np.cos(EL*d)
    y = -np.sin(AZ*d) * np.cos(EL*d)
    z = np.sin(EL*d)
    vec =  np.array([x, y, z])
    return vec

#print(angle_to_pos(75.800407,69.286656))

modRZ = rz.apply(orien)
#print('Modified RZ data:\n',modRZ.head())
#modRZ.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\Sentient Codes\correct_AZ.csv', index = False, header=True)

modRX = rx.apply(elev)
#print('Modified RX data:\n',modRX.head())
#modRX.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\Sentient Codes\correct_EL.csv', index = False, header=True)

ainult_RX_RZ = pd.concat([modRX, modRZ], axis=1)
#print(ainult_RX_RZ.head(),'\n',ainult_RX_RZ.tail())
ainult_RX_RZ.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\Sentient Codes\Modified_RX_RZ.csv',
             index = False, header=True)

#angles = pd.read_csv("Modified_RX_RZ.csv",header=0, names=['RX','RZ'])
nurgad = pd.concat([modRX, modRZ], axis=1)

pos = []
for i in np.arange(0, 45, 1):
    pos += [angle_to_pos(nurgad['RX'][i], nurgad['RZ'][i])]
#print(pos)  #list of arrays

modPOS = pd.DataFrame.from_records(pos, columns = list('XYZ'))
#print(modPOS)   #arrays in list made into a DataFrame

#ainult_RX_RZ = pd.concat([modRX, modRZ], axis=1)
#print(ainult_RX_RZ.head(),'\n',ainult_RX_RZ.tail())


df_new = pd.concat([modRX, modRZ, modPOS], axis=1)
print(df_new.head(),'\n',df_new.tail())

df_new.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\Sentient Codes\Modified_RX_RZ_XYZ.csv',
             index = False, header=True)
