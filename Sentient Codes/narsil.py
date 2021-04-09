import pandas as pd
import numpy as np
import time

start_time = time.time()
# IM coordinates
xk = 0.11096966105001496
yk = -0.63620006179630684
zk = 0.76350194216964518
# Horizons' coodrdinates
hx = 2.4404413268700721E-002
hy = -0.63355444896004787
hz = 0.77331312210251590

alpha   = -np.arctan(yk/xk)
beta    = np.arcsin(zk)
gamma   = np.radians(-11.032961419914068) #
delta   = -np.arcsin(hz)
iota    = np.arctan(hy/hx)
eta     = np.pi/2

# rotation of axis (6 times)
def will_turner(x,y,z):
    # 1. turn - z-axis
    a1 = np.array([ [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha),  0],
                    [0,             0,              1]])
    a11 = np.array([[x,y,z]]).T
    a = a1 @ a11
    # 2. turn - y-axis
    b2 = np.array([ [np.cos(beta),      0,      np.sin(beta)],
                    [0,                 1,          0],
                    [-np.sin(beta),     0,      np.cos(beta)]])
    b = b2 @ a
    # 3. turn - x-axis
    g1 = np.array([[1,      0,                      0],
                   [0,  np.cos(gamma),      -np.sin(gamma)],
                   [0,  np.sin(gamma),      np.cos(gamma)]])
    g = g1 @ b
    # 4. turn - y-axis
    d2 = np.array([ [np.cos(delta),     0,      np.sin(delta)],
                    [0,                 1,          0],
                    [-np.sin(delta),    0,      np.cos(delta)]])
    d = d2 @ g
    # 5. turn - z-axis
    i1 = np.array([ [np.cos(iota),  -np.sin(iota),  0],
                    [np.sin(iota),  np.cos(iota),   0],
                    [0,                 0,          1]])
    i = i1 @ d
    # 6. turn - z-axis
    e1 = np.array([ [np.cos(eta),  -np.sin(eta),    0],
                    [np.sin(eta),   np.cos(eta),    0],
                    [0,                 0,          1]])
    e = e1 @ i
    vec = np.array([(e[0])[0], (e[1])[0], (e[2])[0]])
    return vec
# RZ to AZ
def orien(z):
    if [(z < -180) & (z < 90)]:
        nurk = 90 + z
        return nurk
    else:
        nurk = z - 270
        return nurk
# RX to EL
def elev(x):
    angle = x - 90
    return angle
# Univector pased on EL & AZ values
def angle_to_pos(EL, AZ):
    d =  np.pi/180
    x =  np.cos(AZ*d) * np.cos(EL*d)
    y = -np.sin(AZ*d) * np.cos(EL*d)
    z = np.sin(EL*d)
    vec =  np.array([x, y, z])
    return vec

andmed=pd.read_csv("coord_orien_all.csv",header=0,
                   names=['ID','X','Y','Z','RX','RY','RZ'])
#print('andmed',andmed.head())

# rz is for function orien(z)
rz = andmed.loc[:, ['RZ']]
#print('Raw RZ data:\n',rz.head())

# rx is for function elev(z)
rx = andmed.loc[:, ['RX']]
#print('Raw RX data:\n',rx.head())

# For testing angle_to_pos
#print(angle_to_pos(75.800407,69.286656))

# Modifing RX aka EL and RZ aka AZ
modRX = rx.apply(elev)
#print('Modified RX data:\n',modRX.head())  # Dataframe
modRZ = rz.apply(orien)
#print('Modified RZ data:\n',modRZ.head())   # Dataframe


# Joining modRX and modRZ into consistent DataFrame
nurgad = pd.concat([modRX, modRZ], axis=1)

# pos is an array of modified RX and RZ values
# form now on RX is true EL and RZ true AZ
pos = []
for i in np.arange(0, 45, 1):
    pos += [angle_to_pos(nurgad['RX'][i], nurgad['RZ'][i])]
#print(pos)  #list of arrays

# modPOS is EL and AZ in a nice dataframe
modPOS = pd.DataFrame.from_records(pos, columns = list('XYZ'))
#print(modPOS)   #arrays in list made into a DataFrame

# The holy 6-part-step-vector-turning algorithm is applied here 
molder = []
for i in np.arange(0, 45, 1):
    molder += [will_turner(andmed['X'][i], andmed['Y'][i], andmed['Z'][i])]
#print(molder)  #again, made into a list of arrays

modCoord = pd.DataFrame.from_records(molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

df_new = pd.concat([andmed['ID'], modCoord, modRX, modRZ], axis=1)
print('First 5 transformed values:\n',df_new.head(),
      '\nLast 5 transformed values:\n',df_new.tail())

df_new.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\FINAL_CALC_2.csv',
             index = False, header=True)

aeg = time.time() - start_time
print("\nThe Master Code took", round(aeg, 4), "seconds aka", round(aeg/60, 4),
      "minutes \nto finish its glorious calculations.")
