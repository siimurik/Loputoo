import pandas as pd
from csv import reader
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


start_time = time.time()
# IM coordinates
xk = 0.11096966105001496
yk = -0.63620006179630684
zk = 0.76350194216964518
# Horizons' coodrdinates
hx = 2.4404413268700721E-002
hy = -0.63355444896004787
hz = 0.77331312210251590
# Module's central point
X0 = 1.193314605098880
Y0 = 4.203848106904280
Z0 = -0.534337414313240


alpha   = -np.arctan(yk/xk)
beta    = np.arcsin(zk)
gamma   = np.radians(-11.032961419914068) #
delta   = -np.arcsin(hz)
iota    = np.arctan(hy/hx)
eta     = np.pi/2
#print(np.degrees(alpha))
#print(np.degrees(beta))
#print(np.degrees(gamma))
#print(np.degrees(delta))
#print(np.degrees(iota))
#print(np.degrees(eta))

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

# RZ to AZ doesnt work!!!
def orien(z):
    if [(z < -180) & (z < 90)]:
        nurk = 90 + z
    else:
        nurk = z - 270
    return nurk
# RZ to AZ, that is used in the final code
def orienOG(z):
    nurk = z + 90
    if (nurk >= 180):
        nurkNEW = nurk - 360
        return nurkNEW
    else:
        return nurk

# RX to EL
def elev(x):
    angle = x - 90
    return angle
# Univector pased on EL & AZ values
def angle_to_pos(EL, AZ):
    d =  np.pi/180
    x =  np.cos(AZ*d) * np.cos(EL*d)
    y =  np.sin(AZ*d) * np.cos(EL*d)
    z =  np.sin(EL*d)
    vec =  np.array([x, y, z])
    return vec
# transformed positions into angles
def pos_to_angle(x,y,z):
    AZ_uus = np.arctan(y/x)*180/np.pi + 90
    EL_uus = np.arcsin(z)*180/np.pi
    angles = np.array([EL_uus, AZ_uus])
    return angles

def pos_to_angle2(x,y,z):
    if (x > 0):
        AZ_uus = np.arctan(y/x)*180/np.pi - 90
    else:
        AZ_uus = np.arctan(y/x)*180/np.pi + 90
    EL_uus = np.arcsin(z)*180/np.pi
    angles = np.array([EL_uus, AZ_uus])
    return angles
def pos_to_angle3(x,y,z):
    if (x > 0):
        AZ_uus = np.degrees(np.arctan(y/x))
    else:
        if (y > 0):
            AZ_uus = np.degrees(np.arctan(y/x)) + 180.0 #x<0, y>0
        else:
            AZ_uus = np.degrees(np.arctan(y/x)) - 180.0 #x<0, y<0
    EL_uus = np.arcsin(z)*180/np.pi
    angles = np.array([EL_uus, AZ_uus])
    return angles
# Koordinaatide alguspunt on kuupooduli keskkoht
# mitte kõige lõunapoolsem jalg
def centralize(x,y,z):
    newX = x - X0
    newY = y - Y0
    newZ = z - Z0
    newVECTOR = np.array([newX,newY,newZ])
    return newVECTOR
#-----------------------------------------------------------------------------
# Algandmete imporimine
andmed=pd.read_csv("coord_orien_all.csv",header=0,
                   names=['ID','X','Y','Z','RX','RY','RZ'])
#print('andmed',andmed.head())

#-----------------------------------------------------------------------------
# RX ja RZ töötlus EL ja AZ andmeteks

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
#modRZ = rz.apply(orien) # ----> Annab valed väärtused vahemikus 90 .. 180
                        #Õige funktsioon on [[ orienOG  ]]

# modRZ_test is RZ angle data from the x-axis
# Here the orienOG function is applied successfully
#P.S. Using a.apply() won't work, which is why it's
# applied in this manner
modRZ_test = []
for i in np.arange(0, len(rz),1):
    modRZ_test += [orienOG(rz['RZ'][i])]

# modRZ_test is made into a nice DataFrame
finalRZ = pd.DataFrame(modRZ_test, columns=['RZ'])
#print('Modified RZ data:\n',finalRZ.head())   # Dataframe

# Joining modRX and finalRZ into consistent DataFrame
nurgad = pd.concat([modRX, finalRZ], axis=1)
#nurgad = pd.concat([rx, rz], axis=1)
print('\nnurgad EL & AZ (from the x-axis):\n',nurgad.head())

# pos is an array of modified RX and RZ values
# form now on RX is true EL and RZ true AZ
pos = []
for i in np.arange(0, len(nurgad), 1):
    pos += [angle_to_pos(nurgad['RX'][i], nurgad['RZ'][i])]
#print("position\n",pos[0:5])  #list of arrays

# modPOS is EL and AZ in a nice dataframe
modPOS = pd.DataFrame.from_records(pos, columns = list('XYZ'))
print("\nmodPOS (dir angles EL & AZ made into vectors):\n",modPOS.head())   #arrays in list made into a DataFrame

# Modified positions shall be transformed by the will_turner function
# into a list of arrays
trans_pos = []
for i in np.arange(0, len(modPOS),1):
    trans_pos += [will_turner(modPOS["X"][i], modPOS["Y"][i], modPOS["Z"][i])]
#print(trans_pos)
#print(will_turner(modPOS["X"][0], -modPOS["Y"][0], modPOS["Z"][0]))

# Transformed position coordinates written in a prettier way
# prettier aka "better to work with"
newPOS = pd.DataFrame.from_records(trans_pos, columns=['U','V','W'])
print("\nnewPOS (coord turner applied to recent vectors):\n",newPOS.head())   # pandas DataFrame

# New angles from new position coordinates
newANG = []
for i in range(0, len(newPOS)):
    newANG += [pos_to_angle3(newPOS["U"][i],newPOS["V"][i],newPOS["W"][i])]
#print((newANG[0])[1])   #list of arrays

transANG = pd.DataFrame.from_records(newANG, columns=['EL_uus','AZ_uus'])
print("\ntransANG (vectors --> EL' & AZ'):\n",transANG.head())

#-----------------------------------------------------------------------------
# Kaamerate koordinaatide töötlus

# The holy 6-part-step-vector-turning algorithm is applied here
molder = []
for i in np.arange(0, len(andmed), 1):
    molder += [will_turner(andmed['X'][i], andmed['Y'][i], andmed['Z'][i])]
#print(molder)  #again, made into a list of arrays

modCoord = pd.DataFrame.from_records(molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

shifter = []
for i in np.arange(0, len(modCoord), 1):
    shifter += [centralize(modCoord['X'][i], modCoord['Y'][i], modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

corCOORD = pd.DataFrame.from_records(shifter, columns = list('XYZ'))
print('\nFirst 5 shifted location coord:\n',corCOORD.head())   #arrays in list made into a DataFrame

#df_graph võtab transformeeritud ja nihutatud kaamerate koordinaadid
#ning RX, RZ nurkadest moodustatud vektori, mis on samuti transformeeritud
# ja paneb nad ühte DataFrame'i et
df_graph = pd.concat([corCOORD,newPOS],axis=1)
print('\nAndmed graafiku loomiseks:\n',df_graph.head())
data = df_graph.to_numpy()
#print('\nkatse:\n', data)
#Vajalik graafiku parameerite määramiseks
max_x = max(corCOORD['X']) + 1.0
min_x = min(corCOORD['X']) - 1.0
max_y = max(corCOORD['Y']) + 1.0
min_y = min(corCOORD['Y']) - 1.0
max_z = max(corCOORD['Z']) + 1.0
min_z = min(corCOORD['Z']) - 1.0
#print('max z =', max_z)    #kaamera 9311 kõrgus oli vale
                            #nüüd parandatud :)
# Ma täpselt ei taju mida zip(*data) teeb,
# kuid see vajalik quiver'i funksioneerimiseks
X, Y, Z, U, V, W = zip(*data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#Quiver on maagiline funktsioon, mis määrab 3 esimese väärtuse põhjal
#vektori alguspunti ning ülejäänud kolme põhjal moodustab vektori suuna
ax.quiver(X, Y, Z, U, V, W) #x, y, z määravad vektori alguspunti
                            #u, v, w määravad vektori suuna
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])

#-----------------------------------------------------------------------------
# Lipu koordinaatide töötlus

flag_df=pd.read_csv("lipp_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

# The holy 6-part-step-vector-turning algorithm is applied here
flag_molder = []
for i in np.arange(0, len(flag_df), 1):
    flag_molder += [will_turner(flag_df['X'][i],flag_df['Y'][i],flag_df['Z'][i])]
#print(molder)  #again, made into a list of arrays

flag_modCoord = pd.DataFrame.from_records(flag_molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

flag_shifter = []
for i in np.arange(0, len(flag_modCoord), 1):
    flag_shifter += [centralize(flag_modCoord['X'][i],
                                flag_modCoord['Y'][i],
                                flag_modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

flag_corCOORD = pd.DataFrame.from_records(flag_shifter,
                                          columns = [   'X_flag',
                                                        'Y_flag',
                                                        'Z_flag'])
#print('\nFirst 5 shifted location coord:\n',corCOORD.head())

flag_dat = flag_corCOORD.to_numpy()

X_flag, Y_flag, Z_flag = zip(*flag_dat)
ax.plot(X_flag, Y_flag, Z_flag, color='orange')

#-----------------------------------------------------------------------------
# SWC koordinaatide töötlus
SWC_df=pd.read_csv("SWC_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

# The holy 6-part-step-vector-turning algorithm is applied here
SWC_molder = []
for i in np.arange(0, len(SWC_df), 1):
    SWC_molder += [will_turner(SWC_df['X'][i],SWC_df['Y'][i],SWC_df['Z'][i])]
#print(molder)  #again, made into a list of arrays

SWC_modCoord = pd.DataFrame.from_records(SWC_molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

SWC_shifter = []
for i in np.arange(0, len(SWC_modCoord), 1):
    SWC_shifter += [centralize(SWC_modCoord['X'][i],
                                SWC_modCoord['Y'][i],
                                SWC_modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

SWC_corCOORD = pd.DataFrame.from_records(SWC_shifter,
                                          columns = ['X_SWC', 'Y_SWC', 'Z_SWC'])
#print('\nFirst 5 shifted location coord:\n',corCOORD.head())

SWC_dat = SWC_corCOORD.to_numpy()

X_SWC, Y_SWC, Z_SWC = zip(*SWC_dat)
ax.plot(X_SWC, Y_SWC, Z_SWC, color='orange')

#-----------------------------------------------------------------------------
# S-band antenna koordinaatide töötlus kahes osas (Part 1/2)

Sband1_df=pd.read_csv("Sband_p1_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

# The holy 6-part-step-vector-turning algorithm is applied here
Sband1_molder = []
for i in np.arange(0, len(Sband1_df), 1):
    Sband1_molder += [will_turner(Sband1_df['X'][i],Sband1_df['Y'][i],Sband1_df['Z'][i])]
#print(molder)  #again, made into a list of arrays

Sband1_modCoord = pd.DataFrame.from_records(Sband1_molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

Sband1_shifter = []
for i in np.arange(0, len(Sband1_modCoord), 1):
    Sband1_shifter += [centralize(Sband1_modCoord['X'][i],
                                Sband1_modCoord['Y'][i],
                                Sband1_modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

Sband1_corCOORD = pd.DataFrame.from_records(Sband1_shifter,
                                          columns = ['X_Sband1', 'Y_Sband1', 'Z_Sband1'])
#print('\nFirst 5 shifted location coord:\n',corCOORD.head())

Sband1_dat = Sband1_corCOORD.to_numpy()

X_Sband1, Y_Sband1, Z_Sband1 = zip(*Sband1_dat)
ax.plot(X_Sband1, Y_Sband1, Z_Sband1, color='orange')

#-----------------------------------------------------------------------------
# S-band antenna koordinaatide töötlus kahes osas (Part 2/2)

Sband2_df=pd.read_csv("Sband_p2_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

# The holy 6-part-step-vector-turning algorithm is applied here
Sband2_molder = []
for i in np.arange(0, len(Sband2_df), 1):
    Sband2_molder += [will_turner(Sband2_df['X'][i],Sband2_df['Y'][i],Sband2_df['Z'][i])]
#print(molder)  #again, made into a list of arrays

Sband2_modCoord = pd.DataFrame.from_records(Sband2_molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

Sband2_shifter = []
for i in np.arange(0, len(Sband2_modCoord), 1):
    Sband2_shifter += [centralize(Sband2_modCoord['X'][i],
                                Sband2_modCoord['Y'][i],
                                Sband2_modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

Sband2_corCOORD = pd.DataFrame.from_records(Sband2_shifter,
                                          columns = ['X_Sband2', 'Y_Sband2', 'Z_Sband2'])
#print('\nFirst 5 shifted location coord:\n',corCOORD.head())

Sband2_dat = Sband2_corCOORD.to_numpy()

X_Sband2, Y_Sband2, Z_Sband2 = zip(*Sband2_dat)
ax.plot(X_Sband2, Y_Sband2, Z_Sband2, color='orange')

#-----------------------------------------------------------------------------
# Drawing the ladder
ladder_df=pd.read_csv("ladder_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

# The holy 6-part-step-vector-turning algorithm is applied here
ladder_molder = []
for i in np.arange(0, len(ladder_df), 1):
    ladder_molder += [will_turner(ladder_df['X'][i],ladder_df['Y'][i],ladder_df['Z'][i])]
#print(molder)  #again, made into a list of arrays

ladder_modCoord = pd.DataFrame.from_records(ladder_molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

ladder_shifter = []
for i in np.arange(0, len(ladder_modCoord), 1):
    ladder_shifter += [centralize(ladder_modCoord['X'][i],
                                ladder_modCoord['Y'][i],
                                ladder_modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

ladder_corCOORD = pd.DataFrame.from_records(ladder_shifter,
                                          columns = ['X_ladder', 'Y_ladder', 'Z_ladder'])
#print('\nFirst 5 shifted location coord:\n',corCOORD.head())

ladder_dat = ladder_corCOORD.to_numpy()

X_ladder, Y_ladder, Z_ladder = zip(*ladder_dat)
ax.plot(X_ladder, Y_ladder, Z_ladder, color='orange')
#-----------------------------------------------------------------------------
# Legs done sparetly
legs_df=pd.read_csv("legs_coord.csv",header=0,
                   names=['ID','X','Y','Z'])

# The holy 6-part-step-vector-turning algorithm is applied here
legs_molder = []
for i in np.arange(0, len(legs_df), 1):
    legs_molder += [will_turner(legs_df['X'][i],legs_df['Y'][i],legs_df['Z'][i])]
#print(molder)  #again, made into a list of arrays

legs_modCoord = pd.DataFrame.from_records(legs_molder, columns = list('XYZ'))
#print(modCoord)   #arrays in list made into a DataFrame

legs_shifter = []
for i in np.arange(0, len(legs_modCoord), 1):
    legs_shifter += [centralize(legs_modCoord['X'][i],
                                legs_modCoord['Y'][i],
                                legs_modCoord['Z'][i])]
#print(molder)  #again, made into a list of arrays

legs_corCOORD = pd.DataFrame.from_records(legs_shifter,
                                          columns = ['X_legs', 'Y_legs', 'Z_legs'])
#print('\nFirst 5 shifted location coord:\n',corCOORD.head())

legs_dat = legs_corCOORD.to_numpy()

X_legs, Y_legs, Z_legs = zip(*legs_dat)
ax.plot(X_legs[0:3], Y_legs[0:3], Z_legs[0:3], color='orange')
ax.plot(X_legs[3:6], Y_legs[3:6], Z_legs[3:6], color='orange')
ax.plot(X_legs[6:9], Y_legs[6:9], Z_legs[6:9], color='orange')
ax.plot(X_legs[9:12], Y_legs[9:12], Z_legs[9:12], color='orange')
ax.scatter(X_legs[12:len(legs_df)], Y_legs[12:len(legs_df)], 
           Z_legs[12:len(legs_df)], color='orange')

# Cylinder
R = 2.264704932679790
z1 = 1.30475
z2 = 2.6095
x = np.linspace(-R, R, 50)
z = np.linspace(z1, z2,50)
Xc, Zc = np.meshgrid(x, z)
Yc = np.sqrt(R**2-Xc**2)
#ax.plot_surface(Xc, Yc, Zc, alpha=0.2)
#ax.plot_surface(Xc, -Yc, Zc, alpha=0.2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#-----------------------------------------------------------------------------
#  Modling the final data
df_new = pd.concat([andmed['ID'], corCOORD, transANG], axis=1)
print('\nFirst 5 transformed values:\n',df_new.head(),
      '\nLast 5 transformed values:\n',df_new.tail())

df_new.to_csv(r'FINAL_CALC_test_new_pos_to_ang.csv',
             index = False, header=True)

aeg = time.time() - start_time
print("\nThe Master Code took", round(aeg, 4), "seconds aka", round(aeg*1e03, 4),
      "milliseconds \nto finish its glorious calculations.")

plt.show()
