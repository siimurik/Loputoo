import numpy as np
import pandas as pd
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
gamma   = np.radians(-11.032961419914068)
delta   = -np.arcsin(hz)
iota    = np.arctan(hy/hx)
eta     = np.pi/2

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

a = 0.1109696611
b = -0.6362000618
c = 0.7635019422

#df=pd.read_csv("Modified_RX_RZ_XYZ.csv",header=0, names=['EL','AZ','X','Y','Z'])
df = pd.read_csv("jalgade_lok.csv", header=0, names=["X","Y","Z"])
print(df.head())

molder = []
for i in np.arange(0, (len(df)), 1):
    molder += [will_turner(df['X'][i], df['Y'][i], df['Z'][i])]
#print(molder)  #list of arrays

modCoord = pd.DataFrame.from_records(molder, columns = list('XYZ'))
print(modCoord)   #arrays in list made into a DataFrame

#df_new = pd.concat([df['EL'], df['AZ'], modCoord], axis=1)
#print('\nTransformed data\n',df_new.head(),'\n',df_new.tail())

# For locators!
modCoord.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\Sentient Codes\trans_lok_data.csv',
             index = False, header=True)

#df_new.to_csv(r'C:\Users\siime\OneDrive\Desktop\Fotogrammeetria failid\Sentient Codes\transformed_data.csv',
#             index = False, header=True)
aeg = time.time() - start_time
print("\nWill Turner took", round(aeg, 4), "seconds aka", round(aeg/60, 4),"minutes to orient himself.")
