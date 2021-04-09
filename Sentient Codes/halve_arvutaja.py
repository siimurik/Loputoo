import numpy as np
import time

start_time = time.time()
# Kaamerate vektor:
v9190v = np.array([[3.82564,	4.18553,	1.46657]])
v9196v = np.array([[3.70499,	4.15347,	1.55078]])
cam_avrg = (v9190v + v9196v)/2
#print(cam_avrg)

# Veenus ja Maa asukoha vektorid IM süsteemis 
veen_loc = np.array([[10.857,	-30.625,	94.405]])
maa_loc = np.array([[2.077,	-25.212,	62.602]])

# Veenuse vektor IM süsteemis
veen_vec = np.array(veen_loc-cam_avrg)
veen_norm2 = np.linalg.norm(veen_vec, 2)
veen_unitvec = veen_vec.T/veen_norm2
#print(veen_vec,'\n',veen_unitvec)

# Veenuse vektor IM süsteemis
maa_vec = np.array(maa_loc-cam_avrg)
maa_norm2 = np.linalg.norm(maa_vec, 2)
maa_unitvec = maa_vec.T/maa_norm2
#print(maa_vec,'\n',maa_unitvec)

# PÄIKE
# Antenn
loc_36 = np.array([[11.234,	1.629,	3.688]])
loc_230 = np.array([[7.739,	14.612,	0.488]])
antenn_vec = loc_36 - loc_230 
antenn_norm2 = np.linalg.norm(antenn_vec, 2)
antenn_unitvec = antenn_vec.T/antenn_norm2
#print(antenn_vec)
# Lipp
loc_17 = np.array([[7.502,	7.397,	2.302]])
loc_229 = np.array([[5.468,	14.882,	0.452]])
lipp_vec = loc_17 - loc_229 
lipp_norm2 = np.linalg.norm(lipp_vec, 2)
lipp_unitvec = lipp_vec.T/lipp_norm2
#print(lipp_vec)
# Päikese ühikvektor
paike_vec = (antenn_unitvec + lipp_unitvec)/2
paike_norm2 = np.linalg.norm(paike_vec, 2)
paike_unitvec = paike_vec/paike_norm2
#print(paike_unitvec)
kesk_vec = (maa_unitvec + veen_unitvec + paike_unitvec)/3
kesk_norm2 = np.linalg.norm(kesk_vec, 2)
kesk_unitvec = kesk_vec/kesk_norm2
#print(kesk_unitvec)

# HORIZONS
# Veenus
veen_AZ_EL = np.array([[75.800407,	69.286656]]).T
HOR_veen_unit = np.array([np.cos(veen_AZ_EL[0]*np.pi/180)*np.cos(veen_AZ_EL[1]*np.pi/180),
                           -np.sin(veen_AZ_EL[0]*np.pi/180)*np.cos(veen_AZ_EL[1]*np.pi/180),
                           np.sin(veen_AZ_EL[1]*np.pi/180)])
#print(HOR_veen_unit)
# Maa
maa_AZ_EL = np.array([[94.645414,	66.507913]]).T
HOR_maa_unit = np.array([np.cos(maa_AZ_EL[0]*np.pi/180)*np.cos(maa_AZ_EL[1]*np.pi/180),
                           -np.sin(maa_AZ_EL[0]*np.pi/180)*np.cos(maa_AZ_EL[1]*np.pi/180),
                           np.sin(maa_AZ_EL[1]*np.pi/180)])
#print(HOR_maa_unit)
# Päike
paike_AZ_EL = np.array([[89.324328,	13.697882]]).T
HOR_paike_unit = np.array([np.cos(paike_AZ_EL[0]*np.pi/180)*np.cos(paike_AZ_EL[1]*np.pi/180),
                           -np.sin(paike_AZ_EL[0]*np.pi/180)*np.cos(paike_AZ_EL[1]*np.pi/180),
                           np.sin(paike_AZ_EL[1]*np.pi/180)])
#print(HOR_paike_unit)
HOR_kesk_vec = (HOR_veen_unit + HOR_maa_unit + HOR_paike_unit)/3
HOR_kesk_norm2 = np.linalg.norm(HOR_kesk_vec, 2)
HOR_kesk_unitvec = HOR_kesk_vec/HOR_kesk_norm2
#print(HOR_kesk_unitvec) 

# Keskmine
xk = 0.11096966105001496
yk = -0.63620006179630684
zk = 0.76350194216964518
# Maa
xm = -2.4896859372283819E-002
ym = -0.43327641681010770
zm =  0.90091713993551636
#Veenus
xv =  7.1307701813322341E-002
yv = -0.34986266743991651
zv =  0.93408314704523432
#print(xk,yk,zk)
# Päike
xp = 0.25397972961048276
yp = -0.93902982835097559
zp = 0.23176988245698202

# Horizon
keskmine = np.array([[2.4404413268700721E-002, -0.63355444896004787,       0.77331312210251590]]).T
maa = np.array([[-3.2284018454244459E-002,  -0.39731293636650600,       0.91711513603705763]]).T
veenus = np.array([[8.6760994985265599E-002,  -0.34288635579381749,       0.93536168232379624]]).T
paike = np.array([[1.1457026022790945E-002, -0.97149031888578385,       0.23680223154757402]]).T

rad = np.pi/180
# kraadid peavad olema radiaanides
alpha = -np.arctan(yk/xk)
beta = np.arcsin(zk)
delta = -np.arcsin(0.77331312210251590)
iota = np.arctan(-0.63355444896004787/2.4404413268700721E-002)
#print(alpha*1/rad,beta*1/rad,delta*1/rad,iota*1/rad)

oige = -11.03

alg = -11.034
lopp = -11.01
#samm = 1e-08
samm = 1e-05

print("Vahemik: [", alg,",",lopp,"] sammupikkusega", samm,".")

a = []
for i in np.arange(alg, lopp, samm):
    a += [i]

def coord_turn(angle,x,y,z):
    gamma = angle * rad
    # z-telg
    a1 = np.array([ [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha),  0],
                    [0,             0,              1]])
    a11 = np.array([[x,y,z]]).T
    a = a1 @ a11
    # y-telg
    b2 = np.array([ [np.cos(beta),  0,  np.sin(beta)],
                    [0,             1,  0],
                    [-np.sin(beta), 0,  np.cos(beta)]])
    b = b2 @ a
    # x-telg
    g1 = np.array([[1,  0,              0],
                   [0,  np.cos(gamma),  -np.sin(gamma)],
                   [0,  np.sin(gamma),  np.cos(gamma)]])
    g = g1 @ b
    # y-telg

    d2 = np.array([ [np.cos(delta),     0,  np.sin(delta)],
                    [0,                 1,  0],
                    [-np.sin(delta),    0,  np.cos(delta)]])

    d = d2 @ g
    # z-telg
    i1 = np.array([ [np.cos(iota),  -np.sin(iota),  0],
                    [np.sin(iota),  np.cos(iota),   0],
                    [0,             0,              1]])
    i = i1 @ d
    return i

#print(coord_turn(oige,xm,ym,zm))
summa = []
vastused = []
for i in a:
    halve0 = [(keskmine - coord_turn(i,xk,yk,zk))**2]
    halve1 = [(maa - coord_turn(i,xm,ym,zm))**2]
    halve2 = [(veenus - coord_turn(i,xv,yv,zv))**2]
    halve3 = [(paike - coord_turn(i,xp,yp,zp))**2]
#    print(halve0, halve1, halve2, halve3, i)
#    print(halve1, i)
    total = halve0 + halve1 + halve2 + halve3
    vastused += [sum(sum(total))]
    #print(total, i)
min_value = min(vastused)   # ühe väärtusega list 
print(min_value)
print("The most minimal value:\t",min_value[0], "\nDegrees:\t\t",
      a[vastused.index(min(vastused))],'\n')
aeg = time.time() - start_time
print("\nProgram took", round(aeg, 4), "seconds aka", round(aeg/60, 4),"minutes to run.")
    
