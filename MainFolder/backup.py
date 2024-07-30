import numpy as np
import time

start_time = time.time()

# Kaamerate vektor:
v9190v = np.array([3.82564, 4.18553, 1.46657])
v9196v = np.array([3.70499, 4.15347, 1.55078])
cam_avrg = (v9190v + v9196v) / 2

# Veenus ja Maa asukoha vektorid IM süsteemis
veen_loc = np.array([10.857, -30.625, 94.405])
maa_loc = np.array([2.077, -25.212, 62.602])

# Vektorite arvutamine ja normaliseerimine
veen_vec = veen_loc - cam_avrg
veen_unitvec = veen_vec / np.linalg.norm(veen_vec)

maa_vec = maa_loc - cam_avrg
maa_unitvec = maa_vec / np.linalg.norm(maa_vec)

# PÄIKE
# Antenn
loc_36 = np.array([11.234, 1.629, 3.688])
loc_230 = np.array([7.739, 14.612, 0.488])
antenn_vec = loc_36 - loc_230
antenn_unitvec = antenn_vec / np.linalg.norm(antenn_vec)

# Lipp
loc_17 = np.array([7.502, 7.397, 2.302])
loc_229 = np.array([5.468, 14.882, 0.452])
lipp_vec = loc_17 - loc_229
lipp_unitvec = lipp_vec / np.linalg.norm(lipp_vec)

# Päikese ühikvektor
paike_vec = (antenn_unitvec + lipp_unitvec) / 2
paike_unitvec = paike_vec / np.linalg.norm(paike_vec)

# Keskmine vektor
kesk_vec = (maa_unitvec + veen_unitvec + paike_unitvec) / 3
kesk_unitvec = kesk_vec / np.linalg.norm(kesk_vec)

# HORIZONS
def az_el_to_unitvec(az, el):
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    return np.array([
        np.cos(az_rad) * np.cos(el_rad),
        -np.sin(az_rad) * np.cos(el_rad),
        np.sin(el_rad)
    ])

# Convert azimuth and elevation to unit vectors
HOR_veen_unit = az_el_to_unitvec(75.800407, 69.286656)
HOR_maa_unit = az_el_to_unitvec(94.645414, 66.507913)
HOR_paike_unit = az_el_to_unitvec(89.324328, 13.697882)

# Compute horizon center unit vector
HOR_kesk_vec = (HOR_veen_unit + HOR_maa_unit + HOR_paike_unit) / 3
HOR_kesk_unitvec = HOR_kesk_vec / np.linalg.norm(HOR_kesk_vec)

# Keskmine, Maa, Veenus, Päike
keskmine = np.array([0.024404, -0.633554, 0.773313])
maa = np.array([-0.032284, -0.397313, 0.917115])
veenus = np.array([0.086761, -0.342886, 0.935362])
paike = np.array([0.011457, -0.971490, 0.236802])

# Rotation parameters
xk, yk, zk = 0.11096966105001496, -0.63620006179630684, 0.76350194216964518
xm, ym, zm = -0.024897, -0.433276, 0.900917
xv, yv, zv = 0.071308, -0.349863, 0.934083
xp, yp, zp = 0.253980, -0.939030, 0.231770

# Rotation angles
alpha = -np.arctan(yk / xk)
beta = np.arcsin(zk)
delta = -np.arcsin(0.77331312210251590)
iota = np.arctan(-0.63355444896004787 / 0.024404)

rad = np.pi / 180

# Rotation matrix calculation
def coord_turn(angle, x, y, z):
    gamma = angle * rad
    
    # z-axis rotation
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    
    # y-axis rotation
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    
    # x-axis rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])
    
    # delta rotation
    Rdelta = np.array([
        [np.cos(delta), 0, np.sin(delta)],
        [0, 1, 0],
        [-np.sin(delta), 0, np.cos(delta)]
    ])
    
    # iota rotation
    Riota = np.array([
        [np.cos(iota), -np.sin(iota), 0],
        [np.sin(iota), np.cos(iota), 0],
        [0, 0, 1]
    ])
    
    v = np.array([x, y, z])
    v = Riota @ Rdelta @ Rx @ Ry @ Rz @ v
    return v

# Perform calculations
angles = np.arange(-11.034, -11.03, 1e-10)
errors = []

for angle in angles:
    coord_errors = {
        'keskmine': np.linalg.norm(keskmine - coord_turn(angle, xk, yk, zk)) ** 2,
        'maa': np.linalg.norm(maa - coord_turn(angle, xm, ym, zm)) ** 2,
        'veenus': np.linalg.norm(veenus - coord_turn(angle, xv, yv, zv)) ** 2,
        'paike': np.linalg.norm(paike - coord_turn(angle, xp, yp, zp)) ** 2
    }
    total_error = sum(coord_errors.values())
    errors.append(total_error)

# Find minimum value and corresponding angle
min_index = np.argmin(errors)
min_value = errors[min_index]
best_angle = angles[min_index]

print(f"Minimum Value: {min_value}")
print(f"Best Angle: {best_angle}")

# Execution time
aeg = time.time() - start_time
print(f"\nProgram took {round(aeg, 4)} seconds aka {round(aeg / 60, 4)} minutes to run.")
