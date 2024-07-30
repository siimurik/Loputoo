import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

start_time = time.time()

# Ensure the directories exist
input_dir = "InputFiles"
export_dir = "OutputFiles"
os.makedirs(export_dir, exist_ok=True)

start_time = time.time()

# Convert angles (elevation and azimuth) to a direction vector
def angle_to_dir(EL, AZ):
    d = np.pi / 180  # Conversion factor from degrees to radians
    x = np.cos(AZ * d) * np.cos(EL * d)
    y = np.sin(AZ * d) * np.cos(EL * d)
    z = np.sin(EL * d)
    return np.array([x, y, z])

# Horizon base data for Venus, Earth, and Sun
AZ_venus_HOR, EL_venus_HOR = 75.806271, 69.278559
AZ_earth_HOR, EL_earth_HOR = 94.647066, 66.507475
AZ_sun_HOR,   EL_sun_HOR   = 89.324328, 13.697882

# Compute horizon direction vectors and their average
HOR_venus = angle_to_dir(EL_venus_HOR, AZ_venus_HOR)
HOR_earth = angle_to_dir(EL_earth_HOR, AZ_earth_HOR)
HOR_sun   = angle_to_dir(EL_sun_HOR,   AZ_sun_HOR)
HOR_avrg = (HOR_venus + HOR_earth + HOR_sun) / 3
HOR_final_avrg = HOR_avrg / np.linalg.norm(HOR_avrg)
print("Average Horizons azimuth vector:", HOR_final_avrg)

# Initial coordinates and angles
xk, yk, zk = 0.11096966105001496, -0.63620006179630684, 0.76350194216964518
hx, hy, hz = HOR_final_avrg
X0, Y0, Z0 = 1.193314605098880, 4.203848106904280, -0.534337414313240

# Compute rotation angles
alpha = -np.arctan(yk / xk)
beta = np.arcsin(zk)
gamma = np.radians(-11.032961419914068)
delta = -np.arcsin(hz)
eta = np.pi / 2
iota = -np.arctan(hy / hx) + eta
print("\nDegrees:\nalpha = ", np.degrees(alpha), '\nbeta =', np.degrees(beta), '\ngamma = ', np.degrees(gamma),
      '\ndelta = ', np.degrees(delta), '\neta = ', np.degrees(eta), '\niota = ', np.degrees(iota))

# Initial vector rotation
def extrinsic_to_intrinsic(x, y, z):
    """
    Converts extrinsic rotations (XYZ-axis) to intrinsic rotations (pitch-roll-yaw).

    -------------------------------------------------------
    Function Description:
    -------------------------------------------------------
    In ImageModeler, camera orientations are provided as rotations around the XYZ axes.
    To use these orientations in a different context, such as a "six-step" algorithm,
    they need to be converted into intrinsic rotations, specifically pitch-roll-yaw angles.
    This function performs the conversion from extrinsic rotations (about the XYZ axes)
    to intrinsic rotations (pitch, roll, and yaw).

    -------------------------------------------------------
    Parameters:
    -------------------------------------------------------
    x : float
        Rotation angle around the X-axis in degrees.
    y : float
        Rotation angle around the Y-axis in degrees.
    z : float
        Rotation angle around the Z-axis in degrees.

    -------------------------------------------------------
    Returns:
    -------------------------------------------------------
    numpy.ndarray
        A 3x3 rotation matrix representing the intrinsic rotations (pitch-roll-yaw) 
        corresponding to the provided extrinsic rotations.

    -------------------------------------------------------
    Notes:
    -------------------------------------------------------
    - The function assumes that the input angles are in degrees and converts them to radians
      before applying the rotations.
    - The rotation order applied is first around the X-axis, then the Y-axis, and finally
      the Z-axis.
    - This conversion is necessary for compatibility with algorithms that require intrinsic
      rotation representations.

    -------------------------------------------------------
    Example:
    -------------------------------------------------------
    >>> import numpy as np
    >>> extrinsic_to_intrinsic(30, 45, 60)
    array([[ 0.612372, -0.612372,  0.5     ],
           [ 0.612372,  0.612372, -0.5     ],
           [-0.5     ,  0.5     , -0.707107]])
    """
    rx, ry, rz = np.radians([x, y, z])
    # Rotation around x-axis
    x_turn = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    v001 = np.array([0.0, 0.0, -1.0])
    a = x_turn @ v001.T
    # Rotation around y-axis
    y_turn = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    b = y_turn @ a.T
    # Rotation around z-axis
    z_turn = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return z_turn @ b.T

# Complete axis rotation based on multiple angles
def sequentialRotations(x, y, z):
    """
    Applies a series of rotations to a 3D vector using specified angles around the XYZ axes.

    -------------------------------------------------------
    Function Description:
    -------------------------------------------------------
    This function, also known as the "six-step" algorithm, performs a series of 
    rotations on a 3D vector based on input angles around the X, Y, and Z axes.
    The rotations are applied in the following order:
    1. Rotation around the Z-axis (alpha)
    2. Rotation around the Y-axis (beta)
    3. Rotation around the X-axis (gamma)
    4. Rotation around the Y-axis (delta)
    5. Rotation around the Z-axis [+ 6. Rotation around the Z-axis by 90 degrees] (iota)
    
    These rotations are used to transform the input vector (x, y, z) through a sequence
    of coordinate transformations to achieve the desired orientation.

    -------------------------------------------------------
    Parameters:
    -------------------------------------------------------
    x : float
        The x-coordinate of the initial vector.
    y : float
        The y-coordinate of the initial vector.
    z : float
        The z-coordinate of the initial vector.

    -------------------------------------------------------
    Returns:
    -------------------------------------------------------
    numpy.ndarray
        The transformed vector after applying the specified rotations, flattened to a 1D array.

    -------------------------------------------------------
    Notes:
    -------------------------------------------------------
    - The function rotates the vector around the Z, Y, and X axes in the specified sequence.

    -------------------------------------------------------
    Example:
    -------------------------------------------------------
    >>> import numpy as np
    >>> sequentialRotations(1, 0, 0)
    [ 0.27924625  0.95326278 -0.1153759 ]
    """
    # Rotation around z-axis (alpha)
    a1 = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    a = a1 @ np.array([[x, y, z]]).T
    # Rotation around y-axis (beta)
    b2 = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    b = b2 @ a
    # Rotation around x-axis (gamma)
    g1 = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
    g = g1 @ b
    # Rotation around y-axis (delta)
    d2 = np.array([[np.cos(delta), 0, np.sin(delta)], [0, 1, 0], [-np.sin(delta), 0, np.cos(delta)]])
    d = d2 @ g
    # Rotation around z-axis (iota - eta)
    i1 = np.array([[np.cos(iota), -np.sin(iota), 0], [np.sin(iota), np.cos(iota), 0], [0, 0, 1]])
    return (i1 @ d).flatten()

# Convert RZ to AZ
def azimuth(z):
    angle = z + 90
    return angle - 360 if angle >= 180 else angle

# Convert RX to EL
def elevation(x):
    return x - 90

# Convert direction vector to angles
def vec_to_angles(x, y, z):
    EL_uus = np.degrees(np.arcsin(z))
    AZ_uus = 90.0 - np.degrees(np.arctan(y / x)) if x > 0 else -90.0 - np.degrees(np.arctan(y / x))
    return np.array([EL_uus, AZ_uus])

# Centralize coordinates
def centralize(x, y, z):
    return np.array([x - X0, y - Y0, z - Z0])

# Load data from CSV
data = np.genfromtxt(os.path.join(input_dir, "coord_orien_all.csv"), delimiter=',', skip_header=1)
# Print initial data
print('\nInitial data:')
print('ID           X            Y            Z            RX           RY           RZ')
for row in data[0:5]:
    print(f"{int(row[0]):<5}{' '.join([f'{val:12.5f}' for val in row[1:]])}")

# Process RX and RZ to EL and AZ
modRX = np.vectorize(elevation)(data[:, 4])
modRZ = np.vectorize(azimuth)(data[:, 6])
nurgad = np.column_stack((modRX, modRZ))

# Convert angles to direction vectors
dirANG = np.array([angle_to_dir(el, az) for el, az in nurgad])

# Transform positions with sequentialRotations
trans_pos_old = np.array([sequentialRotations(x, y, z) for x, y, z in dirANG])

# Apply extrinsic_to_intrinsic to initial angles
init_dataANG = np.array([extrinsic_to_intrinsic(rx, ry, rz) for rx, ry, rz in data[:, 4:7]])

# Apply sequentialRotations to transformed angles
transfANG = np.array([sequentialRotations(xd, yd, zd) for xd, yd, zd in init_dataANG])

# Convert new direction vectors to angles
newANG = np.array([vec_to_angles(u, v, w) for u, v, w in transfANG])

# Process camera coordinates
coord_temp = np.array([sequentialRotations(x, y, z) for x, y, z in data[:, 1:4]])
newCOORD = np.array([centralize(x, y, z) for x, y, z in coord_temp])

# Prepare data for graph
data_for_graph = np.hstack((newCOORD, transfANG))
X, Y, Z, U, V, W = data_for_graph.T

# Load data from CSV using csv module
legs_data = []
with open(os.path.join(input_dir, "legs_coord.csv"), newline='') as csvfile:
    legs_reader = csv.reader(csvfile)
    next(legs_reader)  # Skip header
    for row in legs_reader:
        legs_data.append([float(row[1]), float(row[2]), float(row[3])])

legs_data = np.array(legs_data)

# The 6-part-step-vector-turning algorithm is applied here
legs_molder = [sequentialRotations(x, y, z) for x, y, z in legs_data]

# Centralize the coordinates
legs_shifter = [centralize(x, y, z) for x, y, z in legs_molder]

# Convert to numpy array for easy manipulation
legs_shifter = np.array(legs_shifter)
X_legs, Y_legs, Z_legs = legs_shifter.T

# Plotting the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W)
ax.set_xlim([min(X) - 1.0, max(X) + 1.0])
ax.set_ylim([min(Y) - 1.0, max(Y) + 1.0])
ax.set_zlim([min(Z) - 1.0, max(Z) + 1.0])
ax.scatter(-24.23027503, 77.50106191, 1.87140382, color='red')

# Function to process additional datasets and plot them
def process_additional_data(filename, color):
    additional_data = np.genfromtxt(os.path.join(input_dir, filename), delimiter=',', skip_header=1)
    coord_temp = np.array([sequentialRotations(x, y, z) for x, y, z in additional_data[:, 1:4]])
    newCOORD = np.array([centralize(x, y, z) for x, y, z in coord_temp])
    X_add, Y_add, Z_add = newCOORD.T
    ax.plot(X_add, Y_add, Z_add, color=color)

# Process and plot additional datasets
for file_name in ["flag_coord.csv", "SWC_coord.csv", "Sband_p1_coord.csv", "Sband_p2_coord.csv", "ladder_coord.csv"]:
    process_additional_data(file_name, 'orange')

# Plot leg segments separately
for i in range(0, 4):
    ax.plot(X_legs[int(i*3):int((i+1)*3)], Y_legs[int(i*3):int((i+1)*3)], Z_legs[int(i*3):int((i+1)*3)], color='orange')

# Scatter the tips of the legs
ax.scatter(X_legs[12:], Y_legs[12:], Z_legs[12:], color='orange')

# Combine arrays horizontally
data_new = np.hstack([data[:, [0]], newCOORD, newANG])

# Print first 5 rows
print('\nTransformed values:')
print('ID           X            Y            Z            EL           AZ')
for row in data_new[0:5]:
    print(f"{int(row[0]):<5}{' '.join([f'{val:12.5f}' for val in row[1:]])}")

# Save to CSV
np.savetxt(os.path.join(export_dir, 'camera_posANDdir.csv'), data_new, delimiter=',', header='ID, X_(m), Y_(m), Z_(m), EL_new_(DEG), AZ_new_(DEG)', comments='', fmt='%22.16f')
np.savetxt(os.path.join(export_dir, 'camera_dirVECS.csv'), init_dataANG, delimiter=',', header='Vx, Vy, Vz', comments='', fmt='%22.16f')

# Show runtime of the whole code
print("\nExecution time: %e seconds" % (time.time() - start_time))

# Show the plot
plt.show()