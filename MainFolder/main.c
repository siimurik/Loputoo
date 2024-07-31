/*
Compile and execute with:
    $ gcc main.c -o main -lm
    $ ./main
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#define PI 3.14159265358979323846  // Define constant PI for mathematical calculations
#define CLOCK_MONOTONIC 1

void ensure_directories_exist();
void angle_to_dir(double EL, double AZ, double dir[3]);
void print_vector(const char* label, double vec[3]);
void normalize_vector(double vec[3]);
void extrinsic_to_intrinsic(double x, double y, double z, double result[3]);
void sequential_rotations(double x, double y, double z, double result[3]);
double azimuth(double z);
double elevation(double x);
void vec_to_angles(double x, double y, double z, double angles[2]);
void centralize(double x, double y, double z, double result[3]);
void load_data_from_csv(const char* filename, double*** data, int* rows, int* cols);
void free_data(double** data, int rows);

int main() {
    // Timing start
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start); // Get starting time

    ensure_directories_exist(); // Ensure the output directories exist

    // Define horizontal angles for celestial bodies
    double AZ_venus_HOR = 75.806271, EL_venus_HOR = 69.278559;
    double AZ_earth_HOR = 94.647066, EL_earth_HOR = 66.507475;
    double AZ_sun_HOR   = 89.324328, EL_sun_HOR   = 13.697882;

    double HOR_venus[3], HOR_earth[3], HOR_sun[3], HOR_avrg[3], HOR_final_avrg[3];

    // Convert angles to direction vectors
    angle_to_dir(EL_venus_HOR, AZ_venus_HOR, HOR_venus);
    angle_to_dir(EL_earth_HOR, AZ_earth_HOR, HOR_earth);
    angle_to_dir(EL_sun_HOR, AZ_sun_HOR, HOR_sun);

    // Compute the average direction vector
    for (int i = 0; i < 3; ++i) {
        HOR_avrg[i] = (HOR_venus[i] + HOR_earth[i] + HOR_sun[i]) / 3.0;
    }
    normalize_vector(HOR_avrg); // Normalize the average vector
    for (int i = 0; i < 3; ++i) {
        HOR_final_avrg[i] = HOR_avrg[i];
    }
    print_vector("Average Horizons azimuth vector", HOR_final_avrg);

    // This section calculates angles for the "six-step" algorithm, using the average unit vector
    // of Earth, Sun, and Venus (v_IM = [xk, yk, zk]) in the ImageModeler coordinate system. 
    // This vector is converted to the actual coordinate system (v_HOR = [hx, hy, hz]) using data
    // from the Horizons On-Line Ephemeris System. The calculated angles are now hard-coded in the 
    // 'sequentialRotations()' function for simplicity and speed. Although v_IM is not explicitly
    // documented in the thesis, it is included here for reference. Uncomment the following section
    // to view the specific values.
    /*
    double xk = 0.11096966105001496, yk = -0.63620006179630684, zk = 0.76350194216964518;
    double hx = HOR_final_avrg[0],   hy = HOR_final_avrg[1],    hz = HOR_final_avrg[2];

    double alpha = -atan(yk / xk);
    double beta = asin(zk);
    double gamma = -11.0329389306380126 * PI / 180.0;
    double delta = -asin(hz);
    double eta = PI / 2;
    double iota = -atan(hy / hx) + eta;

    printf("\nDegrees:\nalpha = %22.16f\nbeta = %22.16f\ngamma = %22.16f\ndelta = %22.16f\niota-eta = %22.16f\niota = %22.16f\n",
        alpha * 180.0 / PI, beta * 180.0 / PI, gamma * 180.0 / PI, delta * 180.0 / PI,
        (iota - eta) * 180.0 / PI, iota * 180.0 / PI);
    */

    // Load data from a CSV file
    double** data;
    int rows, cols;
    load_data_from_csv("InputFiles/coord_orien_all.csv", &data, &rows, &cols);

    // Print initial data
    printf("\nInitial data:\nID           X            Y            Z            RX           RY           RZ\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d", (int)data[i][0]);
        for (int j = 1; j < cols; ++j) {
            printf("%12.5f ", data[i][j]);
        }
        printf("\n");
    }

    // Compute elevation and azimuth for rotation
    double modRX[rows], modRZ[rows];
    for (int i = 0; i < rows; ++i) {
        modRX[i] = elevation(data[i][4]);
        modRZ[i] = azimuth(data[i][6]);
    }

    double nurgad[rows][2];
    for (int i = 0; i < rows; ++i) {
        nurgad[i][0] = modRX[i];
        nurgad[i][1] = modRZ[i];
    }

    // Convert rotation angles to direction vectors
    double dirANG[rows][3];
    for (int i = 0; i < rows; ++i) {
        angle_to_dir(nurgad[i][0], nurgad[i][1], dirANG[i]);
    }

    // Apply sequential rotations to transform position
    double trans_pos_old[rows][3];
    for (int i = 0; i < rows; ++i) {
        sequential_rotations(dirANG[i][0], dirANG[i][1], dirANG[i][2], trans_pos_old[i]);
    }

    // Convert extrinsic angles to intrinsic system
    double init_dataANG[rows][3];
    for (int i = 0; i < rows; ++i) {
        extrinsic_to_intrinsic(data[i][4], data[i][5], data[i][6], init_dataANG[i]);
    }

    // Apply sequential rotations to intrinsic angles
    double transfANG[rows][3];
    for (int i = 0; i < rows; ++i) {
        sequential_rotations(init_dataANG[i][0], init_dataANG[i][1], init_dataANG[i][2], transfANG[i]);
    }

    // Convert transformed vectors to angles
    double newANG[rows][2];
    for (int i = 0; i < rows; ++i) {
        vec_to_angles(transfANG[i][0], transfANG[i][1], transfANG[i][2], newANG[i]);
    }

    // Apply sequential rotations to original coordinates
    double coord_temp[rows][3];
    for (int i = 0; i < rows; ++i) {
        sequential_rotations(data[i][1], data[i][2], data[i][3], coord_temp[i]);
    }

    // Centralize coordinates
    double newCOORD[rows][3];
    for (int i = 0; i < rows; ++i) {
        centralize(coord_temp[i][0], coord_temp[i][1], coord_temp[i][2], newCOORD[i]);
    }

    // Print transformed values
    printf("\nTransformed values:\nID           X            Y            Z            EL           AZ\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d", (int)data[i][0]);
        for (int j = 0; j < 3; ++j) {
            printf("%12.5f ", newCOORD[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            printf("%12.5f ", newANG[i][j]);
        }
        printf("\n");
    }

    // Save transformed data to CSV files
    FILE* output_file1 = fopen("OutputFiles/camera_posANDdir.csv", "w");
    fprintf(output_file1, "ID, X_(m), Y_(m), Z_(m), EL_new_(DEG), AZ_new_(DEG)\n");
    for (int i = 0; i < rows; ++i) {
        fprintf(output_file1, "%d", (int)data[i][0]);
        for (int j = 0; j < 3; ++j) {
            fprintf(output_file1, ",%22.16f", newCOORD[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            fprintf(output_file1, ",%22.16f", newANG[i][j]);
        }
        fprintf(output_file1, "\n");
    }
    fclose(output_file1);

    FILE* output_file2 = fopen("OutputFiles/camera_dirVECS.csv", "w");
    fprintf(output_file2, "Vx, Vy, Vz\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < 3; ++j) {
            fprintf(output_file2, "%22.16f", init_dataANG[i][j]);
            if (j < 2) fprintf(output_file2, ",");
        }
        fprintf(output_file2, "\n");
    }
    fclose(output_file2);

    // Timing end
    clock_gettime(CLOCK_MONOTONIC, &stop); // Get end time

    // Calculate the elapsed time in seconds
    double time_taken = (stop.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (stop.tv_nsec - start.tv_nsec)) * 1e-9;
    printf("\nExecution time: %.3e seconds.\n", time_taken);

    free_data(data, rows); // Free allocated memory

    return 0;
}

//=================================================================================
//=============================End of main code====================================
//=================================================================================


/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function checks whether a directory named "OutputFiles" exists. If the directory
 * does not exist, it creates the directory with the appropriate permissions. The function
 * is intended to ensure that the necessary output directories are available before 
 * performing operations that involve writing files to these directories.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The directory "OutputFiles" is checked using the `stat` function. If the directory
 *   does not exist (i.e., `stat` returns -1), the `mkdir` function is used to create the
 *   directory with permissions set to 0700 (read, write, and execute for the owner only).
 * - If `mkdir` fails, an error message is printed using `perror`, but the function does not
 *   handle the failure beyond reporting it. This behavior is optional and can be customized
 *   as needed.
 * - This function assumes that the `OutputFiles` directory is used for storing output files,
 *   and it will create it if necessary.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * ensure_directories_exist();
 * // Continue with operations that require the "OutputFiles" directory
 * @endcode
 */
void ensure_directories_exist() {
    struct stat st = {0};
    if (stat("OutputFiles", &st) == -1) {
        if (mkdir("OutputFiles", 0700) != 0) {
            perror("mkdir failed");  // Optional: Print an error message if mkdir fails
        }
    }
}


/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function takes elevation (EL) and azimuth (AZ) angles in degrees and converts
 * them into a 3D direction vector. The resulting direction vector is stored in the
 * provided array `dir`.
 *
 * The conversion is based on the following formulas:
 *  - dir[0] = cos(AZ) * cos(EL)  (X-component)
 *  - dir[1] = sin(AZ) * cos(EL)  (Y-component)
 *  - dir[2] = sin(EL)            (Z-component)
 *
 * These formulas assume that the azimuth angle (AZ) is measured from the positive X-axis
 * in the XY-plane, and the elevation angle (EL) is measured from the XY-plane upwards.
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - EL : double
 *      The elevation angle in degrees. It is the angle between the vector and the
 *      XY-plane. Positive values indicate upward direction.
 * - AZ : double
 *      The azimuth angle in degrees. It is the angle between the vector's projection
 *      on the XY-plane and the positive X-axis. Measured clockwise.
 *
 * ### Output:
 * - dir : double[3]
 *      A 3-element array where the resulting direction vector will be stored. The
 *      array should be pre-allocated by the caller.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The angles are converted from degrees to radians within the function for the
 *   trigonometric calculations.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double dir[3];
 * angle_to_dir(45.0, 30.0, dir);
 * printf("Direction Vector: [%f, %f, %f]\n", dir[0], dir[1], dir[2]);
 * @endcode
 */
void angle_to_dir(double EL, double AZ, double dir[3]) {
    double d = PI / 180.0; // Convert degrees to radians
    dir[0] = cos(AZ * d) * cos(EL * d); // X-component
    dir[1] = sin(AZ * d) * cos(EL * d); // Y-component
    dir[2] = sin(EL * d);               // Z-component
}

// Print a vector with a label
void print_vector(const char* label, double vec[3]) {
    printf("%s: [%f, %f, %f]\n", label, vec[0], vec[1], vec[2]);
}

/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function takes a 3D vector and normalizes it to have a unit length (magnitude of 1).
 * The normalization process involves dividing each component of the vector by its norm (length).
 * The resulting normalized vector overwrites the input vector.
 *
 * The normalization process follows the formula:
 *  - norm = sqrt(vec[0]^2 + vec[1]^2 + vec[2]^2)
 *  - vec[0] /= norm
 *  - vec[1] /= norm
 *  - vec[2] /= norm
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * Input/Output:
 * - vec : double[3]
 *      A 3-element array representing the 3D vector to be normalized. The vector is 
 *      modified in-place to store the normalized values.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The function assumes that the input vector is non-zero. If the input vector has a
 *   magnitude of zero, the function will result in a division by zero error.
 * - The function modifies the input vector directly.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double vec[3] = {3.0, 4.0, 0.0};
 * normalize_vector(vec);
 * printf("Normalized Vector: [%f, %f, %f]\n", vec[0], vec[1], vec[2]);
 * @endcode
 */
void normalize_vector(double vec[3]) {
    double norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    vec[0] /= norm;
    vec[1] /= norm;
    vec[2] /= norm;
}


/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * Converts extrinsic rotations (XYZ-axis) to intrinsic 
 * rotations (pitch-roll-yaw).
 * 
 * In ImageModeler, camera orientations are provided as 
 * rotations around the XYZ axes. To use these orientations 
 * in a different context, such as a "six-step" algorithm,
 * they need to be converted into intrinsic rotations, 
 * specifically pitch-roll-yaw angles. This function performs
 *  the conversion from extrinsic rotations (about the XYZ axes)
 * to intrinsic rotations (pitch, roll, and yaw).
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - x : double
 *      Rotation angle around the X-axis in degrees.
 * - y : double
 *      Rotation angle around the Y-axis in degrees.
 * - z : double
 *      Rotation angle around the Z-axis in degrees.
 *
 * ### Output:
 * - result : double[3]
 *      Array to store the resulting intrinsic direction vector.
 *      result[0] : X-component of the intrinsic direction vector.
 *      result[1] : Y-component of the intrinsic direction vector.
 *      result[2] : Z-component of the intrinsic direction vector.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The function converts input angles from degrees to radians before applying 
 *   the rotations.
 * - The intermediate variables `rx`, `ry`, and `rz` store the rotation angles 
 *   in radians.
 * - Temporary arrays `a` and `b` are used to store intermediate results during 
 *   matrix multiplications.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double x = 30.0, y = 45.0, z = 60.0;
 * double result[3];
 * extrinsic_to_intrinsic(x, y, z, result);
 * printf("Intrinsic direction vector: [%f, %f, %f]\n", result[0], result[1], result[2]);
 * @endcode
 */
void extrinsic_to_intrinsic(double x, double y, double z, double result[3]) {
    double rx = x * PI / 180.0;
    double ry = y * PI / 180.0;
    double rz = z * PI / 180.0;

    // Apply rotation matrices in the x, y, and z directions
    double x_turn[3][3] = {{1, 0, 0}, {0, cos(rx), -sin(rx)}, {0, sin(rx), cos(rx)}};
    double v001[3] = {0.0, 0.0, -1.0};
    double a[3];
    for (int i = 0; i < 3; ++i) {
        a[i] = 0;
        for (int j = 0; j < 3; ++j) {       // More clever and faster way to do
            a[i] += x_turn[i][j] * v001[j]; // a [3x3]*[3x1] matrix multplication.
        }
    }

    double y_turn[3][3] = {{cos(ry), 0, sin(ry)}, {0, 1, 0}, {-sin(ry), 0, cos(ry)}};
    double b[3];
    for (int i = 0; i < 3; ++i) {
        b[i] = 0;
        for (int j = 0; j < 3; ++j) {
            b[i] += y_turn[i][j] * a[j];
        }
    }

    double z_turn[3][3] = {{cos(rz), -sin(rz), 0}, {sin(rz), cos(rz), 0}, {0, 0, 1}};
    for (int i = 0; i < 3; ++i) {
        result[i] = 0;
        for (int j = 0; j < 3; ++j) {
            result[i] += z_turn[i][j] * b[j];
        }
    }
}

/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * Applies a series of rotations to a 3D vector using specified angles around the XYZ axes.
 * This function, also known as the "six-step" algorithm, performs a series of 
 * rotations on a 3D vector based on input angles around the X, Y, and Z axes.
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - x : double
 *      X-component of the input vector.
 * - y : double
 *      Y-component of the input vector.
 * - z : double
 *      Z-component of the input vector.
 *
 * ### Output:
 * - result : double[3]
 *      Array to store the resulting vector after applying the rotations.
 *      result[0] : X-component of the resulting vector.
 *      result[1] : Y-component of the resulting vector.
 *      result[2] : Z-component of the resulting vector.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The rotation angles (alpha, beta, gamma, delta, iota) are pre-defined and 
 *   converted from degrees to radians.
 * - Temporary arrays `a`, `b`, `g`, `d`, and intermediate variables are used 
 *   to store intermediate results during matrix multiplications.
 * - Additional reference for the rotation angles: thesis pages 23-24.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double x = 1.0, y = 0.5, z = -0.5;
 * double result[3];
 * sequential_rotations(x, y, z, result);
 * printf("Resulting vector after rotations: [%f, %f, %f]\n", result[0], result[1], result[2]);
 * @endcode
 */
void sequential_rotations(double x, double y, double z, double result[3]) {
    // Additional reference: thesis pages 23-24.
    double radParam =  PI / 180.0;
    double yaw1    =  80.1056830575758170 * radParam;  // yaw (z)
    double pitch1     =  49.7739016604574402 * radParam;  // pitch (y)
    double roll    = -11.0329389306380126 * radParam;  // roll (x) // value found using rollCALC.c
    double pitch2    = -50.6492951614895688 * radParam;  // pitch (y)
    double yaw2     =   2.2052577138069775 * radParam;  // yaw (z)
    
    // Define rotation matrices for each axis
    double a1[3][3] = {{cos(yaw1), -sin(yaw1), 0}, {sin(yaw1), cos(yaw1), 0}, {0, 0, 1}};
    double a[3] = {0};
    for (int i = 0; i < 3; ++i) {                           // Essentially a more clever way to do
        a[i] = a1[i][0] * x + a1[i][1] * y + a1[i][2] * z;  // a [3x3]*[3x1] matrix multplication.
    }                                                       

    double b2[3][3] = {{cos(pitch1), 0, sin(pitch1)}, {0, 1, 0}, {-sin(pitch1), 0, cos(pitch1)}};
    double b[3] = {0};
    for (int i = 0; i < 3; ++i) {
        b[i] = b2[i][0] * a[0] + b2[i][1] * a[1] + b2[i][2] * a[2];
    }

    double g1[3][3] = {{1, 0, 0}, {0, cos(roll), -sin(roll)}, {0, sin(roll), cos(roll)}};
    double g[3] = {0};
    for (int i = 0; i < 3; ++i) {
        g[i] = g1[i][0] * b[0] + g1[i][1] * b[1] + g1[i][2] * b[2];
    }

    double d2[3][3] = {{cos(pitch2), 0, sin(pitch2)}, {0, 1, 0}, {-sin(pitch2), 0, cos(pitch2)}};
    double d[3] = {0};
    for (int i = 0; i < 3; ++i) {
        d[i] = d2[i][0] * g[0] + d2[i][1] * g[1] + d2[i][2] * g[2];
    }

    double i1[3][3] = {{cos(yaw2), -sin(yaw2), 0}, {sin(yaw2), cos(yaw2), 0}, {0, 0, 1}};
    for (int i = 0; i < 3; ++i) {
        result[i] = i1[i][0] * d[0] + i1[i][1] * d[1] + i1[i][2] * d[2];
    }
}


/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function calculates the azimuth angle based on a given Z angle. The azimuth
 * angle is adjusted by adding 90 degrees to the input Z angle and normalizing it 
 * to fall within the range [-180, 180] degrees.
 *
 * The calculation process follows the formula:
 *  - angle = z + 90.0
 *  - If angle >= 180.0, then angle = angle - 360.0
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - z : double
 *      The Z angle in degrees.
 *
 * ### Output:
 * - double
 *      The calculated azimuth angle in degrees, normalized to the range [-180, 180].
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double z_angle = 45.0;
 * double az = azimuth(z_angle);
 * printf("Azimuth: %f\n", az);
 * @endcode
 */
double azimuth(double z) {
    double angle = z + 90.0;
    return (angle >= 180.0) ? angle - 360.0 : angle;
}

/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function calculates the elevation angle based on a given X angle. The elevation
 * angle is obtained by subtracting 90 degrees from the input X angle.
 *
 * The calculation process follows the formula:
 *  - elevation = x - 90.0
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - x : double
 *      The X angle in degrees.
 *
 * ### Output:
 * - double
 *      The calculated elevation angle in degrees.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double x_angle = 135.0;
 * double el = elevation(x_angle);
 * printf("Elevation: %f\n", el);
 * @endcode
 */
double elevation(double x) {
    return x - 90.0;
}

/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function converts a given direction vector (x, y, z) 
 * into elevation and azimuth angles. The angles are converted 
 * from radians to degrees.
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - x : double
 *      X-component of the direction vector.
 * - y : double
 *      Y-component of the direction vector.
 * - z : double
 *      Z-component of the direction vector.
 *
 * ### Output:
 * - angles : double[2]
 *      Array to store the calculated elevation and azimuth angles in degrees.
 *      angles[0] : Elevation angle in degrees.
 *      angles[1] : Azimuth angle in degrees.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double x = 1.0, y = 0.0, z = 0.5;
 * double angles[2];
 * vec_to_angles(x, y, z, angles);
 * printf("Elevation: %f, Azimuth: %f\n", angles[0], angles[1]);
 * @endcode
 */
void vec_to_angles(double x, double y, double z, double angles[2]) {
    angles[0] = asin(z) * 180.0 / PI; // Elevation angle
    if (x > 0) {
        angles[1] = 90.0 - atan(y / x) * 180.0 / PI; // Azimuth angle
    } else {
        angles[1] = -90.0 - atan(y / x) * 180.0 / PI; // Azimuth angle
    }
}

/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function centralizes a set of coordinates (x, y, z) 
 * by subtracting pre-defined offsets (X0, Y0, Z0) from each 
 * component. These offset values are based on the
 * geometric middle point of the Lunar Module.
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * ### Input:
 * - x : double
 *      X-coordinate.
 * - y : double
 *      Y-coordinate.
 * - z : double
 *      Z-coordinate.
 *
 * ### Output:
 * - result : double[3]
 *      Array to store the centralized coordinates.
 *      result[0] : Centralized X-coordinate.
 *      result[1] : Centralized Y-coordinate.
 *      result[2] : Centralized Z-coordinate.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double x = 2.0, y = 5.0, z = 1.0;
 * double result[3];
 * centralize(x, y, z, result);
 * printf("Centralized coordinates: [%f, %f, %f]\n", result[0], result[1], result[2]);
 * @endcode
 */
void centralize(double x, double y, double z, double result[3]) {
    double X0 =  1.193314605098880;
    double Y0 =  4.203848106904280;
    double Z0 = -0.534337414313240;
    result[0] = x - X0;
    result[1] = y - Y0;
    result[2] = z - Z0;
}

/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function reads data from a CSV (Comma-Separated Values) file and stores 
 * it in a dynamically allocated 2D array. The first line of the CSV file is assumed 
 * to be a header row and is skipped. The function determines the number of rows and 
 * columns by processing the CSV file, allocates memory for the data, and then reads 
 * the data into the allocated array.
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * Input:
 * - filename : const char*
 *      The name of the CSV file to be read.
 *
 * Output:
 * - data : double*** 
 *      A pointer to a pointer to a pointer (double***) where the loaded data will 
 *      be stored. This array is dynamically allocated within the function. 
 *      data[0][0] to data[rows-1][cols-1] will contain the CSV data.
 * - rows : int*
 *      A pointer to an integer where the number of rows in the CSV file will be stored.
 * - cols : int*
 *      A pointer to an integer where the number of columns in the CSV file will be stored.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The function first reads through the file to count the number of rows and columns 
 *   (excluding the header row) by analyzing the first and subsequent lines.
 * - Memory is allocated for the data array based on the number of rows and columns.
 * - The file is read again to populate the data array after allocating memory.
 * - It is essential to free the allocated memory using `free()` when the data is no longer needed.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double** data;
 * int rows, cols;
 * load_data_from_csv("data.csv", &data, &rows, &cols);
 * for (int i = 0; i < rows; ++i) {
 *     for (int j = 0; j < cols; ++j) {
 *         printf("%f ", data[i][j]);
 *     }
 *     printf("\n");
 * }
 * free_data(data, rows);
 * @endcode
 */
void load_data_from_csv(const char* filename, double*** data, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Could not open file");
        exit(EXIT_FAILURE);
    }

    char line[1024];
    int row = 0;
    int col = 0;
    int header_processed = 0;

    // First pass: determine the number of rows and columns
    while (fgets(line, sizeof(line), file)) {
        if (!header_processed) {
            // Skip the header row
            header_processed = 1;
            // Count the number of columns based on the first row
            char* token = strtok(line, ",");
            while (token) {
                col++;
                token = strtok(NULL, ",");
            }
            continue;
        }
        row++;
    }

    *rows = row;
    *cols = col;

    // Allocate memory for the data array
    *data = (double**)malloc(*rows * sizeof(double*));
    for (int i = 0; i < *rows; i++) {
        (*data)[i] = (double*)malloc(*cols * sizeof(double));
    }

    rewind(file);

    // Skip the header row
    fgets(line, sizeof(line), file);

    // Second pass: populate the data array
    row = 0;
    while (fgets(line, sizeof(line), file)) {
        col = 0;
        char* token = strtok(line, ",");
        while (token) {
            (*data)[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);
}


/**
 * -------------------------------------------------------
 * Function Description:
 * -------------------------------------------------------
 * This function frees the memory allocated for the 2D array of data that was created 
 * by the `load_data_from_csv` function. It iterates through each row of the array, 
 * frees the memory allocated for the row, and then frees the memory allocated for 
 * the array of row pointers.
 *
 * -------------------------------------------------------
 * Parameters:
 * -------------------------------------------------------
 * Input:
 * - data : double**
 *      Pointer to a 2D array of doubles that was previously allocated by `load_data_from_csv`.
 * - rows : int
 *      The number of rows in the `data` array, used to determine how many rows to free.
 *
 * -------------------------------------------------------
 * Notes:
 * -------------------------------------------------------
 * - The function assumes that the `data` array was allocated using `malloc` and 
 *   follows the same memory allocation pattern.
 * - It is important to call this function to avoid memory leaks after the data is 
 *   no longer needed.
 *
 * -------------------------------------------------------
 * Example:
 * -------------------------------------------------------
 * @code
 * double** data;
 * int rows;
 * // Assume data is loaded using load_data_from_csv()
 * free_data(data, rows);
 * @endcode
 */
void free_data(double** data, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(data[i]);
    }
    free(data);
}
