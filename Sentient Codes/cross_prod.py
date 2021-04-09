import numpy as np

# taking multiple inputs at a time separated by comma
a = [float(a) for a in input("Enter multiple value: ").split(",")]
print("Number of list is: ", a) 

b = [float(b) for b in input("Enter multiple value: ").split(",")]
print("Number of list is: ", b) 

# converting list to array
vec_a = np.array(a)
vec_b = np.array(b)

#vec_AB = np.array(a,b)
print('Vector a:', vec_a)
print('Vector b:', vec_b)
