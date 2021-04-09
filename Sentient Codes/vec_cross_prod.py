import numpy as np

# taking multiple inputs at a time separated by comma
a = [float(a) for a in input("Enter multiple values for point 1: ").split(",")]
#print("Number of list is: ", a) 

b = [float(b) for b in input("Enter multiple values for point 2: ").split(",")]
#print("Number of list is: ", b) 

c = [float(c) for c in input("Enter multiple values for point 3: ").split(",")]
#print("Number of list is: ", c) 

# converting list to array
vec_a = np.array(a)
vec_b = np.array(b)
vec_c = np.array(c)

#vec_AB = np.array(a,b)
#print('Vector a:', vec_a)
#print('Vector b:', vec_b)

vec_ab = np.array(vec_b-vec_a)
print(vec_ab)
vec_ac = np.array(vec_c-vec_a)
print(vec_ac)
cross_prod = np.cross(vec_ab, vec_ac)
print('Cross product =',cross_prod)
ρ = np.linalg.norm(cross_prod, 2)
print('Lenght ρ =', ρ)
unit_vec = cross_prod/ρ
print('Unitvector υ = [', unit_vec[0], unit_vec[1], unit_vec[2],']')
