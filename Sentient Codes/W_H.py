import numpy as np

print("Width and Height calculator for Moon Pictures!")
print("\nw1 to w5 are horizontal pixel distances")
print("h1 to h5 are vertical pixel distances")
print("\nStart adding values from SMALLEST to LARGEST")
print("\nWidth distances:")

w1 = float(input("Input w1 in px: "))
w2 = float(input("Input w2 in px: "))
w3 = float(input("Input w3 in px: "))
w4 = float(input("Input w4 in px: "))
w5 = float(input("Input w5 in px: "))

print("Height distances:")

h1 = float(input("Input h1 in px: "))
h2 = float(input("Input h2 in px: "))
h3 = float(input("Input h3 in px: "))
h4 = float(input("Input h4 in px: "))
h5 = float(input("Input h5 in px: "))

l1=w2-w1
l2=w3-w2
l3=w4-w3
l4=w5-w4

L=l1+l2+l3+l4
l=L/4
W=3900*10/l

m1=h2-h1
m2=h3-h2
m3=h4-h3
m4=h5-h4

M=m1+m2+m3+m4
h=M/4
H=3900*10/h

print('\nW =',np.round(W,4),"px")
print('H =',np.round(H,4),"px")
