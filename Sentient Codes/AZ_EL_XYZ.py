import math as m

print("[1] - AZ & EL --> XYZ")
print("[2] - XYZ     --> AZ & EL")

num = float(input("Input NUMBER: "))

if num == 1:
    AZ = float(input("Input AZ: "))
    EL = float(input("Input EL: "))
    R = 1
    d = m.pi/180
    x = R * m.cos(AZ*d) * m.cos(EL*d)
    y = -R * m.sin(AZ*d) * m.cos(EL*d)
    z = R * m.sin(EL*d)
    xyz = [x,y,z]
    print("\n[X, Y, Z] =",xyz)
else:
    X = float(input("Input X: "))
    Y = float(input("Input Y: "))
    Z = float(input("Input Z: "))
    R = m.sqrt(X**2 + Y**2 + Z**2)
    d = m.pi/180
    ch = float(input("[1]DEG or [2]RAD: "))
    if ch==1:
        az = m.degrees(m.atan(Y/X))
        el = m.degrees(m.asin(Z/R))
        az_el = [az,el]
        print("\n[AZ, EL]:", az_el)
        print("R =", R)
        print("ÜHIKVEKTOR: [X/R, Y/R, Z/R] =", [X/R, Y/R, Z/R])
    else:
        az = m.atan(Y/X)
        el = m.asin(Z/R)
        az_el = [az,el]
        print("\n[AZ, EL]:", az_el)
    
#print(xyz)
