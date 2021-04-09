print("Focal distance calculator for Moon Pictures!")
print("\nFollowing the equation 1/F = 1/f - 1/d")
print("we can find F according to this equation:")
print("F=(f*d)/(d-f)")
print("\nChoose from 3 distances: 5.3, 15, 74")
f = 61.1
d = float(input("Input d in ft: "))

# d = d/3.280840*1000 or 
# 1200/3937 = 0.3048006096012192
d = d*(1200/3937)*1000
F=(f*d)/(d-f)

print('F =',round(F,2),"mm")
