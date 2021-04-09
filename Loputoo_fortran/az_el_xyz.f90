program Angles_Coordinates
implicit none
!−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
! Declaration of variables
real(8)					:: x, y, z, R, d, az, el
real(8), dimension(3)	:: a, b
real(8), dimension(2)	:: az_el
integer(8)				:: num
real(8), parameter		:: pi = 4.D0*atan(1.D0)
!−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
! Ask user for input:
print *, 'Code for converting AZ & EL angles into XYZ coordinates'
print *, 'or for converting XYZ coordinates into AZ & EL angles.'
print *, ''
print *,'[1] - AZ & EL --> XYZ'
print *,'[2] - XYZ     --> AZ & EL'

print '("num = "$)'
read *, num
print *, ""

if (num==1) then
print *, "Input AZ & EL:"
read *, AZ, EL
R = 1.D0
d = pi/180.D0
a(1) = R * cos(AZ*d) * cos(EL*d)
a(2) = -R * sin(AZ*d) * cos(EL*d)
a(3) = R * sin(EL*d)
print *, 'Unitvector =', a
else
print *, 'Input x, y & z:'
read *, x, y, z
a(1) = x
a(2) = y
a(3) = z
R = NORM2(a)
b(1) = a(1)/R
b(2) = a(2)/R
b(3) = a(3)/R
d = 180.D0/pi
az = atan(y/x)*d
el = asin(z/R)*d
az_el(1) = az
az_el(2) = el
print *, ''
print *, "[AZ, EL]:", az_el
print *, "R =", R
print *, "UNITVECTOR:", b	![X/R, Y/R, Z/R]
end if

end program Angles_Coordinates