program Axes
implicit none
!−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
! Declaration of variables
real(8)					:: x, y, z, angle, phi, theta, psi
real(8), dimension(3)	:: a
integer(8)				:: num
real(8), parameter		:: pi = 4.D0*atan(1.D0)
!−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
! Ask user for input:
print *, 'Code for turning x, y and z axes.'
print *, ''
print *, "Insert 3D vector:"
read *, x, y, z
print *, ''
print *,'[1] - x-axis'
print *,'[2] - y-axis'
print *,'[3] - z-axis'

print '("num = "$)'
read *, num
print *, ""

if (num==1) then
print *, "Input angle:"
read *, angle
phi = angle * pi/180.D0
a(1) = x
a(2) = y * cos(phi) - z * sin(phi)
a(3) = y * sin(phi) + z * cos(phi)
print *, 'Vector =', a
else if (num==2) then
print *, "Input angle:"
read *, angle
phi = theta * pi/180.D0
a(1) =  x * cos(theta) + z * sin(theta)
a(2) =  y
a(3) = -x * sin(theta) + z * cos(theta)
print *, 'Vector =', a
else
print *, "Input angle:"
read *, angle
phi = psi * pi/180.D0
a(1) =  x * cos(psi) - y * sin(psi)
a(2) =  x * sin(psi) + y * cos(psi)
a(3) =  z
print *, 'Vector =', a
end if

end program Axes