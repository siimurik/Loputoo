real x, xtapne, epsilon, delta
integer count, a

xtapne = 1.	!Etteantud täpne lahend
x = 1.5		!Katsetamiseks antud üks alglähend
!epsilon = 1e-3	!Komakohtade täpsus
pi = 3.141592653589793

print *,'Hariliku iteratsiooni meetod'
print *, ''
print *,'Etteantud alglahend x0 on'
print *,x
print *,''
count = 0

print '("Sisestage soovitud tapsus (max: 6): "$)'
read *, delta
epsilon = 10.**(-delta)
print *, 'Soovitud tapusus on', delta,'komakohta ehk', epsilon
print*,''

print *,'Sisestades arvu 0, itereeritakse funktsioon g(x) = sqrt(2.)*cos(pi*x/4.)'
print *,'Sisestades arvu 1, itereeritakse funktsioon g(x) = 2./sqrt(3.)*cos(pi*x/6.)'

!do while(abs(x-xtapne) >= epsilon)
print '("a = "$)'
read *, a

if (a==0) then
do while(abs(x-xtapne) >= epsilon)
x = sqrt(2.)*cos(pi*x/4.)
print *, x, count+1
count = count+1
end do
else
do while(abs(x-xtapne) >= epsilon)
x = 2./sqrt(3.)*cos(pi*x/6.)
print *, x, count+1
count = count+1
end do
end if

print *,''
print *,'Noutud tapsus saavutati n sammuga:'
print *,'n =', count
print *, ''
print *, '|x-xtapne| =', abs(x-xtapne)
print *, 'Vastus:', x
stop
end