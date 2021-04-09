program csv_read_test
implicit none
real, dimension(45, 3) :: a
real(8) :: t1, t2, t3, t4
integer :: n, i, j
call cpu_time ( t1 )

open(10,file="test_data1.input")
read(10,*) a
!print *, 'n:', count(mask, 1)
print *, 'a'
do i = 1, 45 
print *, a(i, 1), a(i, 2), a(i, 3)	!a(:, 1), a(:, 2), a(:, 3)
end do

call cpu_time ( t2 )
call timer ( t4 )
write ( *, * ) 'Elapsed CPU time = ', t2 - t1
end program csv_read_test