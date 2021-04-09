! ======================================================
! Function returning an array: matrix * vector
! ======================================================
  function MatVecMult(A, v) result (w)
   implicit none

   real, dimension(:,:), intent(in) :: A
   real, dimension(:), intent(in)   :: v
   real, dimension( SIZE(A,1) )     :: w   !! Define result using input param

   integer :: i, j
   integer :: N

   N = size(v)

   w = 0.0       !! clear whole vector
   DO i = 1, N
      w = w + v(i) * A( :, i )
   END DO
  end function

! ======================================================

  program Main
   implicit none

! ======================================================
! You need to declare any function that use
! "assumed size" arrays in an interface
! ======================================================
   interface
    function MatVecMult(A, v) result (w)
     real, dimension(:,:), intent(in) :: A
     real, dimension(:), intent(in) :: v
     real, dimension( SIZE(A,1) ) :: w
    end function
   end interface

   real, dimension( 3, 3 ) :: A
   real, dimension( 3 ) :: v1, v2
   integer :: i, j

   DATA A / 1, 1, 2, 1, 2, 3, 1, 1, 1 /
   DATA v1 / 1, 1, 1 /

   v2 = MatVecMult(A, v1)

   print *, "A = "
   DO i = 1, 3
      print *, A(i, :)
   END DO

   print *
   print *, "v1 = ", v1

   print *, "v2 = A * v1 = ", v2
  end program
