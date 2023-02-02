module precision_m
    integer, parameter:: sf = kind(0.)
    integer, parameter:: df = kind(0.d0)

    ! compile with -DDOUBLE to compile with double precision
#ifdef DOUBLE
    integer, parameter:: f = df
#else
    integer, parameter:: f = sf
#endif

end module precision_m