#ifndef PRECISION
    #define PRECISION
    #ifdef DOUBLE
        typedef double userfp_t;
    #else
        typedef float userfp_t;
    #endif
#endif