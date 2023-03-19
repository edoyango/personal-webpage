---
title: Inputs
weight: 2
---

# Inputs

GraSPH uses two files to describe the simulation:
* `common/param.f90`, which defines:
    * numerical precision (single or double)
    * simulation geometry e.g., 2D or 3D, the initial spacing between particles
    * material properties e.g., reference density and mass
    * time-step size
    * algorithm parameters (e.g., artificial viscosity parameters)
    * input and output paths
* an HDF5 file, which defines:
    * the number of real and virtual particles
    * each real and virtual particles':
        * position
        * velocity
        * density
        * pressure
        * ID
        * type index

## The input HDF5 file
The arrangement of the HDF5 file is
```
/
├── real (group) 
│   ├── n     (attribute)  # number of real particles, size: 1,       type: integer
│   ├── ind   (dataset)    # particle ID,              size: n,       type: integer
│   ├── type  (dataset)    # particle type index,      size: n,       type: integer
│   ├── p     (dataset)    # pressure,                 size: n,       type: float64
│   ├── rho   (dataset)    # density,                  size: n,       type: float64
│   ├── v     (dataset)    # velocity,                 size: n x dim, type: float64
│   └── x     (dataset)    # position,                 size: n x dim, type: float64 
└── virt (group)
    ├── n     (attribute)
    ├── ind   (dataset)  
    ├── type  (dataset)  
    ├── p     (dataset)  
    ├── rho   (dataset)  
    ├── v     (dataset)  
    └── x     (dataset)
```
The two groups `real` and `virt` describe the properties of virtual and real particles. The `n` attribute in each group defines how many particles are in each group. The datasets in the groups describe each of the particles' properties. When writing the HDF5 file, ensure the dataset sizes are congruent with `dim` in `common/param.f90` and `n` in each group, or your results won't be as expected or the GraSPH will fail. 

## Examples
`example/dambreak.h5` is an input file that comes with the GraSPH code. The `common/param.f90` file points to this file via the `input_file` parameter. After installing HDF5, you can inspect the input file with `h5dump`. For example, printing only the `/real/n` and `/virt/n` attributes, i.e., the number of real and virtual particles, respectively:
```
$ h5dump -a /real/n -a /virt/n example/dambreak.h5 

HDF5 "/home/edwardy/GraSPH/example/dambreak.h5" {
ATTRIBUTE "n" {
   DATATYPE  H5T_STD_I32LE
   DATASPACE  SCALAR
   DATA {
   (0): 62500
   }
}
ATTRIBUTE "n" {
   DATATYPE  H5T_STD_I32LE
   DATASPACE  SCALAR
   DATA {
   (0): 188112
   }
}
}
```
we can see that there are 62,500 real particles and 188,112 virtual particles. You can check that they match with the datasets with
```
$ h5dump -H ~/GraSPH/example/dambreak.h5 
HDF5 "/home/edwardy/GraSPH/example/dambreak.h5" {
GROUP "/" {
   GROUP "real" {
      ATTRIBUTE "n" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SCALAR
      }
      DATASET "ind" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 62500 ) / ( 62500 ) }
      }
      DATASET "p" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 62500 ) / ( 62500 ) }
      }
      DATASET "rho" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 62500 ) / ( 62500 ) }
      }
      DATASET "type" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 62500 ) / ( 62500 ) }
      }
      DATASET "v" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 62500, 3 ) / ( 62500, 3 ) }
      }
      DATASET "x" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 62500, 3 ) / ( 62500, 3 ) }
      }
   }
   GROUP "virt" {
      ATTRIBUTE "n" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SCALAR
      }
      DATASET "ind" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 188112 ) / ( 188112 ) }
      }
      DATASET "p" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 188112 ) / ( 188112 ) }
      }
      DATASET "rho" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 188112 ) / ( 188112 ) }
      }
      DATASET "type" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 188112 ) / ( 188112 ) }
      }
      DATASET "v" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 188112, 3 ) / ( 188112, 3 ) }
      }
      DATASET "x" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 188112, 3 ) / ( 188112, 3 ) }
      }
   }
}
}
```
which shows that all the datasets in group `real` are of size 62500 x 3 i.e., 62500 real particles and 3 spatial dimensions; and group 'virt' are of size 118,112 x 3; which both match the `n` attributes inspected earlier, as well as the `dim` parameter in `common/param.f90`

Accompanying this input file are example scripts in Python and Matlab that can be used to generate this input file: `dambreak-create-input.py` and `dambreak-create-input.m`. You can modify these scripts to create your own input files. But remember to modify the `common/param.f90` file to match.