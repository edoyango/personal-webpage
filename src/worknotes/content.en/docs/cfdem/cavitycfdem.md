---
title: OpenFOAM cavity case to CFDEM
weight: 1
---

# Converting the OpenFOAM cavity example for CFDEM

The cavity case is a good one to start with as it forms the beginning of the 
[official OpenFOAM tutorials](https://www.openfoam.com/documentation/tutorial-guide).
These steps assume we're using the PUBLIC version of CFDEM which couples
LIGGGHTS-PUBLIC 3.8.0 and OpenFOAM-5.x. It also assumes that your environment
variables have already been setup as per the 
[CFDEM insallation instructions](https://www.cfdem.com/media/CFDEM/docu/CFDEMcoupling_Manual.html#installation).

## Getting the case files

The lid-driven cavity flow example case comes with the OpenFOAM source code. If
you don't have it already from install OpenFOAM, get it by:

```bash
git clone https://github.com/OpenFOAM/OpenFOAM-5.x.git
```

and the cavity tutorial files are located in 
`OpenFOAM-5.x/tutorials/incompressible/icoFoam/cavity/cavity`.

## 1. Set up the directory structure

In the directory of your choice, setup the directory structure. You will need a
"DEM" folder and a "CFD" folder. Initialize the CFD folder with the files from
the cavity OpenFOAM example.

```bash
# create a cavity-cfdem directory to store all the case files, and the DEM
# subdirectory
mkdir -p cavity-cfdem/DEM

# copy the cavity case into the CFD folder
cp -r OpenFOAM-5.x/tutorials/incompressible/icoFoam/cavity/cavity cavity-cfdem/CFD

# move into the case folder
cd cavity-cfdem
```

## 2. Copy a CFDEM executor wrapper script and modify

From the CFDEM tutorial files, get a `parCFDDEMrun.sh` script.

```bash
cp "$CFDEM_PROJECT_DIR"/tutorials/cfdemSolverIB/twoSpheresGlowinskiMPI/{Allrun.sh,parCFDDEMrun.sh} .
```

The `Allrun.sh` script is fine as-is, but we need to modify the `parCFDDEMrun.sh`
script. From the existing file, change the corresponding variables to match
what's below:

```bash
headerTest=cfdemSolverIB_cavity_CFDEM
runOctave="false"
```

You may also want to set the `nrProcs` variable to suit your computer. The case
we will be building is also quite small, so I will choose `nrProcs=2`. 
The remaining variables can remain the same. Note that we will still be using
the "Immersed Boundary" CFDEM solver.

## 3. Update OpenFOAM mesh

### 3a. Make the mesh finer

The cavity example case uses a 2D grid of cells which posesses 20 cells in both
directions. This is a little course, so we will double the number of cells in
each direction. Do this by changing the `blocks` dictionary to

```cpp
blocks
(
    hex (0 1 2 3 4 5 6 7) (40 40 1) simpleGrading (1 1 1)
);
```

Everything else can remain the same.

### 3b. Add the mesh decomposition dictionary

Create the `CFD/system/decomposeParDict` file with the contents below:

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
|       o          |                                                          |
|    o     o       | HELYX-OS                                                  |
|   o   O   o      | Version: v2.4.0                                           |
|    o     o       | Web:     http://www.engys.com                            |
|       o          |                                                          |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location system;
    object decomposeParDict;
}

    numberOfSubdomains 2;
    method simple;
    simpleCoeffs
    {
        n ( 2 1 1);
        delta 0.001;
    }
```

This ensures the decomposition is done on two processors. Modify this accordingly
for the number of processors you would like to run this case on.

## 4. Update initial conditions

In the original cavity example, only initial boundary velocity and pressure needs to be
defined. To adapt this for CFDEM, initial boundary conditions for the variables `phiIB`,
 `Us`, and `voidfraction` need to be defined.

### phiIB
```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  5                                     |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      phiB;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }

    fixedWalls
    {
        type            zeroGradient;
    }

    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
```

### Us

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  5                                     |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      Us;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }

    fixedWalls
    {
        type            zeroGradient;
    }

    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
```

### voidfraction

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  5                                     |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      voidfraction;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 1;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }

    fixedWalls
    {
        type            zeroGradient;
    }

    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
```

## 5. Update solver schemes

These are controlled by the `CFD/system/fvSchemes` dictionary. This will exist
already, but need schemes related to the additional parameters we've added. Do
so by modifying the file to match below:

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         Gauss linear;
    div(phi,U)      Gauss limitedLinearV 1;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,epsilon) Gauss limitedLinear 1;
    div(phi,R)      Gauss limitedLinear 1;
    div(R)          Gauss linear;
    div(phi,nuTilda) Gauss limitedLinear 1;
    div((nuEff*dev(grad(U).T()))) Gauss linear;
    div(U)          Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
    laplacian(nuEff,U) Gauss linear corrected;
    laplacian((1|A(U)),p) Gauss linear corrected;
    laplacian((voidfraction2|A(U)),p) Gauss linear corrected;
    laplacian(DkEff,k) Gauss linear corrected;
    laplacian(DepsilonEff,epsilon) Gauss linear corrected;
    laplacian(DREff,R) Gauss linear corrected;
    laplacian(DnuTildaEff,nuTilda) Gauss linear corrected;
    laplacian(phiIB) Gauss linear corrected;
    laplacian(U) Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
    interpolate(U)  linear;
}

snGradSchemes
{
    default         corrected;
}

fluxRequired
{
    default         no;
    p               ;
}


// ************************************************************************* //
```

## 6. Update fvSolution

This needs to be modified to ensure solvers are added for the turbulence and
CFDEM parameters being used with the `cfdemSolverIB` solver.

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.1;
    }

    pFinal
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0;
    }

    U
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0;
    }

    k
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0;
    }

    epsilon
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0;
    }

    R
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0;
    }

    nuTilda
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0;
    }

    phiIB
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0;
    }
}

PISO
{
    nCorrectors     4;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}


// ************************************************************************* //
```

## 7. Add constants

### transportProperties

This will already exist from the original case. Modify it to match below:

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


transportModel  Newtonian;

nu              nu [ 0 2 -1 0 0 0 0 ] 0.01;//0.111426;//1.875e-03;//7.5e-03;//0.265883;

CrossPowerLawCoeffs
{
    nu0             nu0 [ 0 2 -1 0 0 0 0 ] 1e-06;
    nuInf           nuInf [ 0 2 -1 0 0 0 0 ] 1e-06;
    m               m [ 0 0 1 0 0 0 0 ] 1;
    n               n [ 0 0 0 0 0 0 0 ] 1;
}

BirdCarreauCoeffs
{
    nu0             nu0 [ 0 2 -1 0 0 0 0 ] 1e-06;
    nuInf           nuInf [ 0 2 -1 0 0 0 0 ] 1e-06;
    k               k [ 0 0 1 0 0 0 0 ] 0;
    n               n [ 0 0 0 0 0 0 0 ] 1;
}

// ************************************************************************* //
```

### g

the `cfdemSolverIB` solver needs gravity to be described. We won't apply a
meaningful gravity here, so we are assigning a value of 0 with the file below.

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 0);
//value           (0 0 0);

// ************************************************************************* //
```

### RASProperties

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      RASProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

RASModel        laminar;//kEpsilon;

turbulence      on;

printCoeffs     on;


// ************************************************************************* //
```

### turbulenceProperties

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//simulationType  RASModel;//OFversion24x
simulationType      laminar;//OFversion30x


// ************************************************************************* //
```

### couplingProperties

```cpp
/*---------------------------------------------------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.4                                   |
|   \\  /    A nd           | Web:      http://www.openfoam.org               |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/


FoamFile
{
    version         2.0;
    format          ascii;

    root            "";
    case            "";
    instance        "";
    local           "";

    class           dictionary;
    object          couplingProperties;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//===========================================================================//
// sub-models & settings

modelType none;

couplingInterval 10;

depth 0;

voidFractionModel IB;//bigParticle;//centre; //

locateModel engineIB;//standard;//

meshMotionModel noMeshMotion;

dataExchangeModel twoWayMPI;//twoWayFiles;

IOModel basicIO;

probeModel off;

averagingModel dilute;

clockModel off;

smoothingModel off;

forceModels
(
    ShirgaonkarIB
    ArchimedesIB
);

momCoupleModels
(
);

//turbulenceModelType RASProperties;//LESProperties; //OFversion24x
turbulenceModelType turbulenceProperties; //OFversion30x

//===========================================================================//
// sub-model properties

ShirgaonkarIBProps
{
    velFieldName "U";
    pressureFieldName "p";
}

ArchimedesIBProps
{
    gravityFieldName "g";
    voidfractionFieldName "voidfractionNext";
}

twoWayFilesProps
{
    maxNumberOfParticles 2;
    DEMts 0.0002;
}

twoWayMPIProps
{
    maxNumberOfParticles 2;
    liggghtsPath "../DEM/in.liggghts_run";
}

IBProps
{
    maxCellsPerParticle 1000;
    alphaMin 0.30;
    scaleUpVol 1.0;
}

bigParticleProps
{
    maxCellsPerParticle 1000;
    alphaMin 0.30;
    scaleUpVol 1.0;
}
centreProps
{
    alphaMin 0.30;
}

dividedProps
{
    alphaMin 0.05;
    scaleUpVol 1.2;
}

engineIBProps
{
    treeSearch false;
    zSplit 8;
    xySplit 16;
}

// ************************************************************************* //
```

### dynamicMeshDict

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.6                                   |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      dynamicMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dynamicFvMesh   dynamicRefineFvMesh;//staticFvMesh;//

dynamicRefineFvMeshCoeffs
{
    refineInterval  1;//refine every refineInterval timesteps
    field           interFace;
    lowerRefineLevel .0001;
    upperRefineLevel 0.99;
    unrefineLevel   10;
    nBufferLayers   1;
    maxRefinement   1;//maximum refinement level (starts from 0)
    maxCells        1000000;
    correctFluxes
    (
        (phi    U)
        (phi_0  U)
    );
    dumpLevel       false;
}


// ************************************************************************* //
```

### liggghtsCommands

```cpp
/*---------------------------------------------------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.4                                   |
|   \\  /    A nd           | Web:      http://www.openfoam.org               |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/


FoamFile
{
    version         2.0;
    format          ascii;

    root            "";
    case            "";
    instance        "";
    local           "";

    class           dictionary;
    object          liggghtsCommands;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

liggghtsCommandModels
(
   runLiggghts
);

// ************************************************************************* //
```

## 8. Write the LIGGGHTS input file

```
atom_style      granular
atom_modify     map array
communicate     single vel yes

boundary        f f f
newton          off

units           si
processors      * * 1

region          reg block -0.01 0.11 -.01 0.11 -.01 0.03 units box
create_box      1 reg

neigh_modify    delay 0 binsize 0.0


# Material properties required for new pair styles

fix             m1 all property/global youngsModulus peratomtype 5.e7
fix             m2 all property/global poissonsRatio peratomtype 0.45
fix             m3 all property/global coefficientRestitution peratomtypepair 1 0.9
fix             m4 all property/global coefficientFriction peratomtypepair 1 0.5

# pair style
pair_style      gran model hertz tangential history #Hertzian without cohesion
pair_coeff      * *

# timestep, gravity
timestep        0.0002

fix             gravi all gravity 0 vector 0.0 0.0 -1.0

# walls
fix     xwalls1 all wall/gran model hertz tangential history primitive type 1 xplane 0.
fix     xwalls2 all wall/gran model hertz tangential history primitive type 1 xplane 0.1
fix     ywalls1 all wall/gran model hertz tangential history primitive type 1 yplane 0.
fix     ywalls2 all wall/gran model hertz tangential history primitive type 1 yplane 0.1 shear x 1
fix     zwalls1 all wall/gran model hertz tangential history primitive type 1 zplane 0.
fix     zwalls2 all wall/gran model hertz tangential history primitive type 1 zplane 0.1

# cfd coupling
fix     cfd  all couple/cfd couple_every 10 mpi
fix     cfd2 all couple/cfd/force


# create single partciles
create_atoms 1 single .05 .025 0.01
create_atoms 1 single .05 .075 0.01
set atom 1 diameter 0.005 density 2600 vx 0 vy 0 vz 0
set atom 2 diameter 0.005 density 2600 vx 0 vy 0 vz 0

# apply nve integration to all particles that are inserted as single particles
fix             integr all nve/sphere #wenn das ausgeblendet, dann kein vel update

# screen output
compute         rke all erotate/sphere
thermo_style    custom step atoms ke c_rke vol
thermo          1000
thermo_modify   lost ignore norm no
compute_modify  thermo_temp dynamic yes

# insert the first particles so that dump is not empty
dump		    dmp2 all custom/vtk 50 post/dump*.vtk id type type x y z vx vy vz radius

run             1
```