# Demonstration of topology optimization given Stokes flow using the SiMPL method.

This example demonstrates how to solve a Topology Optimization problem (TO) given constraints
from Stokes flow using the SiMPL methodology.


## Run the example

**NOTE:** If you built a serial version of MFEM, comment out the `find_package(MPI REQUIRED)` line in `CMakeLists.txt` file.

To build, use `cmake` with the following command:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DMFEM_DIR=<PATH_TO_MFEM_BUILD> -S . -B <PATH_TO_BUILD>
```
Note that you need to build main mfem library first using `cmake`.

Then make the stokes to example:
```bash
make stokes_to
```
Run with 
```bash
./stokes -r 6 -d 1.5
```
