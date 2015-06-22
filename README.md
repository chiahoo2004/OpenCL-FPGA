# Image Detail Enhancement Using OpenCL
## Objective
This project tries to implement using 3 filtering algorithm with OpenCL.
The output of filtering algorithm can be used for image detail enhancement.
In order to compare OpenCL on CPU, GPU and FPGA.

## Build and Run
At the git repo root

    mkdir build
    cd build
    cmake ../src && make

After build the project, the related OpenCL files must be copied along with
the binary. This should be fixed in the future.
