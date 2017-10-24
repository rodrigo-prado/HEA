#!/bin/bash

export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

(cd source; make; mv HEA_cuda ../bin/)

# If everything is ok, the exec file will be written on folder .bin/