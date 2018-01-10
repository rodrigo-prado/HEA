#!/bin/bash

export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

path=$(pwd)
rm   "$path/source/include"
mkdir  "$path/bin"
ln -s "$path/include" "$path/source/include"

(cd "$path/build"; rm -rf CMakeFiles; rm CMakeCache.*; rm cmake*; rm Makefile; cmake -DCMAKE_CUDA_FLAGS='-arch=sm_61 -O3 -Xcompiler -fopenmp -std=c++11' ../source; make -j4; mv HEA_cuda ../bin/)

# If everything is ok, the exec file will be written on folder .bin/
