#!/bin/bash
g++ video_test.cpp -o uselib_v2  -std=c++11 -std=c++11 -Iinclude/ -I3rdparty/stb/include -DOPENCV `pkg-config --cflags opencv` -DGPU -I/usr/local/cuda/include/ -DCUDNN -Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC -Ofast -DOPENCV -fopenmp -DGPU -DCUDNN -I/usr/local/cudnn/include -fPIC -lm -pthread `pkg-config --libs opencv` -lgomp -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -L/usr/local/cudnn/lib64 -lcudnn -lstdc++ -L ./ -l:libdarknet.so 
./uselib_v2
