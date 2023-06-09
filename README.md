# XNN
A foundational C library for neural network and deep learning.

XNN (not to be confused with other similarly named libraries) is a multithread C library for neural network and deep learning written from scratch (GPU not yet supported). Compiling only requires a POSIX environment (just use msys2 on Windows folks), no other third party libraries needed. In addition, compiling samples requires libpng.

This page only includes the source code for XNN. For the complete project, including tests, MNIST dataset in png format, refer to the release page at https://github.com/xiongyunuo/XNN/releases.

Supported neural network architecture: fully connected neural network, dune neural network

Supported layer: Linear, ReLU, Softmax

Supported optimization algorithms: SGD, momentum, RMSProp, Adam, MOLD, AdaMOLD

Supported input representation: vector, Magics
