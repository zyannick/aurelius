# Aurelius : Neural Network with C++ SIMD

## Overview
This project is a neural network implementation in C++ utilizing SIMD (Single Instruction, Multiple Data) for optimized performance. The goal is to leverage SIMD instructions to accelerate computations.

## Features
- Implementation of core neural network components
- SIMD optimization for faster computations
- Modular and extensible code structure
- Current work in progress: implementing the attention mechanism

## Installation
To build the project, ensure you have a C++ compiler that supports SIMD instructions (e.g., GCC, Clang, or MSVC) and CMake installed.

### Build Instructions
```sh
mkdir build
cd build
cmake ..
make
```

## Dependencies
- C++17 or later
- CMake (for build configuration)

## Roadmap
- [x] Linear Layer
- [x] Convolution Layer
- [x] Add optimizer
- [x] Add most common schedulers such as steps, linear, cosine, reduce on plateau-
- [ ] Complete attention mechanism implementation
- [ ] Add more layers with SIMD acceleration
- [ ] Benchmark performance against non-SIMD implementations
- [ ] Setup a AlexNet model from scratch
- [ ] Setup a Variational Auto Encoder from scratch

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For any questions or discussions, feel free to reach out.


