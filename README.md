# Aurelius : Neural Network with C++ SIMD

## Overview
This project is a neural network implementation in C++ utilizing SIMD (Single Instruction, Multiple Data) for optimized performance. The goal is to leverage SIMD instructions to accelerate computations, making the neural network more efficient.

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

## Usage
Once built, you can run the executable:
```sh
./neural_net_simd
```

## Dependencies
- C++17 or later
- CMake (for build configuration)

## Roadmap
- [ ] Complete attention mechanism implementation
- [ ] Add more layers with SIMD acceleration
- [ ] Benchmark performance against non-SIMD implementations

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For any questions or discussions, feel free to reach out.


