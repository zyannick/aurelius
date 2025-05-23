#include <stdio.h>

#ifdef __AVX__
#pragma message "AVX is detected by the compiler!"
#endif

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>



int main() {

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(3, 3).cwiseAbs();
    std::cout << "Random Matrix:\n" << matrix.array() << std::endl;
    std::cout << "Random Matrix sqrt :\n" << matrix.array().sqrt() << std::endl;
}
