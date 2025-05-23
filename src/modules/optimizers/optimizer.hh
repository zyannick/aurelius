#include <immintrin.h>        
#include <eigen3/Eigen/Dense> 
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#define EIGEN_USE_THREADS

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept> 
#include <memory>   
#include <chrono>    

constexpr int ALIGNMENT = 32;


class Optimizer
{
public:
    virtual ~Optimizer() = default;

    virtual void update_params(float learning_rate,
                               Eigen::MatrixXf &weights, const Eigen::MatrixXf &grad_weights,
                               Eigen::VectorXf &bias, const Eigen::VectorXf &grad_bias) = 0;

    virtual void reset_state() {}
};


