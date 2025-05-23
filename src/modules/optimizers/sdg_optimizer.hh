#include <immintrin.h>        
#include <eigen3/Eigen/Dense> 
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept> 
#include <memory>   
#include <chrono>    
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#define EIGEN_USE_THREADS

#include "src/modules/optimizers/optimizer.hh"

constexpr int ALIGNMENT = 32;

class SGDOptimizer : public Optimizer
{
public:
    SGDOptimizer() = default;

    void update_params(float learning_rate,
                       Eigen::MatrixXf &weights, const Eigen::MatrixXf &grad_weights,
                       Eigen::VectorXf &bias, const Eigen::VectorXf &grad_bias) override
    {
        weights -= learning_rate * grad_weights;
        bias -= learning_rate * grad_bias;
    }
};