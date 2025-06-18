#pragma once

#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <chrono>

namespace aurelius
{

    class Layer
    {

    public:
        Layer() = default;
        virtual ~Layer() = default;
        std::shared_ptr<Layer> previous = nullptr;
        std::weak_ptr<Layer> next;
        virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &input) = 0;
        virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) = 0;

    protected:
        bool use_avx = true;
        Eigen::MatrixXf last_input;
        Eigen::MatrixXf last_output;
    };

}