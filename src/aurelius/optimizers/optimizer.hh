#pragma once
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
namespace aurelius
{
    namespace optimizers
    {

        class Optimizer
        {
        public:
            virtual ~Optimizer() = default;

            virtual void update_params(float lr,
                                       Eigen::MatrixXf &W, const Eigen::MatrixXf &dW,
                                       Eigen::VectorXf &b, const Eigen::VectorXf &db)
            {
                update_param(lr, W, dW);
                update_param(lr, b, db);
            }

            virtual void update_param(float lr,
                                      Eigen::MatrixXf &param,
                                      const Eigen::MatrixXf &grad) = 0;

            virtual void update_param(float lr,
                                      Eigen::VectorXf &param,
                                      const Eigen::VectorXf &grad) = 0;

            virtual std::unique_ptr<Optimizer> clone() const = 0;

            virtual void reset_state() {}

            float weight_decay = 0.0f;
        };

    }
}