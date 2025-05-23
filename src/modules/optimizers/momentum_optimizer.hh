#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "src/modules/optimizers/optimizer.hh"

namespace aurelius
{
    namespace optimizers
    {

        class MomentumOptimizer : public Optimizer
        {
        private:
            float beta1;
            Eigen::MatrixXf m_weights_momentum;
            Eigen::VectorXf m_bias_momentum;
            bool initialized = false;

        public:
            MomentumOptimizer(float b1 = 0.9f) : beta1(b1) {}

            void initialize_state_if_needed(const Eigen::MatrixXf &weights, const Eigen::VectorXf &bias)
            {
                if (!initialized)
                {
                    m_weights_momentum = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
                    m_bias_momentum = Eigen::VectorXf::Zero(bias.size());
                    initialized = true;
                }
            }

            void update_params(float learning_rate,
                               Eigen::MatrixXf &weights, const Eigen::MatrixXf &grad_weights,
                               Eigen::VectorXf &bias, const Eigen::VectorXf &grad_bias) override
            {
                initialize_state_if_needed(weights, bias);

                m_weights_momentum = beta1 * m_weights_momentum + grad_weights;
                m_bias_momentum = beta1 * m_bias_momentum + grad_bias;

                weights -= learning_rate * m_weights_momentum;
                bias -= learning_rate * m_bias_momentum;
            }

            void reset_state() override
            {
                initialized = false;
            }
        };

    }
}