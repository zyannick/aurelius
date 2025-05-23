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

        class AdamOptimizer : public Optimizer
        {
        private:
            float beta1;
            float beta2;
            float epsilon;
            int t = 0;

            Eigen::MatrixXf m_weights;
            Eigen::VectorXf m_bias;
            Eigen::MatrixXf v_weights;
            Eigen::VectorXf v_bias;
            bool initialized = false;

        public:
            AdamOptimizer(float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
                : beta1(b1), beta2(b2), epsilon(eps) {}

            void initialize_state_if_needed(const Eigen::MatrixXf &weights_shape, const Eigen::VectorXf &bias_shape)
            {
                if (!initialized)
                {
                    m_weights = Eigen::MatrixXf::Zero(weights_shape.rows(), weights_shape.cols());
                    m_bias = Eigen::VectorXf::Zero(bias_shape.size());
                    v_weights = Eigen::MatrixXf::Zero(weights_shape.rows(), weights_shape.cols());
                    v_bias = Eigen::VectorXf::Zero(bias_shape.size());
                    t = 0;
                    initialized = true;
                }
            }

            void update_params(float learning_rate,
                               Eigen::MatrixXf &weights, const Eigen::MatrixXf &grad_weights,
                               Eigen::VectorXf &bias, const Eigen::VectorXf &grad_bias) override
            {
                initialize_state_if_needed(weights, bias);

                t++;

                m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
                m_bias = beta1 * m_bias + (1.0f - beta1) * grad_bias;

                v_weights = beta2 * v_weights.array() + (1.0f - beta2) * grad_weights.array().square();
                v_bias = beta2 * v_bias.array() + (1.0f - beta2) * grad_bias.array().square();

                float beta1_t = std::pow(beta1, t);
                float beta2_t = std::pow(beta2, t);

                float alpha_t = learning_rate;
                if (beta1_t < 1.0f)
                {
                    alpha_t = learning_rate * std::sqrt(1.0f - beta2_t) / (1.0f - beta1_t);
                }

                Eigen::MatrixXf m_weights_hat = m_weights / (1.0f - beta1_t);
                Eigen::VectorXf m_bias_hat = m_bias / (1.0f - beta1_t);
                Eigen::MatrixXf v_weights_hat = v_weights / (1.0f - beta2_t);
                Eigen::VectorXf v_bias_hat = v_bias / (1.0f - beta2_t);

                Eigen::MatrixXf denom_weights = v_weights_hat.array().sqrt() + epsilon;
                Eigen::VectorXf denom_bias = v_bias_hat.array().sqrt() + epsilon;

                weights = weights.array() - learning_rate * m_weights_hat.array() / denom_weights.array();
                bias = bias.array() - learning_rate * m_bias_hat.array() / denom_bias.array();
            }

            void reset_state() override
            {
                initialized = false;
                t = 0;
                m_weights = Eigen::MatrixXf();
                m_bias = Eigen::VectorXf();
                v_weights = Eigen::MatrixXf();
                v_bias = Eigen::VectorXf();
            }
        };

    }
}