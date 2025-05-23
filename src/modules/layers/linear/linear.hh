#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

#include "src/modules/layers/layer.hh"
#include "src/modules/optimizers/optimizer.hh"

namespace aurelius
{
    namespace layers
    {

        constexpr int ALIGNMENT = 32;

        class Linear : public Layer
        {
        public:
            Linear(int in_f, int out_f)
            {
                in_features = in_f;
                out_features = out_f;
                if (in_f <= 0 || out_f <= 0)
                {
                    throw std::invalid_argument("Input and output features must be positive.");
                }
                weights = Eigen::MatrixXf::Random(out_f, in_f);
                bias = Eigen::VectorXf::Zero(out_f);
                weight_gradients = Eigen::MatrixXf::Zero(out_f, in_f);
                bias_gradients = Eigen::VectorXf::Zero(out_f);
            }

            Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override
            {
                if (input.rows() != in_features)
                {
                    throw std::invalid_argument("Input features dimension mismatch in Linear::forward.");
                }
                last_input = input;
                if (use_avx && in_features >= 8)
                    return forward_simd(input);
                return forward_vectorized(input);
            }

            Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) override
            {
                if (grad_output.rows() != out_features || grad_output.cols() != last_input.cols())
                {
                    throw std::invalid_argument("Gradient output dimensions mismatch in Linear::backward.");
                }
                int batch_size = grad_output.cols();
                if (batch_size == 0)
                    return Eigen::MatrixXf::Zero(in_features, 0);

                float inv_batch_size = 1.0f / static_cast<float>(batch_size);
                Eigen::MatrixXf grad_input = weights.transpose() * grad_output;
                weight_gradients = (grad_output * last_input.transpose()) * inv_batch_size;
                bias_gradients = grad_output.rowwise().sum() * inv_batch_size;
                return grad_input;
            }

            void apply_gradients(float learning_rate) override
            {
                if (!layer_optimizer)
                {
                    throw std::runtime_error("Optimizer not set for Linear layer.");
                }
                layer_optimizer->update_params(learning_rate, weights, weight_gradients, bias, bias_gradients);
            }

            void set_optimizer(std::unique_ptr<Optimizer> opt) override
            {
                if (opt)
                {
                    opt->reset_state();
                }
                layer_optimizer = std::move(opt);
            }

            void set_use_avx(bool flag) { use_avx = flag; }
            bool get_use_avx() const { return use_avx; }
            int get_in_features() const { return in_features; }
            int get_out_features() const { return out_features; }
            Eigen::MatrixXf get_weights() const { return weights; }
            Eigen::VectorXf get_bias() const { return bias; }

            void set_weights(const Eigen::MatrixXf &new_weights) override
            {
                if (new_weights.rows() != out_features || new_weights.cols() != in_features)
                {
                    throw std::invalid_argument("Weight dimensions mismatch in set_weights");
                }
                weights = new_weights;
                if (layer_optimizer)
                    layer_optimizer->reset_state();
            }

            void set_bias(const Eigen::VectorXf &new_bias) override
            {
                if (new_bias.size() != out_features)
                {
                    throw std::invalid_argument("Bias dimensions mismatch in set_bias");
                }
                bias = new_bias;
                if (layer_optimizer)
                    layer_optimizer->reset_state();
            }

        private:
            Eigen::MatrixXf forward_simd(const Eigen::MatrixXf &input)
            {
                Eigen::MatrixXf output(out_features, input.cols());
                for (int i = 0; i < out_features; ++i)
                {
                    for (int b = 0; b < input.cols(); ++b)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        int j = 0;
                        for (; j + 8 <= in_features; j += 8)
                        {
                            __m256 input_vec = _mm256_loadu_ps(input.col(b).data() + j);
                            __m256 weight_vec = _mm256_loadu_ps(weights.row(i).data() + j);
                            sum = _mm256_fmadd_ps(weight_vec, input_vec, sum);
                        }

                        float temp[8];
                        _mm256_storeu_ps(temp, sum);
                        float dot_product = 0.0f;
                        for (int k = 0; k < 8; ++k)
                        {
                            dot_product += temp[k];
                        }

                        for (; j < in_features; ++j)
                        {
                            dot_product += weights(i, j) * input(j, b);
                        }

                        output(i, b) = dot_product + bias(i);
                    }
                }
                return output;
            }

            Eigen::MatrixXf forward_vectorized(const Eigen::MatrixXf &input)
            {
                return (weights * input).colwise() + bias;
            }
        };
    }
}