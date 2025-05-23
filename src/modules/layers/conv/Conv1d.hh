#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

#include "src/modules/layers/layer.hh"
#include "src/modules/optimizers/optimizer.hh"

using namespace aurelius::layers;

namespace aurelius
{
    namespace convolution
    {

        constexpr int ALIGNMENT = 32;

        class Conv1d : public ConvLayer
        {
        public:
            Conv1d(int in_c, int out_c, int k_s, int s, int p)
            {
                in_channels = in_c;
                out_channels = out_c;
                kernel_size = k_s;
                stride = s;
                padding = p;
                if (in_c <= 0 || out_c <= 0 || k_s <= 0 || s <= 0)
                {
                    throw std::invalid_argument("Input and output channels, kernel size, and stride must be positive.");
                }
                if (padding < 0)
                {
                    throw std::invalid_argument("Padding must be non-negative.");
                }
                weights = Eigen::MatrixXf::Random(out_c, in_c * k_s);
                bias = Eigen::VectorXf::Zero(out_c);
                weight_gradients = Eigen::MatrixXf::Zero(out_c, in_c * k_s);
                bias_gradients = Eigen::VectorXf::Zero(out_c);
            }

        private:
            Eigen::MatrixXf forward_simd(const Eigen::MatrixXf &input)
            {
                int output_length = (input.cols() - kernel_size + 2 * padding) / stride + 1;
                Eigen::MatrixXf output(out_channels, output_length);
                for (int i = 0; i < out_channels; ++i)
                {
                    for (int b = 0; b < output_length; ++b)
                    {
                        __m256 sum = _mm256_setzero_ps();

                        int j = 0;
                        for (; j + 8 <= in_channels * kernel_size; j += 8)
                        {
                            __m256 input_vec = _mm256_loadu_ps(input.col(b * stride + j).data());
                            __m256 weight_vec = _mm256_loadu_ps(weights.row(i).data() + j);
                            sum = _mm256_fmadd_ps(weight_vec, input_vec, sum);
                        }

                        alignas(ALIGNMENT) float temp[8];
                        _mm256_store_ps(temp, sum);
                        float dot_product = 0.0f;
                        for (int k = 0; k < 8; ++k)
                        {
                            dot_product += temp[k];
                        }

                        for (; j < in_channels * kernel_size; ++j)
                        {
                            dot_product += weights(i, j) * input(j, b * stride);
                        }

                        output(i, b) = dot_product + bias(i);
                    }
                }
                return output;
            }

            Eigen::MatrixXf forward_vectorized(const Eigen::MatrixXf &input_signal)
            {


                int L_in = input_signal.cols();
                int output_length = (L_in - kernel_size + 2 * padding) / stride + 1;
                if (output_length <= 0)
                {
                    throw std::runtime_error("Output length is non-positive. Check parameters.");
                }
                Eigen::MatrixXf output(out_channels, output_length);
                output.setZero(); 

                for (int oc = 0; oc < out_channels; ++oc)
                { 
                    for (int t_out = 0; t_out < output_length; ++t_out)
                    { 
                        float dot_product = 0.0f;
                        int t_in_start = t_out * stride - padding; 

                        for (int ic = 0; ic < in_channels; ++ic)
                        { 
                            for (int k = 0; k < kernel_size; ++k)
                            { 
                                int current_t_in = t_in_start + k;
                                float input_val = 0.0f;

                                if (current_t_in >= 0 && current_t_in < L_in)
                                {
                                    input_val = input_signal(ic, current_t_in);
                                }

                                int flat_kernel_idx = ic * kernel_size + k;
                                dot_product += weights(oc, flat_kernel_idx) * input_val;
                            }
                        }
                        output(oc, t_out) = dot_product + bias(oc);
                    }
                }
                return output;
            }

            Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
            {
                last_input = input;
                if (use_avx)
                {
                    return forward_simd(last_input);
                }
                else
                {
                    return forward_vectorized(last_input);
                }
            }

            Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output)
            {
                weight_gradients.setZero();
                bias_gradients.setZero();

                int L_in = last_input.cols();
                int output_length = grad_output.cols();
                for (int oc = 0; oc < out_channels; ++oc)
                {
                    for (int t_out = 0; t_out < output_length; ++t_out)
                    {
                        float go = grad_output(oc, t_out);
                        bias_gradients(oc) += go;

                        int t_in_start = t_out * stride - padding;
                        for (int ic = 0; ic < in_channels; ++ic)
                        {
                            for (int k = 0; k < kernel_size; ++k)
                            {
                                int current_t_in = t_in_start + k;
                                float input_val = 0.0f;

                                if (current_t_in >= 0 && current_t_in < L_in)
                                {
                                    input_val = last_input(ic, current_t_in);
                                }

                                int flat_kernel_idx = ic * kernel_size + k;
                                weight_gradients(oc, flat_kernel_idx) += go * input_val;
                            }
                        }
                    }
                }

                Eigen::MatrixXf grad_input(in_channels, L_in);
                grad_input.setZero();

                for (int ic = 0; ic < in_channels; ++ic)
                {
                    for (int t_in = 0; t_in < L_in; ++t_in)
                    {
                        float sum_grad = 0.0f;
                        for (int oc = 0; oc < out_channels; ++oc)
                        {
                            for (int k = 0; k < kernel_size; ++k)
                            {

                                if ((t_in + padding - k) % stride == 0)
                                {
                                    int t_out = (t_in + padding - k) / stride;
                                    if (t_out >= 0 && t_out < output_length)
                                    {
                                        int flat_kernel_idx = ic * kernel_size + k;
                                        sum_grad += grad_output(oc, t_out) * weights(oc, flat_kernel_idx);
                                    }
                                }
                            }
                        }
                        grad_input(ic, t_in) = sum_grad;
                    }
                }
                return grad_input;
            }

            void apply_gradients(float learning_rate) override
            {
                weights -= learning_rate * weight_gradients;
                bias -= learning_rate * bias_gradients;
            }
            void set_optimizer(std::unique_ptr<Optimizer> opt) override
            {
                layer_optimizer = std::move(opt);
            }
            void set_use_avx(bool flag) override
            {
                use_avx = flag;
            }
            bool get_use_avx() const override
            {
                return use_avx;
            }
            int get_in_features() const override
            {
                return in_channels;
            }
            int get_out_features() const override
            {
                return out_channels;
            }
            Eigen::MatrixXf get_weights() const override
            {
                return weights;
            }
            Eigen::VectorXf get_bias() const override
            {
                return bias;
            }
            void set_weights(const Eigen::MatrixXf &new_weights) override
            {
                weights = new_weights;
            }
            void set_bias(const Eigen::VectorXf &new_bias) override
            {
                bias = new_bias;
            }
            void reset_gradients()
            {
                weight_gradients.setZero();
                bias_gradients.setZero();
            }
        };
    }
}