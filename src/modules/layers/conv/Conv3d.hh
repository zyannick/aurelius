#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

constexpr int ALIGNMENT = 32;

struct Conv3d
{
    int in_channels, out_channels, kernel_size, stride, padding;
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    Eigen::MatrixXf last_input;
    Eigen::MatrixXf weight_gradients;
    Eigen::VectorXf bias_gradients;
    bool use_avx = true;

    Conv3d(int in_c, int out_c, int k_s, int s, int p) : in_channels(in_c), out_channels(out_c), kernel_size(k_s), stride(s), padding(p)
    {
        weights = Eigen::MatrixXf::Random(out_c, in_c * k_s * k_s * k_s);
        bias = Eigen::VectorXf::Zero(out_c);
        weight_gradients = Eigen::MatrixXf::Zero(out_c, in_c * k_s * k_s * k_s);
        bias_gradients = Eigen::VectorXf::Zero(out_c);
    }

    Eigen::MatrixXf forward_simd(const Eigen::MatrixXf &input)
    {
        int output_depth = (input.rows() - kernel_size + 2 * padding) / stride + 1;
        int output_height = (input.cols() - kernel_size + 2 * padding) / stride + 1;
        int output_width = (input.cols() - kernel_size + 2 * padding) / stride + 1;
        Eigen::MatrixXf output(out_channels, output_depth * output_height * output_width);
        for (int i = 0; i < out_channels; ++i)
        {
            for (int b = 0; b < output_depth * output_height * output_width; ++b)
            {
                __m256 sum = _mm256_setzero_ps();
                int j = 0;
                for (; j + 8 <= in_channels * kernel_size * kernel_size * kernel_size; j += 8)
                {
                    __m256 input_vec = _mm256_loadu_ps(input.col(b * stride + j).data());
                    __m256 weight_vec = _mm256_loadu_ps(weights.row(i).data() + j);
                    sum = _mm256_fmadd_ps(weight_vec, input_vec, sum);
                }

                // Sum up 8 floats in the SIMD register
                alignas(ALIGNMENT) float temp[8];
                _mm256_store_ps(temp, sum);
                float dot_product = 0.0f;
                for (int k = 0; k < 8; ++k)
                {
                    dot_product += temp[k];
                }

                // Handle remaining elements (if in_channels * kernel_size % 8 != 0)
                for (; j < in_channels * kernel_size * kernel_size * kernel_size; ++j)
                {
                    dot_product += weights(i, j) * input(j, b * stride);
                }
                output(i, b) = dot_product + bias(i);
            }
        }
        return output;
    }

    Eigen::MatrixXf forward_vectorized(const Eigen::MatrixXf &input)
    {
        int output_depth = (input.rows() - kernel_size + 2 * padding) / stride + 1;
        int output_height = (input.cols() - kernel_size + 2 * padding) / stride + 1;
        int output_width = (input.cols() - kernel_size + 2 * padding) / stride + 1;
        Eigen::MatrixXf output(out_channels, output_depth * output_height * output_width);
        for (int i = 0; i < out_channels; ++i)
        {
            for (int b = 0; b < output_depth * output_height * output_width; ++b)
            {
                float dot_product = 0.0f;
                for (int j = 0; j < in_channels * kernel_size * kernel_size * kernel_size; ++j)
                {
                    dot_product += weights(i, j) * input(j, b * stride);
                }
                output(i, b) = dot_product + bias(i);
            }
        }
        return output;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        if (use_avx)
        {
            return forward_simd(input);
        }
        else
        {
            return forward_vectorized(input);
        }
    }
    Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output)
    {
        Eigen::MatrixXf grad_input = weights.transpose() * grad_output;
        return grad_input;
    }
    void update_weights(const Eigen::MatrixXf &grad_input)
    {
        for (int i = 0; i < out_channels; ++i)
        {
            for (int j = 0; j < in_channels * kernel_size * kernel_size * kernel_size; ++j)
            {
                weight_gradients(i, j) += grad_input(i, j);
            }
            bias_gradients(i) += grad_input.row(i).sum();
        }
        weights -= weight_gradients;
        bias -= bias_gradients;
    }
    void set_use_avx(bool flag)
    {
        use_avx = flag;
    }
    void set_weights(const Eigen::MatrixXf &new_weights)
    {
        weights = new_weights;
    }
    void set_bias(const Eigen::VectorXf &new_bias)
    {
        bias = new_bias;
    }
    Eigen::MatrixXf get_weights() const
    {
        return weights;
    }
};