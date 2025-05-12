#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

constexpr int ALIGNMENT = 32;

struct Conv2d
{
    int in_channels, out_channels, kernel_size, stride, padding;
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    Eigen::MatrixXf last_input;
    Eigen::MatrixXf weight_gradients;
    Eigen::VectorXf bias_gradients;
    bool use_avx = true;

    Conv2d(int in_c, int out_c, int k_s, int s, int p) : in_channels(in_c), out_channels(out_c), kernel_size(k_s), stride(s), padding(p)
    {
        weights = Eigen::MatrixXf::Random(out_c, in_c * k_s * k_s);
        bias = Eigen::VectorXf::Zero(out_c);
        weight_gradients = Eigen::MatrixXf::Zero(out_c, in_c * k_s * k_s);
        bias_gradients = Eigen::VectorXf::Zero(out_c);
    }

    Eigen::MatrixXf forward_simd(const Eigen::MatrixXf &input)
    {
        int output_height = (input.rows() - kernel_size + 2 * padding) / stride + 1;
        int output_width = (input.cols() - kernel_size + 2 * padding) / stride + 1;
        Eigen::MatrixXf output(out_channels, output_height * output_width);
        for (int i = 0; i < out_channels; ++i)
        {
            for (int b = 0; b < output_height * output_width; ++b)
            {
                __m256 sum = _mm256_setzero_ps();
                int j = 0;
                for (; j + 8 <= in_channels * kernel_size * kernel_size; j += 8)
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
                for (; j < in_channels * kernel_size * kernel_size; ++j)
                {
                    dot_product += weights(i, j) * input(j, b * stride);
                }
            }
        }
        return output;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        int output_height = (input.rows() - kernel_size + 2 * padding) / stride + 1;
        int output_width = (input.cols() - kernel_size + 2 * padding) / stride + 1;
        Eigen::MatrixXf output(out_channels, output_height * output_width);
        for (int i = 0; i < out_channels; ++i)
        {
            for (int b = 0; b < output_height * output_width; ++b)
            {
                float sum = 0.0f;
                for (int j = 0; j < in_channels * kernel_size * kernel_size; ++j)
                {
                    sum += weights(i, j) * input(j, b * stride);
                }
                output(i, b) = sum + bias(i);
            }
        }
        return output;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output)
    {
        int input_height = last_input.rows();
        int input_width = last_input.cols();
        Eigen::MatrixXf grad_input(input_height, input_width);
        grad_input.setZero();

        for (int i = 0; i < out_channels; ++i)
        {
            for (int b = 0; b < grad_output.cols(); ++b)
            {
                for (int j = 0; j < in_channels * kernel_size * kernel_size; ++j)
                {
                    grad_input(j, b * stride) += weights(i, j) * grad_output(i, b);
                    weight_gradients(i, j) += last_input(j, b * stride) * grad_output(i, b);
                }
                bias_gradients(i) += grad_output(i, b);
            }
        }
        return grad_input;
    }

    void update(float learning_rate)
    {
        weights -= learning_rate * weight_gradients;
        bias -= learning_rate * bias_gradients;
    }

    void set_use_avx(bool flag)
    {
        use_avx = flag;
    }
    
    int get_in_features() const { return in_channels; }
    int get_out_features() const { return out_channels; }
    Eigen::MatrixXf get_weights() const { return weights; }
    Eigen::VectorXf get_bias() const { return bias; }
};