#include <immintrin.h>        // For AVX
#include <eigen3/Eigen/Dense> // For dense matrices and vectors
#include <iostream>
#include <vector>
#include <cmath>
#include "modules/layer.hh"

constexpr int ALIGNMENT = 32;

struct Linear : public Layer
{
    int in_features, out_features;
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    Eigen::MatrixXf last_input;
    Eigen::MatrixXf weight_gradients;
    Eigen::VectorXf bias_gradients;
    bool use_avx = true;

    Linear(int in_f, int out_f) : Layer("Linear"), in_features(in_f), out_features(out_f)
    {
        weights = Eigen::MatrixXf::Random(out_f, in_f);
        bias = Eigen::VectorXf::Zero(out_f);
        weight_gradients = Eigen::MatrixXf::Zero(out_f, in_f);
        bias_gradients = Eigen::VectorXf::Zero(out_f);
    }

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

                // Sum up 8 floats in the SIMD register
                alignas(ALIGNMENT) float temp[8];
                _mm256_store_ps(temp, sum);
                float dot_product = 0.0f;
                for (int k = 0; k < 8; ++k)
                {
                    dot_product += temp[k];
                }

                // Handle remaining elements (if in_features % 8 != 0)
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

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        last_input = input;
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

        weight_gradients = grad_output * last_input.transpose();
        bias_gradients = grad_output.rowwise().sum();

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

    int get_in_features() const { return in_features; }
    int get_out_features() const { return out_features; }
    Eigen::MatrixXf get_weights() const { return weights; }
    Eigen::VectorXf get_bias() const { return bias; }
    bool get_use_avx() const { return use_avx; }

    void set_weights(const Eigen::MatrixXf &new_weights)
    {
        weights = new_weights;
    }

    void set_bias(const Eigen::VectorXf &new_bias)
    {
        bias = new_bias;
    }
};
