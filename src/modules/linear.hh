#include <immintrin.h>  // For AVX
#include <eigen3/Eigen/Dense>  // For dense matrices and vectors
#include <iostream>
#include <vector>
#include <cmath>

constexpr int ALIGNMENT = 32;

struct Linear {
    int in_features, out_features;
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;

    /**
     * @class Linear
     * @brief A class representing a linear layer in a neural network.
     *
     * This class initializes the weights and biases for a linear layer with
     * the specified number of input and output features.
     *
     * @param in_f Number of input features.
     * @param out_f Number of output features.
     *
     * The weights are initialized with random values between 0 and 1, and
     * the biases are initialized to 0.
     */
    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        weights = Eigen::MatrixXf::Random(out_f, in_f);
        bias = Eigen::VectorXf::Zero(out_f);
    }


    
    /**
     * @brief Performs the forward pass of the linear module.
     *
     * This function computes the output of a linear transformation applied to the input matrix.
     * It uses AVX (Advanced Vector Extensions) instructions for efficient computation.
     *
     * @param input The input matrix of size (in_features, batch_size).
     * @return The output matrix of size (out_features, batch_size).
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        Eigen::MatrixXf output(out_features, input.cols());
        __m256 bias_vec = _mm256_loadu_ps(bias.data());

        for (int i = 0; i < out_features; i++) {
            __m256 sum = _mm256_setzero_ps();
            for (int j = 0; j < in_features; j += 8) {
                __m256 input_vec = _mm256_loadu_ps(input.row(j).data());
                __m256 weight_vec = _mm256_loadu_ps(weights.row(i).data() + j);
                sum = _mm256_fmadd_ps(weight_vec, input_vec, sum);
            }
            sum = _mm256_add_ps(sum, bias_vec);
            _mm256_storeu_ps(output.row(i).data(), sum);
        }

        return output;
    }
};
