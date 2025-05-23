#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <vector>
#include <cmath>

#include "src/modules/layers/layer.hh"
#include "src/modules/optimizers/optimizer.hh"

#define EIGEN_USE_THREADS

constexpr int ALIGNMENT = 32;

class Linear : public Layer
{
public:
    Linear(int in_f, int out_f)
    {
        in_features = in_f;
        out_features = out_f;
        weights = Eigen::Tensor<float, 2>(out_f, in_f);
        weights.setRandom();
        bias = Eigen::Tensor<float, 1>(out_f);
        bias.setZero();
        weight_gradients = Eigen::Tensor<float, 2>(out_f, in_f);
        weight_gradients.setRandom();
        bias_gradients = Eigen::Tensor<float, 1>(out_f);
        bias_gradients.setZero();
    }

    Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2> &input) override
    {
        last_input = input;
        if (use_avx && in_features >= 8)
            return forward_simd(input);
        return forward_vectorized(input);
    }

    Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2> &grad_output) override
    {
        if (grad_output.dimension(0) != out_features || grad_output.dimension(1) != last_input.cols())
        {
            throw std::invalid_argument("Gradient output dimensions mismatch in Linear::backward.");
        }
        int batch_size = grad_output.dimension(1);
        if (batch_size == 0)
            return Eigen::Tensor<float, 1>(in_features, 0);

        float inv_batch_size = 1.0f / static_cast<float>(batch_size);
        Eigen::Tensor<float, 2> grad_input = weights.shuffle(Eigen::array<int, 2>{1, 0}) * grad_output;
        weight_gradients = (grad_output * last_input.shuffle(Eigen::array<int, 2>{1, 0})) * inv_batch_size;
        bias_gradients = grad_output.sum(Eigen::array<int, 1>({1})) * inv_batch_size;
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
    Eigen::Tensor<float, 2> get_weights() const { return weights; }
    Eigen::Tensor<float, 1> get_bias() const { return bias; }

    void set_weights(const Eigen::Tensor<float, 2> &new_weights) override
    {
        if (new_weights.dimension(0) != out_features || new_weights.dimension(1) != in_features)
        {
            throw std::invalid_argument("Weight dimensions mismatch in set_weights");
        }
        weights = new_weights;
        if (layer_optimizer)
            layer_optimizer->reset_state();
    }

    void set_bias(const Eigen::Tensor<float, 1> &new_bias) override
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
    Eigen::Tensor<float, 2> forward_simd(const Eigen::Tensor<float, 2> &input)
    {
        Eigen::Tensor<float, 2> output(out_features, input.dimension(1));
        for (int i = 0; i < out_features; ++i)
        {
            for (int b = 0; b < input.dimension(1); ++b)
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

    Eigen::Tensor<float, 2> forward_vectorized(const Eigen::Tensor<float, 2> &input)
    {
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 1)};
        Eigen::Tensor<float, 2> output = weights.contract(input, product_dims);
        return output + bias.reshape(Eigen::array<int, 2>{out_features, 1}).broadcast(Eigen::array<int, 2>{1, input.dimension(1)});
    }
};
