#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

#include "aurelius/layer.hh"
#include "aurelius/optimizers/optimizer.hh"
#include "aurelius/initialization.hh"

using namespace aurelius::optimizers;

namespace aurelius
{
    namespace layers
    {

        constexpr int ALIGNMENT = 32;

        class LinearLayer : public Layer
        {

        public:
            LinearLayer() = default;
            virtual ~LinearLayer() = default;
            virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &input) = 0;
            virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) = 0;
            virtual void apply_gradients(float learning_rate) = 0;
            virtual void set_optimizer(std::unique_ptr<Optimizer> opt) = 0;
            virtual void set_use_avx(bool flag) = 0;
            virtual bool get_use_avx() const = 0;
            virtual int get_in_features() const = 0;
            virtual int get_out_features() const = 0;
            virtual Eigen::MatrixXf get_weights() const = 0;
            virtual Eigen::VectorXf get_bias() const = 0;
            virtual void set_weights(const Eigen::MatrixXf &new_weights) = 0;
            virtual void set_bias(const Eigen::VectorXf &new_bias) = 0;

        protected:
            int in_features, out_features;
            Eigen::MatrixXf weights;
            Eigen::VectorXf bias;
            Eigen::MatrixXf weight_gradients;
            Eigen::VectorXf bias_gradients;
            std::unique_ptr<Optimizer> layer_optimizer;
            Eigen::MatrixXf forward_simd(const Eigen::MatrixXf &input);
            Eigen::MatrixXf forward_eigen(const Eigen::MatrixXf &input);
        };

        class Linear : public LinearLayer
        {
        public:
            Linear(int in_f, int out_f, InitType init_type = InitType::Xavier)
            {
                in_features = in_f;
                out_features = out_f;
                if (in_f <= 0 || out_f <= 0)
                {
                    throw std::invalid_argument("Input and output features must be positive.");
                }
                initialize_weights(init_type);
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

            void initialize_weights(InitType type)
            {
                std::random_device rd;
                std::mt19937 gen(rd());

                switch (type)
                {
                case InitType::Uniform:
                    weights = Eigen::MatrixXf::Random(out_features, in_features);
                    break;

                case InitType::Xavier:
                {
                    float limit = std::sqrt(2.0f / (in_features + out_features));
                    std::uniform_real_distribution<float> dist(-limit, limit);
                    weights = Eigen::MatrixXf::NullaryExpr(out_features, in_features, [&]()
                                                           { return dist(gen); });
                    break;
                }

                case InitType::He:
                {
                    float stddev = std::sqrt(2.0f / in_features);
                    std::normal_distribution<float> dist(0.0f, stddev);
                    weights = Eigen::MatrixXf::NullaryExpr(out_features, in_features, [&]()
                                                           { return dist(gen); });
                    break;
                }

                case InitType::Zero:
                    weights = Eigen::MatrixXf::Zero(out_features, in_features);
                    break;

                case InitType::Custom:
                    // Expect user to call `set_weights(...)`
                    break;

                default:
                    throw std::invalid_argument("Unknown InitType.");
                }
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