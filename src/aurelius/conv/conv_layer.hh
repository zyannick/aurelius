
#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

#include "aurelius/optimizers/optimizer.hh"
#include "aurelius/layer.hh"


using namespace aurelius::optimizers;

namespace aurelius
{
    namespace convolution
    {

        constexpr int ALIGNMENT = 32;

        class ConvLayer : public Layer
        {

        protected:
            int in_channels, out_channels, kernel_size, stride, padding;
            Eigen::MatrixXf weights;
            Eigen::VectorXf bias;
            Eigen::MatrixXf last_input;
            Eigen::MatrixXf weight_gradients;
            Eigen::VectorXf bias_gradients;
            std::unique_ptr<Optimizer> layer_optimizer;
            bool use_avx = true;
            Eigen::MatrixXf forward_simd(const Eigen::MatrixXf &input);
            Eigen::MatrixXf forward_eigen(const Eigen::MatrixXf &input);

        public:
            ConvLayer() = default;
            virtual ~ConvLayer() = default;
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
        };

    }

}