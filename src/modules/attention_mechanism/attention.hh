#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense> // For dense matrices and vectors
#include <layer.hh>

using namespace aurelius::layers;

namespace aurelius
{
    namespace attention
    {

        template <typename Scalar>
        struct exp_op
        {
            EIGEN_STRONG_INLINE Scalar operator()(const Scalar &x) const
            {
                return std::exp(x);
            }
        };

        // Softmax implementation with unaryExpr
        template <typename Scalar>
        struct mysoftmax
        {
            EIGEN_STRONG_INLINE Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
            operator()(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x) const
            {
                Scalar max_val = x.maxCoeff(); // Numerical stability
                Eigen::Array<Scalar, Eigen::Dynamic, 1> shifted = (x.array() - max_val);

                // Apply exp using unaryExpr
                Eigen::Array<Scalar, Eigen::Dynamic, 1> exp_shifted = shifted.unaryExpr(exp_op<Scalar>());

                Scalar sum_exp = exp_shifted.sum();
                return (exp_shifted / sum_exp).matrix();
            }
        };

        class Attention : public Layer
        {
        public:
            Attention(int d_model, int num_heads) : d_model(d_model), num_heads(num_heads)
            {
                W_q = Eigen::MatrixXf::Random(d_model, d_model);
                W_k = Eigen::MatrixXf::Random(d_model, d_model);
                W_v = Eigen::MatrixXf::Random(d_model, d_model);
            }

            Eigen::MatrixXf scaled_dot_product_attention(const Eigen::MatrixXf &Q, const Eigen::MatrixXf &K, const Eigen::MatrixXf &V)
            {
                Eigen::MatrixXf attention_scores = Q * K.transpose();
                attention_scores /= sqrt(d_model);
                Eigen::MatrixXf attention_weights = attention_scores.unaryExpr(mysoftmax<float>());
                return attention_weights * V;
            }

        private:
            int d_model;
            int num_heads;
            Eigen::MatrixXf W_q;
            Eigen::MatrixXf W_k;
            Eigen::MatrixXf W_v;
        };
    }
}

template <typename Scalar>
