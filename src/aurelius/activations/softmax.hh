#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include <stdexcept>

namespace aurelius
{
    namespace activation
    {

        inline Eigen::MatrixXf softmax(const Eigen::MatrixXf &logits)
        {

            Eigen::VectorXf max_coeffs = logits.rowwise().maxCoeff();
            Eigen::MatrixXf stable_logits = logits.colwise() - max_coeffs;
            Eigen::MatrixXf exp_logits = stable_logits.array().exp();
            Eigen::VectorXf sum_exp = exp_logits.rowwise().sum();
            Eigen::MatrixXf probs = exp_logits.array().colwise() / sum_exp.array();

            return probs;
        }

    }
}