#include <iostream>
#include <eigen3/Eigen/Dense>  // For dense matrices and vectors
#include <layer.hh>

class BatchNormalization : public Layer<BatchNormalization, Eigen::MatrixXf, Eigen::MatrixXf> {
    int d_model;
    Eigen::MatrixXf gamma;
    Eigen::MatrixXf beta;
    Eigen::MatrixXf mean;
    Eigen::MatrixXf variance;
    float epsilon = 1e-5;

    BatchNormalization(int d_model) : d_model(d_model) {
        gamma = Eigen::MatrixXf::Ones(d_model);
        beta = Eigen::MatrixXf::Zero(d_model);
        mean = Eigen::MatrixXf::Zero(d_model);
        variance = Eigen::MatrixXf::Ones(d_model);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        Eigen::MatrixXf normalized = (input.rowwise() - mean.transpose()).array().rowwise() / (variance.transpose().array() + epsilon).sqrt();
        return (normalized.array().rowwise() * gamma.array()).rowwise() + beta.transpose().array();
    }

};