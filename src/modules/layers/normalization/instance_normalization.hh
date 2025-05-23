#include <iostream>
#include <eigen3/Eigen/Dense>


class InstanceNormalization
{
public:
    InstanceNormalization(int d_model) : d_model(d_model)
    {
        gamma = Eigen::MatrixXf::Ones(d_model);
        beta = Eigen::MatrixXf::Zero(d_model);
        mean = Eigen::MatrixXf::Zero(d_model);
        variance = Eigen::MatrixXf::Ones(d_model);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        Eigen::MatrixXf normalized = (input.rowwise() - mean.transpose()).array().rowwise() / (variance.transpose().array() + epsilon).sqrt();
        return (normalized.array().rowwise() * gamma.array()).rowwise() + beta.transpose().array();
    }
    Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output)
    {
        Eigen::MatrixXf grad_input = (grad_output.array() * gamma.transpose().array()).rowwise() / (variance.transpose().array() + epsilon).sqrt();
        return grad_input;
    }
    void update_parameters(float learning_rate)
    {
        gamma = gamma - learning_rate * (gamma - Eigen::MatrixXf::Ones(d_model));
        beta = beta - learning_rate * beta;
    }
    void set_epsilon(float eps)
    {
        epsilon = eps;
    }
    void set_gamma(const Eigen::MatrixXf &new_gamma)
    {
        if (new_gamma.size() != d_model)
        {
            throw std::invalid_argument("Gamma dimensions mismatch in set_gamma");
        }
        gamma = new_gamma;
    }
    void set_beta(const Eigen::MatrixXf &new_beta)
    {
        if (new_beta.size() != d_model)
        {
            throw std::invalid_argument("Beta dimensions mismatch in set_beta");
        }
        beta = new_beta;
    }
    void set_mean(const Eigen::MatrixXf &new_mean)
    {
        if (new_mean.size() != d_model)
        {
            throw std::invalid_argument("Mean dimensions mismatch in set_mean");
        }
        mean = new_mean;
    }
    void set_variance(const Eigen::MatrixXf &new_variance)
    {
        if (new_variance.size() != d_model)
        {
            throw std::invalid_argument("Variance dimensions mismatch in set_variance");
        }
        variance = new_variance;
    }
    Eigen::MatrixXf get_gamma() const
    {
        return gamma;
    }
    Eigen::MatrixXf get_beta() const
    {
        return beta;
    }
    Eigen::MatrixXf get_mean() const
    {
        return mean;
    }
    Eigen::MatrixXf get_variance() const
    {
        return variance;
    }
    int get_d_model() const
    {
        return d_model;
    }
    void set_epsilon(float new_epsilon)
    {
        if (new_epsilon <= 0)
        {
            throw std::invalid_argument("Epsilon must be positive");
        }
        epsilon = new_epsilon;
    }
    float get_epsilon() const
    {
        return epsilon;
    }
private:
    int d_model;
    Eigen::MatrixXf gamma;
    Eigen::MatrixXf beta;
    Eigen::MatrixXf mean;
    Eigen::MatrixXf variance;
    float epsilon = 1e-5;
};