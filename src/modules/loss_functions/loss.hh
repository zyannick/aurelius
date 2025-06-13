#pragma once
#include <eigen3/Eigen/Dense>
#include <string>


namespace aurelius
{
    namespace loss_functions
    {
        class LossFunction
        {
        public:
            virtual ~LossFunction() = default;

            virtual double compute_loss(Eigen::MatrixXf predictions, Eigen::MatrixXf  true_values, size_t size) const = 0;

            virtual void compute_gradient(Eigen::MatrixXf predictions, Eigen::MatrixXf true_values, Eigen::MatrixXf gradient, size_t size) const = 0;

            virtual std::string name() const = 0;
        };
    }
}