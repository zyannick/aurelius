
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include "aurelius/loss_functions/loss.hh"

namespace aurelius
{
    namespace loss_functions
    {

        class CrossEntropy : Loss
        {

            CrossEntropy()
            {
            }

            Eigen::MatrixXf forward(Eigen::MatrixXf predictions, Eigen::MatrixXf ground_truth, float epsilon = 1e-5)
            {
                
            }

            Eigen::MatrixXf backward()
            {
            }
            private:
            std::string reduction = "sum";
            Eigen::MatrixXf last_predictions;
            Eigen::MatrixXf last_labels;
        };

    }
}