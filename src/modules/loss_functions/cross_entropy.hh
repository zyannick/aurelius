
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>

namespace aurelius
{
    namespace loss_functions
    {

        class CrossEntropy
        {

            Eigen::MatrixXf last_predictions;
            Eigen::MatrixXf last_ground_truth;

            CrossEntropy()
            {
            }

            Eigen::MatrixXf forward(Eigen::MatrixXf predictions, Eigen::MatrixXf ground_truth, float epsilon = 1e-5)
            {
                
            }

            Eigen::MatrixXf backward()
            {
            }
        };

    }
}