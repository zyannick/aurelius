#pragma once

#include <eigen3/Eigen/Dense>
#include <string>
#include <stdexcept>
#include "aurelius/loss_functions/loss.hh"

namespace aurelius
{
    namespace loss_functions
    {

        class MAE_Loss : public Loss
        {
        public:
            explicit MAE_Loss(const std::string &reduction = "mean") : reduction(reduction)
            {
                if (reduction != "sum" && reduction != "mean")
                {
                    throw std::invalid_argument("Unsupported reduction type: " + reduction + ". Use 'sum' or 'mean'.");
                }
            }

            ~MAE_Loss() = default;

            float forward(Eigen::MatrixXf predictions, Eigen::MatrixXf labels)
            {
                if (predictions.rows() != labels.rows() || predictions.cols() != labels.cols())
                {
                    throw std::invalid_argument("Predictions and labels must have the same dimensions.");
                }

                last_predictions = predictions;
                last_labels = labels;

                Eigen::MatrixXf error = (predictions - labels).array().abs();

                if (reduction == "sum")
                {
                    return error.sum();
                }
                else // "mean"
                {
                    return error.mean();
                }
            }

            Eigen::MatrixXf backward()
            {
                if (last_predictions.size() == 0 || last_labels.size() == 0)
                {
                    throw std::runtime_error("backward() called before forward(). Call forward() first.");
                }

                Eigen::MatrixXf gradient = (last_predictions - last_labels).array().sign();

                if (reduction == "mean")
                {
                    gradient /= static_cast<float>(last_predictions.size());
                }

                return gradient;
            }

        protected:
            std::string reduction;
            Eigen::MatrixXf last_predictions;
            Eigen::MatrixXf last_labels;
        };

    }
}
