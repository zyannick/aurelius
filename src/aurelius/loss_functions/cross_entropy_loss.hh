
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include "aurelius/loss_functions/loss.hh"
#include "aurelius/activations/softmax.hh"

namespace aurelius
{
    namespace loss_functions
    {

        class CrossEntropyLoss : Loss
        {
        public:
            explicit CrossEntropyLoss(const std::string &reduction = "sum") : reduction(reduction)
            {
                if (reduction != "sum" && reduction != "mean")
                {
                    throw std::invalid_argument("Unsupported reduction type: " + reduction + ". Use 'sum' or 'mean'.");
                }
            }

            ~CrossEntropyLoss() = default;

            virtual float forward(Eigen::MatrixXf predictions, Eigen::MatrixXf labels, float epsilon = 1e-9)
            {
                assert((predictions.array() > 0.0f).all() && (predictions.array() < 1.0f).all());

                if (predictions.rows() != labels.rows() || predictions.cols() != labels.cols())
                {
                    throw std::invalid_argument("Predictions and labels must have the same dimensions");
                }

                last_labels = labels;
                last_predictions = predictions;

                Eigen::MatrixXf clamped_preds = predictions.cwiseMax(epsilon);

                float total_loss = -(labels.array() * clamped_preds.array().log()).sum();

                if (reduction == "sum")
                {
                    return total_loss;
                }
                else // "mean"
                {
                    return total_loss / static_cast<float>(predictions.rows());
                }
            }

            virtual Eigen::MatrixXf backward(float epsilon = 1e-9)
            {
                assert((last_predictions.array() > 0.0f).all() && (last_predictions.array() < 1.0f).all());
                if (last_predictions.rows() != last_labels.rows() || last_predictions.cols() != last_labels.cols())
                {
                    throw std::invalid_argument("Predictions and labels must have the same dimensions");
                }
                Eigen::MatrixXf clamped_preds = last_predictions.cwiseMax(epsilon);
                Eigen::MatrixXf gradient = -last_labels.array() / clamped_preds.array();

                if (reduction == "mean")
                {
                    gradient /= static_cast<float>(last_predictions.rows());
                }

                return gradient;
            }

        protected:
            std::string reduction = "sum";
            Eigen::MatrixXf last_predictions;
            Eigen::MatrixXf last_labels;
        };

    }
}