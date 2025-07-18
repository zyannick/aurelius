
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include <stdexcept>
#include "aurelius/loss_functions/loss.hh"

namespace aurelius
{
    namespace loss_functions
    {

        class BinaryCrossEntropy : public Loss
        {

        public:
            BinaryCrossEntropy() : reduction("sum")
            {
            }

            explicit BinaryCrossEntropy(const std::string &reduction) : reduction(reduction)
            {
                if (reduction != "sum" && reduction != "mean")
                {
                    throw std::invalid_argument("Unsupported reduction type: " + reduction + ". Use 'sum' or 'mean'.");
                }
            }

            virtual ~BinaryCrossEntropy() = default;

            virtual float forward(Eigen::MatrixXf predictions, Eigen::MatrixXf labels, float epsilon = 1e-9)
            {
                assert((predictions.array() > 0.0f).all() && (predictions.array() < 1.0f).all());

                if (predictions.rows() != labels.rows() || predictions.cols() != labels.cols())
                {
                    throw std::invalid_argument("Predictions and labels must have the same dimensions");
                }

                last_predictions = predictions;
                last_labels = labels;

                Eigen::MatrixXf clamped_preds = predictions.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);
                Eigen::MatrixXf loss = -1 * (labels.array() * clamped_preds.array().log() + (1 - labels.array()) * (1 - clamped_preds.array()).log());
                if (reduction == "sum")
                {
                    return loss.sum();
                }
                else if (reduction == "mean")
                {
                    return loss.mean();
                }
                else
                {
                    throw std::invalid_argument("Unsupported reduction type: " + reduction + ". Use 'sum' or 'mean'.");
                }
            }

            virtual Eigen::MatrixXf backward(float epsilon = 1e-9)
            {

                if (last_predictions.size() == 0 || last_labels.size() == 0)
                {
                    throw std::runtime_error("backward() called before forward(). Call forward() first.");
                }
                Eigen::MatrixXf clamped_preds = last_predictions.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);
                Eigen::MatrixXf gradient = (clamped_preds.array() - last_labels.array()) / (clamped_preds.array() * (1.0f - clamped_preds.array()));

                if (reduction == "mean")
                {
                    gradient /= static_cast<float>(last_predictions.size());
                }

                return gradient;
            }

            const std::string &getReduction() const
            {
                return reduction;
            }

            void setReduction(const std::string &new_reduction)
            {
                if (new_reduction != "sum" && new_reduction != "mean")
                {
                    throw std::invalid_argument("Unsupported reduction type: " + new_reduction + ". Use 'sum' or 'mean'.");
                }
                reduction = new_reduction;
            }

        protected:
            std::string reduction = "sum";
            Eigen::MatrixXf last_predictions;
            Eigen::MatrixXf last_labels;
        };

    }
}