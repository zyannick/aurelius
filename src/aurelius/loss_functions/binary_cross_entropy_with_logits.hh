
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include <stdexcept>
#include <xsimd/xsimd.hpp>
#include "aurelius/activations/sigmoid.hh"
#include "aurelius/loss_functions/loss.hh"
#include "aurelius/loss_functions/binary_cross_entropy.hh"



namespace aurelius
{
    namespace loss_functions
    {

        class BinaryCrossEntropyWithLogits : public BinaryCrossEntropy
        {

        public:
            BinaryCrossEntropyWithLogits() : BinaryCrossEntropy("sum") {} 
            BinaryCrossEntropyWithLogits(const std::string &reduction) : BinaryCrossEntropy(reduction) {}

            float forward(Eigen::MatrixXf predictions, Eigen::MatrixXf labels, float epsilon = 1e-9)
            {

                if (predictions.rows() != labels.rows() || predictions.cols() != labels.cols())
                {
                    throw std::invalid_argument("Predictions and labels must have the same dimensions");
                }

                aurelius::activation::sigmoid(predictions);
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

        };

    }
}