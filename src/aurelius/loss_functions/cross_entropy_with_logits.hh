#pragma once

#include <eigen3/Eigen/Dense>
#include <string>
#include <stdexcept>
#include "aurelius/loss_functions/cross_entropy_loss.hh"
#include "aurelius/activations/softmax.hh"

namespace aurelius
{
    namespace loss_functions
    {

        class CrossEntropyLossWithLogits : public CrossEntropyLoss
        {
        public:
            explicit CrossEntropyLossWithLogits(const std::string &reduction = "sum") : CrossEntropyLoss(reduction) {}

            float forward(Eigen::MatrixXf logits, Eigen::MatrixXf labels, float epsilon = 1e-9) override
            {

                Eigen::MatrixXf probabilities = aurelius::activation::softmax(logits);

                return CrossEntropyLoss::forward(probabilities, labels, epsilon);
            }

            Eigen::MatrixXf backward(float /*epsilon*/ = 1e-9) override
            {
                if (last_predictions.size() == 0 || last_labels.size() == 0)
                {
                    throw std::runtime_error("backward() called before forward(). Call forward() first.");
                }

                Eigen::MatrixXf gradient = last_predictions - last_labels;

                if (reduction == "mean")
                {
                    gradient /= static_cast<float>(last_predictions.rows());
                }

                return gradient;
            }
        };

    }
}
