
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include "aurelius/layer.hh"

namespace aurelius
{
    namespace loss_functions
    {

        class Loss
        {
        public:
            virtual ~Loss() = default;
            virtual float forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &labels) = 0;
            virtual Eigen::MatrixXf backward() const = 0;
        };

    }
}