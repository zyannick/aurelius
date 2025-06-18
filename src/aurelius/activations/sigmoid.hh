
#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include <stdexcept>
#include "aurelius/layer.hh"
// #include <xsimd/xsimd.hpp>

namespace aurelius
{
    namespace activation
    {
        inline void sigmoid(Eigen::MatrixXf &logits)
        {
            logits = 1.0f / (1.0f + (-logits.array().exp()));
        }

        // inline void sigmoid_simd(float *input, float *output, std::size_t size)
        // {
        //     using batch = xsimd::batch<float>;
        //     std::size_t simd_size = batch::size;
        //     std::size_t i = 0;

        //     for (; i + simd_size <= size; i += simd_size)
        //     {
        //         batch x = batch::load_unaligned(input + i);
        //         batch res = 1.0f / (1.0f + xsimd::exp(-x));
        //         res.store_unaligned(output + i);
        //     }

        //     // tail loop
        //     for (; i < size; ++i)
        //     {
        //         output[i] = 1.0f / (1.0f + std::exp(-input[i]));
        //     }
        // }

    }
}