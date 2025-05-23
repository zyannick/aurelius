#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <immintrin.h>
#include <x86intrin.h>

namespace aurelius
{
    namespace attention
    {

        class RoPE
        {

        public:
        private:
            int max_seq_len;
            int d_model;
            int batch_size = 1;
            Eigen::MatrixXf positional_encoding;

            void rotate_half(__m256 &x1, __m256 &x2)
            {
                __m256 tmp = x1;
                x1 = _mm256_sub_ps(_mm256_setzero_ps(), x2); // -x2
                x2 = tmp;                                    // x1
            }

            void apply_rope(float *x, const float *freq)
            {
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int pos = 0; pos < max_seq_len; ++pos)
                    {
                        for (int d = 0; d < d_model; d += 8)
                        { // Process 8 floats at once (256 bits)
                            float theta = freq[d / 2] * pos;
                            float cos_theta = std::cos(theta);
                            float sin_theta = std::sin(theta);

                            __m256 data = _mm256_loadu_ps(&x[(b * max_seq_len * d_model) + (pos * d_model) + d]);
                            __m256 cos_vec = _mm256_set1_ps(cos_theta);
                            __m256 sin_vec = _mm256_set1_ps(sin_theta);

                            __m256 x1 = _mm256_permute_ps(data, 0b10100000); // Extract even indices
                            __m256 x2 = _mm256_permute_ps(data, 0b11110101); // Extract odd indices

                            rotate_half(x1, x2); // Apply rotation

                            __m256 result = _mm256_add_ps(_mm256_mul_ps(data, cos_vec), _mm256_mul_ps(x1, sin_vec));
                            _mm256_storeu_ps(&x[(b * max_seq_len * d_model) + (pos * d_model) + d], result);
                        }
                    }
                }
            }
        };
    }
}