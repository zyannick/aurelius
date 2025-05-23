#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <immintrin.h>
#include <x86intrin.h>
#include <sleef.h>

namespace aurelius
{
    namespace attention
    {

        class PositionalEncoding
        {

        public:
            PositionalEncoding(int max_seq_len, int d_model) : max_seq_len(max_seq_len), d_model(d_model)
            {
                positional_encoding = Eigen::MatrixXf(max_seq_len, d_model);
                calculate_positional_encoding();
            }

            const Eigen::MatrixXf &get_positional_encoding() const
            {
                return positional_encoding;
            }

        private:
            int max_seq_len;
            int d_model;
            Eigen::MatrixXf positional_encoding;

            /**
             * @brief Calculates the positional encoding for a sequence.
             *
             * This function populates the positional encoding matrix with sine and cosine
             * values based on the position and dimension. The positional encoding is used
             * to inject information about the relative or absolute position of the tokens
             * in the sequence.
             *
             * The encoding is calculated as follows:
             * - For even indices (i % 2 == 0), the encoding is calculated using the sine function.
             * - For odd indices (i % 2 != 0), the encoding is calculated using the cosine function.
             *
             * The formula used for the encoding is:
             * - pos_enc(pos, i) = sin(pos / pow(10000, i / d_model)) for even i
             * - pos_enc(pos, i) = cos(pos / pow(10000, (i - 1) / d_model)) for odd i
             *
             * @param max_seq_len The maximum length of the sequence.
             * @param d_model The dimensionality of the model.
             * @param positional_encoding A reference to the matrix where the positional encoding will be stored.
             */
            void calculate_positional_encoding()
            {
                for (int pos = 0; pos < max_seq_len; pos++)
                {
                    for (int i = 0; i < d_model; i++)
                    {
                        if (i % 2 == 0)
                        {
                            positional_encoding(pos, i) = sin(pos / pow(10000, i / d_model));
                        }
                        else
                        {
                            positional_encoding(pos, i) = cos(pos / pow(10000, (i - 1) / d_model));
                        }
                    }
                }
            }

            /**
             * @brief Calculates the positional encoding using SIMD instructions.
             *
             * This function computes the positional encoding for a sequence of length `max_seq_len`
             * and a model dimension of `d_model`. It uses AVX (Advanced Vector Extensions) SIMD
             * instructions to perform parallel computations for efficiency.
             *
             * The positional encoding is calculated using sine and cosine functions of different
             * frequencies. For each position `pos` in the sequence, and for each dimension `i` in
             * the model, the angle is computed as `pos / (10000^(2i/d_model))`. The sine of the angle
             * is used for even indices, and the cosine of the angle is used for odd indices.
             *
             * The computed positional encodings are stored in the `positional_encoding` matrix.
             *
             * @note This function assumes that the `positional_encoding` matrix is pre-allocated
             *       and has dimensions `[max_seq_len][d_model]`.
             *
             * @note This function uses AVX intrinsics, so it requires a CPU that supports AVX.
             *
             * @warning The function does not perform bounds checking on the `positional_encoding`
             *          matrix, so ensure that `max_seq_len` and `d_model` are within valid ranges.
             */
            void simd_calculate_positional_encoding()
            {
                for (int pos = 0; pos < max_seq_len; ++pos)
                {
                    for (int i = 0; i < d_model; i += 8)
                    {
                        __m256 pos_vec = _mm256_set1_ps(static_cast<float>(pos));
                        __m256 i_vec = _mm256_set_ps(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);
                        __m256 angle_rates = _mm256_div_ps(i_vec, _mm256_set1_ps(static_cast<float>(d_model)));
                        __m256 angles = _mm256_mul_ps(pos_vec, angle_rates);

                        __m256 sin_angles = _mm256_sin_ps(angles);
                        __m256 cos_angles = _mm256_cos_ps(angles);

                        for (int j = 0; j < 8; ++j)
                        {
                            positional_encoding(pos, i + j) = (j % 2 == 0) ? sin_angles[j] : cos_angles[j];
                        }
                    }
                }
            }

            void sleef_calculate_positional_encoding()
            {
                for (int pos = 0; pos < max_seq_len; ++pos)
                {
                    for (int i = 0; i < d_model; i += 8)
                    {
                        __m256 pos_vec = _mm256_set1_ps(static_cast<float>(pos));
                        __m256 i_vec = _mm256_set_ps(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);
                        __m256 angle_rates = _mm256_div_ps(i_vec, _mm256_set1_ps(static_cast<float>(d_model)));
                        __m256 angles = _mm256_mul_ps(pos_vec, angle_rates);

                        __m256 sin_angles = Sleef_sinf8_u35(angles);
                        __m256 cos_angles = Sleef_cosf8_u35(angles);

                        for (int j = 0; j < 8; ++j)
                        {
                            positional_encoding(pos, i + j) = (j % 2 == 0) ? sin_angles[j] : cos_angles[j];
                        }
                    }
                }
            }

            /**
             * @brief Computes the sine of each element in a 256-bit vector using a polynomial approximation.
             *
             * This function uses a polynomial approximation to compute the sine of each element in the input
             * vector. The approximation is based on the Taylor series expansion of the sine function.
             *
             * @param x A 256-bit vector (__m256) containing the input values for which to compute the sine.
             * @return A 256-bit vector (__m256) containing the sine of each input value.
             */
            __m256 _mm256_sin_ps(__m256 x)
            {
                // Polynomial approximation of sine function
                __m256 x2 = _mm256_mul_ps(x, x);
                __m256 result = x;
                result = _mm256_sub_ps(result, _mm256_mul_ps(_mm256_set1_ps(1.6666667163e-01f), _mm256_mul_ps(x2, x)));
                result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_set1_ps(8.3333337680e-03f), _mm256_mul_ps(x2, _mm256_mul_ps(x2, x))));
                result = _mm256_sub_ps(result, _mm256_mul_ps(_mm256_set1_ps(1.9841270114e-04f), _mm256_mul_ps(x2, _mm256_mul_ps(x2, _mm256_mul_ps(x2, x)))));
                return result;
            }

            /**
             * @brief Computes the cosine of each element in the input vector using a polynomial approximation.
             *
             * This function uses a polynomial approximation to compute the cosine of each element in the
             * input __m256 vector. The approximation is based on a truncated Taylor series expansion.
             *
             * @param x Input vector of type __m256 containing the values for which the cosine is to be computed.
             * @return A __m256 vector containing the cosine of each element in the input vector.
             */
            __m256 _mm256_cos_ps(__m256 x)
            {
                // Polynomial approximation of cosine function
                __m256 x2 = _mm256_mul_ps(x, x);
                __m256 result = _mm256_set1_ps(1.0f);
                result = _mm256_sub_ps(result, _mm256_mul_ps(_mm256_set1_ps(5.0000001201e-01f), x2));
                result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_set1_ps(4.1666667908e-02f), _mm256_mul_ps(x2, x2)));
                result = _mm256_sub_ps(result, _mm256_mul_ps(_mm256_set1_ps(1.3888889225e-03f), _mm256_mul_ps(x2, _mm256_mul_ps(x2, x2))));
                return result;
            }
        };

    }
}
