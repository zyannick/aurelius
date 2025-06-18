#pragma once
#include <immintrin.h>
#include "aurelius/layer.hh"

namespace aurelius
{
    namespace activations
    {

        inline __m256 relu_avx(__m256 x)
        {
            __m256 zero = _mm256_setzero_ps();
            return _mm256_max_ps(x, zero);
        }
    }
}
