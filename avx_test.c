#include <stdio.h>
#include <immintrin.h>

int main() {
    #ifdef __AVX__
        printf("AVX is enabled at compile time!\n");
        __m256 a = _mm256_set1_ps(1.0f);  // Crée un registre AVX
        __m256 b = _mm256_set1_ps(2.0f);
        __m256 c = _mm256_add_ps(a, b);   // Additionne avec AVX
        float res[8];
        _mm256_storeu_ps(res, c);         // Stocke le résultat
        printf("AVX computation result: %.1f\n", res[0]);  // Vérifie que AVX fonctionne
    #else
        printf("AVX is NOT enabled!\n");
    #endif
    return 0;
}
