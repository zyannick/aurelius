#include <stdio.h>

#ifdef __AVX__
#pragma message "AVX is detected by the compiler!"
#endif

int main() {
    #ifdef __AVX__
        printf("AVX is enabled!\n");
    #else
        printf("AVX is NOT enabled!\n");
    #endif

    return 0;
}
