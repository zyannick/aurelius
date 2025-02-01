#include <iostream>

int main() {
    std::cout << "Hello from WSL2!" << std::endl;
    #ifdef __AVX__
        printf("__AVX__ is defined\n");
    #else
        printf("__AVX__ is NOT defined\n");
    #endif
    return 0;
}