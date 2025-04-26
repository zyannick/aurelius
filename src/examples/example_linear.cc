#include <iostream>
#include <eigen3/Eigen/Dense>
#include "modules/linear.hh"

int main()
{
    srand(42);

    int input_size = 10;   
    int output_size = 5;   
    int batch_size = 3;    

    Linear layer(input_size, output_size);

    Eigen::MatrixXf input = Eigen::MatrixXf::Random(input_size, batch_size);
    std::cout << "Input:\n" << input << "\n\n";

    Eigen::MatrixXf output = layer.forward(input);
    std::cout << "Output (forward pass):\n" << output << "\n\n";

    Eigen::MatrixXf grad_output = Eigen::MatrixXf::Random(output_size, batch_size);

    Eigen::MatrixXf grad_input = layer.backward(grad_output);
    std::cout << "Gradient wrt input (backward pass):\n" << grad_input << "\n\n";

    float learning_rate = 0.01f;
    layer.update(learning_rate);

    Eigen::MatrixXf new_output = layer.forward(input);
    std::cout << "Output after one update:\n" << new_output << "\n\n";

    return 0;
}
