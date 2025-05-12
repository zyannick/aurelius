#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>  

template <typename child, typename input, typename output>
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Performs the forward pass of the neural network layer.
     * 
     * This function is intended to be overridden by derived classes to implement
     * the specific forward pass logic for different types of layers. It takes an
     * input matrix and returns the result of the forward pass.
     * 
     * @param input The input matrix to the layer.
     * @return The output matrix after applying the forward pass.
     */
    inline virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input){
        return static_cast<child*>(this)->forward(input);
    }

    /**
     * @brief Performs the backward pass of the neural network layer.
     *
     * This function is intended to be overridden by derived classes to implement
     * the specific backward pass logic for the layer. It computes the gradient of
     * the loss with respect to the input of the layer.
     *
     * @param input The gradient of the loss with respect to the output of the layer.
     * @return The gradient of the loss with respect to the input of the layer.
     */
    inline virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& input){
        return static_cast<child*>(this)->backward(input);
    }

    /**
     * @brief Updates the layer with the given learning rate.
     *
     * This function is intended to be overridden by derived classes.
     * It calls the `update` method of the derived class, passing the
     * learning rate as an argument.
     *
     * @param learning_rate The rate at which the layer should learn.
     */
    inline virtual void update(float learning_rate){
        static_cast<child*>(this)->update(learning_rate);
    }
};