#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Activations.h"
#include "Matrix.h"
#include <iostream>
#include <vector>

class NeuralNetwork {
private:
  std::vector<size_t> topology;
  std::vector<Matrix<double>> weights;      // weights[i]: (Lik+1 x Lik)
  std::vector<Matrix<double>> biases;       // biases[i]: (Lik+1 x 1)
  std::vector<Matrix<double>> layerOutputs; // A: (Lk x 1) for each layer k
  std::vector<Matrix<double>> layerInputs;  // Z: (Lk+1 x 1) for each layer k

  double learningRate;

public:
  // Constructor
  NeuralNetwork(const std::vector<size_t> &topology, double lr = 0.1)
      : topology(topology), learningRate(lr) {

    for (size_t i = 0; i < topology.size() - 1; ++i) {
      // Weights from layer i to i+1
      // Matrix dimensions: (next_layer_size x current_layer_size) -> (Lik+1 x
      // Lik)
      Matrix<double> w(topology[i + 1], topology[i]);
      w.randomize();
      weights.push_back(w);

      // Biases for layer i+1
      // Matrix dimensions: (next_layer_size x 1) -> (Lik+1 x 1)
      Matrix<double> b(topology[i + 1], 1);
      b.randomize();
      biases.push_back(b);
    }
  }

  Matrix<double> feedForward(const std::vector<double> &inputArray) {
    Matrix<double> input = Matrix<double>::fromVector(inputArray);

    layerOutputs.clear();
    layerInputs.clear();

    // Input layer has no weights/biases leading to it, just store it
    layerOutputs.push_back(input);

    Matrix<double> current = input;

    for (size_t i = 0; i < weights.size(); ++i) {
      // z = W * a + b
      // (Lik+1 x Lik) * (Lik x 1) + (Lik+1 x 1) -> (Lik+1 x 1)
      Matrix<double> z = weights[i].dot(current) + biases[i];
      layerInputs.push_back(z);

      // a = sigma(z)
      // Using sigmoid for all hidden/output layers for now
      current = z.map(Activations::sigmoid);
      layerOutputs.push_back(current);
    }

    return current;
  }

  void train(const std::vector<double> &inputArray,
             const std::vector<double> &targetArray) {
    // Forward pass
    feedForward(inputArray);

    Matrix<double> targets = Matrix<double>::fromVector(targetArray);

    // Backpropagation

    // Calculate output error: error = targets - output
    Matrix<double> finalOutput = layerOutputs.back();
    Matrix<double> outputErrors = targets - finalOutput;

    // Output gradients = error * sigmoid_derivative(output)
    // Note: layerInputs.back() contains z values for the last layer
    // But sigmoid derivative can be calculated from output `a`: a * (1-a)
    // Let's use the derivative of sigmoid function on `z` or use `a` shortcut.
    // Activations.h has sigmoidDerivative(x).
    // Let's stick to strict chain rule: delta = error * f'(z)

    // We iterate backwards from the last weight matrix
    Matrix<double> errors = outputErrors;

    for (int i = weights.size() - 1; i >= 0; --i) {
      // i corresponds to the weights between layer i and i+1
      // layerInputs[i] corresponds to z values of layer i+1
      // layerOutputs[i] corresponds to activations of layer i (input to these
      // weights) layerOutputs[i+1] corresponds to activations of layer i+1
      // (output of these weights)

      // Calculate gradients for layer i+1
      // gradients: (Lik+1 x 1)
      Matrix<double> gradients =
          layerInputs[i].map(Activations::sigmoidDerivative);

      // gradients = gradients .* errors
      // (Lik+1 x 1) .* (Lik+1 x 1) -> (Lik+1 x 1)
      gradients = gradients.multiply(errors); // element-wise
      gradients = gradients * learningRate;

      // Calculate deltas for weights: delta_w = gradients *
      // transpose(prev_layer_activations) prevLayerT (Ai^T): (1 x Lik)
      Matrix<double> prevLayerT = layerOutputs[i].transpose();

      // weightDeltas: (Lik+1 x 1) * (1 x Lik) -> (Lik+1 x Lik)
      Matrix<double> weightDeltas = gradients.dot(prevLayerT);

      // Adjust weights and biases
      weights[i] = weights[i] + weightDeltas;
      biases[i] = biases[i] + gradients;

      // Calculate errors for the next previous layer (layer i)
      // error_prev = transpose(weights) * errors

      // weightsT: (Lik x Lik+1)
      Matrix<double> weightsT = weights[i].transpose();

      // errors (for next step, layer i): (Lik x Lik+1) * (Lik+1 x 1) -> (Lik x
      // 1)
      errors = weightsT.dot(errors);
    }
  }
};

#endif
