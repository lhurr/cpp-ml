#include "Matrix.h"
#include "NeuralNetwork.h"
#include <iostream>

int main() {
  std::cout << "Training Neural Network on XOR..." << std::endl;

  // Topology: 2 inputs, 4 hidden neurons, 1 output
  std::vector<size_t> topology = {2, 4, 1};
  NeuralNetwork nn(topology, 0.5); // Learning rate 0.5

  // XOR Training Data
  std::vector<std::vector<double>> trainingInputs = {
      {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

  std::vector<std::vector<double>> trainingTargets = {
      {0.0}, {1.0}, {1.0}, {0.0}};

  // Train
  for (int epoch = 0; epoch < 20000; ++epoch) {
    for (size_t i = 0; i < trainingInputs.size(); ++i) {
      nn.train(trainingInputs[i], trainingTargets[i]);
    }

    if (epoch % 1000 == 0) {
      std::cout << "Epoch " << epoch << " complete." << std::endl;
    }
  }

  // Verify
  std::cout << "\nTesting trained network:" << std::endl;
  for (size_t i = 0; i < trainingInputs.size(); ++i) {
    Matrix<double> output = nn.feedForward(trainingInputs[i]);
    std::cout << "Input: " << trainingInputs[i][0] << ", "
              << trainingInputs[i][1] << " -> Output: " << output.at(0, 0)
              << std::endl;
  }

  return 0;
}
