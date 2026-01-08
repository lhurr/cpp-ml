# Simple C++ Neural Network Library

A lightweight, header-only C++ library for building and training neural networks from scratch. This library was built to understand the core mathematical concepts behind deep learning, including matrix operations, forward propagation, and backpropagation.

## Features

- **Matrix Library**: A custom template-based Matrix class with support for:
  - Basic arithmetic (+, -, \*, scalar multiplication)
  - Matrix multiplication (optimized with `ikj` loop order for cache locality)
- **Neural Network**:
  - Fully connected (Dense) layers
  - Sigmoid and ReLU activation functions
  - Backpropagation training algorithm

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

## Running the Example

The project includes an example that trains a neural network to solve the XOR problem.

```bash
./xor_demo
```

## Usage

Include the necessary headers in your project:

```cpp
#include "NeuralNetwork.h"

// Define network topology: 2 inputs -> 4 hidden -> 1 output
std::vector<size_t> topology = {2, 4, 1};
NeuralNetwork nn(topology, 0.5); // Learning rate 0.5

// Train
std::vector<double> input = {0.0, 1.0};
std::vector<double> target = {1.0};
nn.train(input, target);

// Predict
Matrix<double> output = nn.feedForward(input);
```
