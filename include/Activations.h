#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <algorithm>
#include <cmath>

namespace Activations {

// Sigmoid
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

inline double sigmoidDerivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

// ReLU
inline double relu(double x) { return std::max(0.0, x); }

inline double reluDerivative(double x) { return x > 0.0 ? 1.0 : 0.0; }
} // namespace Activations

#endif
