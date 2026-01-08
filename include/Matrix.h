#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

template <typename T> class Matrix {
private:
  std::vector<std::vector<T>> data;
  size_t rows;
  size_t cols;

public:
  // Constructor
  Matrix(size_t rows, size_t cols, const T &initial = T())
      : rows(rows), cols(cols) {
    data.resize(rows, std::vector<T>(cols, initial));
  }

  // Accessors
  size_t getRows() const { return rows; }
  size_t getCols() const { return cols; }

  // Element access
  T &at(size_t r, size_t c) { return data.at(r).at(c); }
  const T &at(size_t r, size_t c) const { return data.at(r).at(c); }

  // Randomize
  void randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        data[i][j] = static_cast<T>(dis(gen));
      }
    }
  }

  // Matrix Multiplication (Dot Product)
  // Optimized for row-major memory layout (ikj loop order)
  Matrix<T> dot(const Matrix<T> &other) const {
    if (cols != other.rows) {
      throw std::invalid_argument(
          "Matrix dimensions do not match for multiplication.");
    }

    Matrix<T> result(rows, other.cols);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = 0; k < cols; ++k) {
        T r = data[i][k];
        for (size_t j = 0; j < other.cols; ++j) {
          result.data[i][j] += r * other.data[k][j];
        }
      }
    }
    return result;
  }

  // Element-wise addition
  Matrix<T> operator+(const Matrix<T> &other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = data[i][j] + other.data[i][j];
      }
    }
    return result;
  }

  // Element-wise subtraction
  Matrix<T> operator-(const Matrix<T> &other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = data[i][j] - other.data[i][j];
      }
    }
    return result;
  }

  // Element-wise multiplication (Hadamard product)
  Matrix<T> multiply(const Matrix<T> &other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument(
          "Matrix dimensions must match for element-wise multiplication.");
    }
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = data[i][j] * other.data[i][j];
      }
    }
    return result;
  }

  // Scalar multiplication
  Matrix<T> operator*(const T &scalar) const {
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = data[i][j] * scalar;
      }
    }
    return result;
  }

  // Transpose
  Matrix<T> transpose() const {
    Matrix<T> result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(j, i) = data[i][j];
      }
    }
    return result;
  }

  // Map function (apply function to every element)
  Matrix<T> map(std::function<T(T)> func) const {
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = func(data[i][j]);
      }
    }
    return result;
  }

  void print() const {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        std::cout << data[i][j] << " ";
      }
      std::cout << "\n";
    }
  }

  static Matrix<T> fromVector(const std::vector<T> &input) {
    Matrix<T> m(input.size(), 1);
    for (size_t i = 0; i < input.size(); ++i) {
      m.at(i, 0) = input[i];
    }
    return m;
  }

  std::vector<T> toVector() const {
    std::vector<T> res;
    for (const auto &row : data) {
      for (const auto &elem : row) {
        res.push_back(elem);
      }
    }
    return res;
  }
};

#endif
