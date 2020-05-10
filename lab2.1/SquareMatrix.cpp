#include "SquareMatrix.h"
#include <stdexcept>

SquareMatrix::SquareMatrix(){}

SquareMatrix::~SquareMatrix(){}

void SquareMatrix::operator=(vector<vector<double>> rhs){

  matrix = rhs;
}

vector<double> SquareMatrix::operator*(vector<double> rhs){

  if (!isSquare()) {
    throw invalid_argument("Error : vector<double> SquareMatrix::operator*(vector<double> rhs) : the matrix is not square");
  }
  vector<double> result;

  try {
    size_t order = getOrder();

    if (order != rhs.size()) {
      throw invalid_argument("Error : vector<double> SquareMatrix::operator*(vector<double> rhs) : the order of the matrix is not equal to the size of the vector");
    }

    result.resize(order);

#pragma omp parallel for 
    for (size_t i = 0; i < order; i++) {
#pragma omp parallel for 
      for (size_t j = 0; j < order; j++) {
        result[i] += matrix[i][j] * rhs[j];
      }
    }
  }
  catch (exception e) {
    throw e;
  }
  
  return result;
}

bool SquareMatrix::isSquare(){

  for (auto i : matrix) {
    if (i.size() != matrix.size()) {
      return false;
    }
  }
  return true;
}

size_t SquareMatrix::getOrder(){

  if (!isSquare()) {
    throw invalid_argument("Error : int SquareMatrix::getOrder() : the matrix is not square");
  }
  return matrix.size();
}

SquareMatrix operator*(double lhs, SquareMatrix rhs) {

  SquareMatrix result;
  try {
    size_t order = rhs.getOrder();

    result.matrix.resize(order);

    for (auto &i : result.matrix) {
      i.resize(order);
    }

#pragma omp parallel for 
    for (size_t i = 0; i < rhs.getOrder(); i++) {
#pragma omp parallel for 
      for (size_t j = 0; j < rhs.getOrder(); j++) {
        result.matrix[i][j] = lhs * rhs.matrix[i][j];
      }
    }
  }
  catch (exception e) {
    throw e;
  }
  return result;
}
