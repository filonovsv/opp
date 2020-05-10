#include "optionalVectorOperations.h"
#include <stdexcept>

double operator*(vector<double> lhs, vector<double> rhs){

  if (lhs.size() != rhs.size()) {
    throw invalid_argument("Error : operator*(vector<double> lhs, vector<double> rhs) : vectors of different lengths");
  }

  size_t size = lhs.size();

  double result = 0;

  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    result += lhs[i] * rhs[i];
  }
  return result;
}

vector<double> operator*(double lhs, vector<double> rhs){

  vector<double> result;
  result.resize(rhs.size());

  #pragma omp parallel for
  for (int i = 0; i < rhs.size(); i++) {
    result[i] = rhs[i] * lhs;
  }
  return result;
}

vector<double> operator-(vector<double> lhs, vector<double> rhs){

  if (lhs.size() != rhs.size()) {
    throw invalid_argument("Error : vector<double> operator-(vector<double> lhs, vector<double> rhs) : vectors of different lengths");
  }
  vector<double> result;
  result.resize(rhs.size());

  #pragma omp parallel for
  for (int i = 0; i < rhs.size(); i++) {
    result[i] = lhs[i] - rhs[i];
  }
  return result;
}

vector<double> operator+(vector<double> lhs, vector<double> rhs){

  if (lhs.size() != rhs.size()) {
    throw invalid_argument("Error : vector<double> operator+(vector<double> lhs, vector<double> rhs) : vectors of different lengths");
  }
  vector<double> result;
  result.resize(rhs.size());

  #pragma omp parallel for
  for (int i = 0; i < rhs.size(); i++) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}

double norma(vector<double> a){

  double result = 0;

  #pragma omp parallel for
  for (int i = 0; i < a.size(); i++) {
    result += pow(a[i],2);
  }
  result = sqrt(result);
  return result;
}
