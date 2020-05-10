#pragma once
#include <vector>

using namespace std;

class SquareMatrix
{
public:
  SquareMatrix();
  ~SquareMatrix();
  void operator=(vector<vector<double>> rhs);
  vector<double> operator*(vector<double> rhs);
  friend SquareMatrix operator*(double lhs, SquareMatrix rhs);

private:
  bool isSquare();
  size_t getOrder();
  vector<vector<double>> matrix;
};

SquareMatrix operator*(double lhs, SquareMatrix rhs);
