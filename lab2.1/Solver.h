#pragma once
#include <vector>
#include "SquareMatrix.h"
#include "optionalVectorOperations.h"
using namespace std;

class Solver
{
public:
  Solver();
  ~Solver();
  vector<double> solve(SquareMatrix A, vector<double> b);
};

