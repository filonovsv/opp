#include "Solver.h"

Solver::Solver(){}

Solver::~Solver(){}

vector<double> Solver::solve(SquareMatrix A, vector<double> b){
  
  vector<double> x;
  vector<double> r;
  vector<double> z;
  double alpha;
  double beta;

  const double epsilon = 1e-3;

  x.resize(b.size());

  try {
    r = b - A * x;
    z = r;

    do {
      alpha = (r*r) / (A*z*z);
      x = x + alpha * z;
      vector<double> old_r = r;
      r = r - alpha * A *z;
      beta = (r*r) / (old_r*old_r);
      z = r + beta * z;

    } while (norma(r) / norma(b) > epsilon);
  }
  catch (exception e) {
    throw e;
  } 
  return x;
}
