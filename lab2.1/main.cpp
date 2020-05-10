#include <iostream>
#include <omp.h>
#include "SquareMatrix.h"
#include "Solver.h"

using namespace std;

int main(int argc, char* argv[]) {

  SquareMatrix A;
  A = { {1.0, 2.0}, 
        {3.0 ,1.0} }; 
  vector<double> b;
  b = { 1.0, 1.0 };
  
  omp_set_num_threads(atoi(argv[1]));
  vector<double> x;
  double start = omp_get_wtime();
  try {
    Solver solver;
    x = solver.solve(A, b);
  }
  catch (exception e) {
    cout << e.what();
    return -1;
  }
  double finish = omp_get_wtime();
  
  for (auto i : x) {
    cout << i << " ";
  }

  cout << endl;

  cout << finish - start;

  getchar();
  return 0;
}
