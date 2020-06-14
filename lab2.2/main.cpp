#include "LinearAlgebraSubprograms.h"
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {

  int size;
  cin >> size;

  double** A = new double*[size];
  for (int i = 0; i < size; i++) {

    A[i] = new double[size];
  }

  double* b = new double[size];

  double* x = new double[size];

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {

      cin >> A[i][j];
    }
  }

  for (int i = 0; i < size; i++) {

    cin >> b[i];
    x[i] = 0;
  }

  omp_set_num_threads(atoi(argv[1]));
  
  double start = omp_get_wtime();
  solve(A, b, 2, x);
  double finish = omp_get_wtime();

  for (int i = 0; i < 2; i++) {

    cout << x[i] << " ";
  }
  cout << endl << finish - start;

  for (int i = 0; i < size; i++) {

    delete[] A[i];
  }
  delete[] A;
  delete[] b;
  delete[] x;

  system("pause");
  return 0;
}
