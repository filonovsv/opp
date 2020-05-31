#include <iostream>
#include <mpi.h>
#include "LinearAlgebraSubprogrammes.h"

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int n1 = atoi(argv[1]);
  int n2 = atoi(argv[2]);
  int n3 = atoi(argv[3]);

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (n1 < size || n2 < size || n3 < size ) {
    MPI_Finalize();
    return 0;
  }

  double* A = nullptr;
  double* B = nullptr;
  double* C = nullptr;

  if (rank == 0) {

    A = new double[n1 * n1];
    B = new double[n2 * n2];
    C = new double[n3 * n3];

    for (int i = 0; i < n1; ++i) {
      for (int j = 0; j < n2; ++j) {
        A[i * n2 + j] = i;
      }
    }

    for (int i = 0; i < n2; ++i) {
      for (int j = 0; j < n3; ++j) {
        B[i * n3 + j] = i;
      }
    }

  }

  double start = MPI_Wtime();

  mult_matrix_by_matrix(A, B, C, &rank, size, n1, n2, n3);

  double finish = MPI_Wtime();

  if (rank == 0) {

    std::cout << "time: " << finish - start << " s" << std::endl;
    std::cout  << "C: "<< std::endl;

    for (int i = 0; i < n1; ++i) {
      for (int j = 0; j < n3; ++j) {
        std::cout <<C[i * n3 + j]<<" ";
      }
      std::cout << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
  }

  MPI_Finalize();
  return 0;
}
