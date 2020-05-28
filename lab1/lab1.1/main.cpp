#include "LinearAlgebraSubprograms.h"
#include "ThreadManager.h"
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int mpi_rank;
  int mpi_size;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int* revcounts = nullptr;
  int* displs = nullptr;
  int size;

  size = atoi(argv[1]);

  if (size < mpi_size) {
    MPI_Finalize();
    return 0;
  }

  thread_manager_init(size, mpi_size, revcounts, displs);

  double* A = new double[size * revcounts[mpi_rank]];

  int current_rank = 0;
  while (current_rank < mpi_size) {
    if (mpi_rank == current_rank) {
      for (size_t i = 0; i < revcounts[mpi_rank]; ++i)
      {
        for (size_t j = 0; j < size; ++j) {
          A[i*size + j] = 1.0;
          if (displs[mpi_rank] + i  == j ) {
            A[i*size +j] = 2.0;
          }
        }
      }
    }
    ++current_rank;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  double* b = new double[size];
  double* x = new double[size];

  for (size_t i = 0; i < size; ++i) {
    b[i] = (double)size + 1.0;
    x[i] = 0.0;
  }
 
  double start = MPI_Wtime();

  solve(A, b, size, x, mpi_size, mpi_rank, revcounts, displs);


  double finish = MPI_Wtime();

  if (mpi_rank == 0) {
    std::cout << "result: ";
    for (size_t i = 0; i < size; ++i)
    {
      std::cout << x[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "for: " << finish - start;
  }

  thread_manager_finalize(revcounts, displs);
  MPI_Finalize();
  delete[] A;
  delete[] b;
  delete[] x;

  return 0;
}
