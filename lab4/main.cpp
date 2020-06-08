#include <mpi.h>
#include <iostream>
#include "LinearAlgebraSubprograms.h"

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double delta;

  double start = MPI_Wtime();

  solve(rank, size, delta);

  double finish = MPI_Wtime();
  
  if (rank == 0) {
    std::cout << "delta: " << delta << std::endl;
    std::cout << "time: " << finish - start << std::endl;
  }
  MPI_Finalize();
  return 0;
}
