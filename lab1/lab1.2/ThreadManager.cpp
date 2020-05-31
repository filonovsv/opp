#include "ThreadManager.h"

void thread_manager_init(int size, int mpi_size, int* &revcounts, int* &displs) {

  revcounts = new int[size];
  displs = new int[size];

  for (size_t i = 0; i < mpi_size; ++i) {
    revcounts[i] = size / mpi_size;
  }

  for (size_t i = 0; i < size % mpi_size; ++i) {
    revcounts[i]++;
  }

  displs[0] = 0;

  for (size_t i = 1; i < mpi_size; ++i) {
    displs[i] = displs[i - 1] + revcounts[i - 1];
  }
}

void thread_manager_finalize(int* revcounts, int* displs) {

  delete[] revcounts;
  delete[] displs;
}
