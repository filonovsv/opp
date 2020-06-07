#include "mpi.h"
#include "ThreadManager.h"

void mult_matrix_by_matrix(double* A, double* B, double* C, int* rank, int size, int n1, int n2, int n3) {

  int dims[2] = { 0, 0 };
  MPI_Dims_create(size, 2, dims);

  int periods[2] = { 0, 0 };
  int reorder = 1;
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm_2d);

  MPI_Comm_rank(comm_2d, rank);

  int coords[2] = { 0, 0 };
  MPI_Cart_get(comm_2d, 2, dims, periods, coords);

  int* sendcounts_A = new int[dims[0]];
  int* displs_A = new int[dims[0]];

  int rest_A = n1 % dims[0];

  for (int i = 0; i < dims[0]; ++i) {
    if (i < rest_A) {
      sendcounts_A[i] = (n1 / dims[0] + 1) * n2;
      displs_A[i] = (n1 / dims[0] + 1) * i * n2;
    }
    else {
      sendcounts_A[i] = (n1 / dims[0]) * n2;
      displs_A[i] = ((n1 / dims[0] + 1) * rest_A + (i - rest_A) * (n1 / size)) * n2;
    }
  }

  MPI_Comm commcol;
  MPI_Comm commrow;
  int belongs[2] = { 0, 1 };
  MPI_Cart_sub(comm_2d, belongs, &commrow);

  belongs[0] = 1;
  belongs[1] = 0;
  MPI_Cart_sub(comm_2d, belongs, &commcol);

  int row_per_proc;
  int col_per_proc;

  thread_manager(row_per_proc, col_per_proc, n1, n3, dims, coords);

  double* A_for_proc = new double[row_per_proc * n2];
  double* B_for_proc = new double[n2 * col_per_proc];
  double* C_for_proc = new double[row_per_proc * col_per_proc];

  if (coords[1] == 0) {
    MPI_Scatterv(A, sendcounts_A, displs_A, MPI_DOUBLE, A_for_proc, row_per_proc * n2, MPI_DOUBLE, 0, commcol);
  }

  MPI_Datatype AUX_B_TYPE;
  MPI_Type_vector(n2, col_per_proc, n3, MPI_DOUBLE, &AUX_B_TYPE);
  MPI_Type_create_resized(AUX_B_TYPE, 0, col_per_proc * sizeof(double), &AUX_B_TYPE);
  MPI_Type_commit(&AUX_B_TYPE);

  int* sendcounts_B = new int[dims[1]];
  int* displs_B = new int[dims[1]];
  for (int i = 0; i < dims[1]; ++i) {
    sendcounts_B[i] = 1;
    displs_B[i] = i;
  }

  if (coords[0] == 0) {
    MPI_Scatterv(B, sendcounts_B, displs_B, AUX_B_TYPE, B_for_proc, col_per_proc * n2, MPI_DOUBLE, 0, commrow);
  }

  MPI_Bcast(A_for_proc, row_per_proc * n2, MPI_DOUBLE, 0, commrow);
  MPI_Bcast(B_for_proc, col_per_proc * n2, MPI_DOUBLE, 0, commcol);

  for (int i = 0; i < row_per_proc; ++i) {
    for (int j = 0; j < col_per_proc; ++j) {
      C_for_proc[i * col_per_proc + j] = 0;
    }
  }

  for (int i = 0; i < row_per_proc; ++i) {
    for (int j = 0; j < n2; ++j) {
      for (int k = 0; k < col_per_proc; ++k) {
        C_for_proc[i * col_per_proc + k] += A_for_proc[i * n2 + j] * B_for_proc[j * col_per_proc + k];
      }
    }
  }

  MPI_Datatype AUX_C_TYPE;
  MPI_Type_vector(row_per_proc, col_per_proc, n3, MPI_DOUBLE, &AUX_C_TYPE);
  MPI_Type_create_resized(AUX_C_TYPE, 0, col_per_proc * sizeof(double), &AUX_C_TYPE);
  MPI_Type_commit(&AUX_C_TYPE);

  int* sendcounts_C = new int[size];
  int* displs_C = new int[size];

  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      displs_C[i * dims[1] + j] = dims[1] * row_per_proc * i + j;
      sendcounts_C[i * dims[1] + j] = 1;
    }
  }

  MPI_Gatherv(C_for_proc, col_per_proc * row_per_proc, MPI_DOUBLE, C, sendcounts_C, displs_C, AUX_C_TYPE, 0, comm_2d);

  delete[] A_for_proc;
  delete[] B_for_proc;
  delete[] C_for_proc;
  delete[] sendcounts_A;
  delete[] displs_A;
  delete[] sendcounts_B;
  delete[] displs_B;
  delete[] sendcounts_C;
  delete[] displs_C;
}
