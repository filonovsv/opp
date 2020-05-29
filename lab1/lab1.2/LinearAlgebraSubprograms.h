#pragma once

void mult_double_by_matrix(double x, double* m, size_t size, double*  result, int mpi_rank, int* revcounts);

void mult_matrix_by_vector(double* m, double* v, size_t size, double*  result, int mpi_rank, int* revcounts, int* displs);

void mult_vector_by_vector(double* v1, double* v2, double &result, int mpi_rank, int* revcounts);

void mult_double_by_vector(double x, double* v, double*  result, int mpi_rank, int* revcounts);

void add_vector_to_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts);

void sub_vector_from_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts);

double norma(double* v, int mpi_rank, int* revcounts);

void solve(double* A, double* b, size_t size, double*  result, int mpi_size, int mpi_rank, int* revcounts, int* displs);

void copy_vector_to_vector(double* from, double* to, int mpi_rank, int* revcounts);
