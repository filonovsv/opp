#pragma once

void mult_double_by_matrix(double x, double* m, size_t size, double*  result, int mpi_rank, int* revcounts);

void mult_matrix_by_vector(double* m, double* v, size_t size, double*  result, int mpi_rank, int* revcounts, int* displs);

void mult_vector_by_vector(double* v1, double* v2, size_t size, double &result, int mpi_rank, int* revcounts, int* displs);

void mult_double_by_vector(double x, double* v, size_t size, double*  result, int mpi_rank, int* revcounts, int* displs);

void add_vector_to_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts, int* displs);

void sub_vector_from_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts, int* displs);

double norma(double* v, int mpi_rank, int* revcounts, int* displs);

void solve(double* A, double* b, size_t size, double*  result, int mpi_size, int mpi_rank, int* revcounts, int* displs);

void copy_vector_to_vector(double* from, double* to, size_t size);
