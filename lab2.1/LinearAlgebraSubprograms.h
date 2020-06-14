#pragma once
void mult_double_by_matrix(double x, double** m, size_t size, double**  result);

void mult_matrix_by_vector(double** m, double* v, size_t size, double*  result);

void mult_vector_by_vector(double* v1, double* v2, size_t size, double &result);

void mult_double_by_vector(double x, double* v, size_t size, double*  result);

void add_vector_to_vector(double* v1, double* v2, size_t size, double*  result);

void sub_vector_from_vector(double* v1, double* v2, size_t size, double*  result);

double norma(double* v, size_t size);

void solve(double** A, double* b, size_t size, double*  result);

void copy_vector_to_vector(double* from, double* to, size_t size);
