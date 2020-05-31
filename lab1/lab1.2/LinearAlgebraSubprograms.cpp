#include "LinearAlgebraSubprograms.h"
#include <algorithm>
#include <mpi.h>

void mult_double_by_matrix(double x, double* m, size_t size, double*  result, int mpi_rank, int* revcounts) {

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {
    for (size_t j = 0; j < size; ++j) {
      result[i*size + j] = x * m[i * size + j];
    }
  }
}

void mult_matrix_by_vector(double* m, double* v, size_t size, double*  result, int mpi_rank, int* revcounts, int* displs, double* full_x) {

  MPI_Allgatherv(v, revcounts[mpi_rank], MPI_DOUBLE, full_x, revcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {
    for (size_t j = 0; j < size; ++j) {
      result[i] = 0.0;
    }
  }

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {
    for (size_t j = 0; j < size; ++j) {
      result[i] += m[i * size + j] * full_x[j];
    }
  }
}

void mult_vector_by_vector(double* v1, double* v2, double &result, int mpi_rank, int* revcounts) {

  result = 0.0;

  double temp = 0.0;

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    temp += v1[i] * v2[i];
  }

  MPI_Allreduce(&temp, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void mult_double_by_vector(double x, double* v, double*  result, int mpi_rank, int* revcounts) {

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    result[i] = x * v[i];
  }
}

void add_vector_to_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts) {

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    result[i] = v1[i] + v2[i];
  }
}

void sub_vector_from_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts) {

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    result[i] = v1[i] - v2[i];
  }
}

double norma(double* v, int mpi_rank, int* revcounts) {

  double result = 0.0;
  double temp = 0.0;

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    temp += v[i] * v[i];
  }

  MPI_Allreduce(&temp, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  result = sqrt(result);

  return result;
}

void copy_vector_to_vector(double* from, double* to, int mpi_rank, int* revcounts) {

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    to[i] = from[i];
  }
}

void solve(double* A, double* b, size_t size, double*  result, int mpi_size, int mpi_rank, int* revcounts, int* displs) {

  const double epsilon = 1e-3;

  double* r = new double[revcounts[mpi_rank]];
  double* old_r = new double[revcounts[mpi_rank]];
  double* z = new double[revcounts[mpi_rank]];
  double alpha;
  double beta;
  double b_norm;
  double* Ax = new double[revcounts[mpi_rank]];
  double* Az = new double[revcounts[mpi_rank]];
  double Azz;
  double* alpha_A = new double[size * revcounts[mpi_rank]];
  double* alpha_A_z = new double[revcounts[mpi_rank]];
  double* alpha_z = new double[revcounts[mpi_rank]];
  double* beta_z = new double[revcounts[mpi_rank]];
  double rr;
  double oldr_oldr;
  double* full_x = new double[size];

  //r = b - A*x;
  mult_matrix_by_vector(A, result, size, Ax, mpi_rank, revcounts, displs, full_x);
  sub_vector_from_vector(b, Ax, r, mpi_rank, revcounts);

  //z = r;
  copy_vector_to_vector(r, z, mpi_rank, revcounts);

  b_norm = norma(b, mpi_rank, revcounts);

  do {

    //alpha = (r*r) / (A * z * z);
    mult_vector_by_vector(r, r, rr, mpi_rank, revcounts);
    mult_matrix_by_vector(A, z, size, Az, mpi_rank, revcounts, displs, full_x);
    mult_vector_by_vector(Az, z, Azz, mpi_rank, revcounts);
    alpha = rr / Azz;

    //x = x + alpha * z;
    mult_double_by_vector(alpha, z, alpha_z, mpi_rank, revcounts);
    add_vector_to_vector(result, alpha_z, result, mpi_rank, revcounts);

    //old_r = r
    copy_vector_to_vector(r, old_r, mpi_rank, revcounts);

    //r = r - alpha * A *z;
    mult_double_by_matrix(alpha, A, size, alpha_A, mpi_rank, revcounts);
    mult_matrix_by_vector(alpha_A, z, size, alpha_A_z, mpi_rank, revcounts, displs, full_x);
    sub_vector_from_vector(r, alpha_A_z, r, mpi_rank, revcounts);

    //beta = (r * r) / (old_r*old_r);
    mult_vector_by_vector(r, r, rr, mpi_rank, revcounts);
    mult_vector_by_vector(old_r, old_r, oldr_oldr, mpi_rank, revcounts);
    beta = rr / oldr_oldr;

    //z = r + beta * z;
    mult_double_by_vector(beta, z, beta_z, mpi_rank, revcounts);
    add_vector_to_vector(r, beta_z, z, mpi_rank, revcounts);

  } while (norma(r, mpi_rank, revcounts) / b_norm > epsilon);

  delete[] r;
  delete[] old_r;
  delete[] z;
  delete[] Ax;
  delete[] Az;
  delete[] alpha_A;
  delete[] alpha_A_z;
  delete[] alpha_z;
  delete[] beta_z;
  delete[] full_x;
}
