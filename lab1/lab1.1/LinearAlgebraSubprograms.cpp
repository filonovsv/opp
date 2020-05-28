#include "LinearAlgebraSubprograms.h"
#include <algorithm>
#include <mpi.h>


void mult_double_by_matrix(double x, double* m, size_t size, double*  result, int mpi_rank, int* revcounts) {

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {
    for (size_t j = 0; j < size; ++j) {
      result[i*size +j] = x * m[i * size + j];
    }
  }
}


void mult_matrix_by_vector(double* m, double* v, size_t size, double*  result, int mpi_rank, int* revcounts, int* displs) {


  double* rev = new double[revcounts[mpi_rank]];

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {
    rev[i] = 0.0;
  }

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {
    for (size_t j = 0; j < size; ++j) {
      rev[i] += m[i * size + j] * v[j];
    }
  }

  MPI_Allgatherv(rev, revcounts[mpi_rank], MPI_DOUBLE, result, revcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

  delete[] rev;
}


void mult_vector_by_vector(double* v1, double* v2, size_t size, double &result, int mpi_rank, int* revcounts, int* displs) {

  result = 0.0;

  double temp = 0.0;

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    temp += v1[displs[mpi_rank] + i] * v2[displs[mpi_rank] + i];
  }

  MPI_Allreduce(&temp, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void mult_double_by_vector(double x, double* v, size_t size, double*  result, int mpi_rank, int* revcounts, int* displs) {

  double* rev = new double[revcounts[mpi_rank]];

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    rev[i] = x * v[displs[mpi_rank] + i];
  }

  MPI_Allgatherv(rev, revcounts[mpi_rank], MPI_DOUBLE, result, revcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

  delete[] rev;
}

void add_vector_to_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts, int* displs) {

  double* rev = new double[revcounts[mpi_rank]];

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    rev[i] = v1[displs[mpi_rank] + i] + v2[displs[mpi_rank] + i];
  }

  MPI_Allgatherv(rev, revcounts[mpi_rank], MPI_DOUBLE, result, revcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

  delete[] rev;
}

void sub_vector_from_vector(double* v1, double* v2, double*  result, int mpi_rank, int* revcounts, int* displs) {

  double* rev = new double[revcounts[mpi_rank]];

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    rev[i] = v1[displs[mpi_rank] + i] - v2[displs[mpi_rank] + i];
  }

  MPI_Allgatherv(rev, revcounts[mpi_rank], MPI_DOUBLE, result, revcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

  delete[] rev;
}

double norma(double* v, int mpi_rank, int* revcounts, int* displs) {

  double result = 0.0;
  double temp = 0.0;

  for (size_t i = 0; i < revcounts[mpi_rank]; ++i) {

    temp += v[displs[mpi_rank] + i] * v[displs[mpi_rank] + i];
  }

  MPI_Allreduce(&temp, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  result = sqrt(result);

  return result;
}

void copy_vector_to_vector(double* from, double* to, size_t size) {

  for (size_t i = 0; i < size; ++i) {

    to[i] = from[i];
  }
}

void solve(double* A, double* b, size_t size, double*  result, int mpi_size, int mpi_rank, int* revcounts, int* displs) {

  const double epsilon = 1e-3;

  double* r = new double[size];
  double* old_r = new double[size];
  double* z = new double[size];
  double alpha;
  double beta;
  double b_norm;
  double* Ax = new double[size];
  double* Az = new double[size];
  double Azz;
  double* alpha_A = new double[size * revcounts[mpi_rank]];
  double* alpha_A_z = new double[size];
  double* alpha_z = new double[size];
  double* beta_z = new double[size];
  double rr;
  double oldr_oldr;

  //r = b - A*x;
  mult_matrix_by_vector(A, result, size, Ax, mpi_rank, revcounts, displs);
  sub_vector_from_vector(b, Ax, r, mpi_rank, revcounts, displs);

  //z = r;
  copy_vector_to_vector(r, z, size);

  b_norm = norma(b, mpi_rank, revcounts, displs);

  do {

    //alpha = (r*r) / (A * z * z);
    mult_vector_by_vector(r, r, size, rr, mpi_rank, revcounts, displs);
    mult_matrix_by_vector(A, z, size, Az, mpi_rank, revcounts, displs);
    mult_vector_by_vector(Az, z, size, Azz, mpi_rank, revcounts, displs);
    alpha = rr / Azz;

    //x = x + alpha * z;
    mult_double_by_vector(alpha, z, size, alpha_z, mpi_rank, revcounts, displs);
    add_vector_to_vector(result, alpha_z, result, mpi_rank, revcounts, displs);

    //old_r = r
    copy_vector_to_vector(r, old_r, size);

    //r = r - alpha * A *z;
    mult_double_by_matrix(alpha, A, size, alpha_A, mpi_rank, revcounts);
    mult_matrix_by_vector(alpha_A, z, size, alpha_A_z, mpi_rank, revcounts, displs);
    sub_vector_from_vector(r, alpha_A_z, r, mpi_rank, revcounts, displs);

    //beta = (r * r) / (old_r*old_r);
    mult_vector_by_vector(r, r, size, rr, mpi_rank, revcounts, displs);
    mult_vector_by_vector(old_r, old_r, size, oldr_oldr, mpi_rank, revcounts, displs);
    beta = rr / oldr_oldr;

    //z = r + beta * z;
    mult_double_by_vector(beta, z, size, beta_z, mpi_rank, revcounts, displs);
    add_vector_to_vector(r, beta_z, z, mpi_rank, revcounts, displs);

  } while (norma(r, mpi_rank, revcounts, displs) / b_norm > epsilon);

  delete[] r;
  delete[] old_r;
  delete[] z;
  delete[] Ax;
  delete[] Az;
  delete[] alpha_A;
  delete[] alpha_A_z;
  delete[] alpha_z;
  delete[] beta_z;
}
