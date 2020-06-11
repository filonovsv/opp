#include "LinearAlgebraSubprograms.h"
#include <algorithm>
#include <omp.h>

void mult_double_by_matrix(double x, double** m, const size_t size, double**  result) {

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
#pragma omp parallel for
    for (int j = 0; j < size; j++) {

      result[i][j] = 0.0;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
#pragma omp parallel for
    for (int j = 0; j < size; j++) {  

      result[i][j] += m[i][j] * x;
    }
  }
}

void mult_matrix_by_vector(double** m, double* v, const size_t size, double*  result) {

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    result[i] = 0.0;
#pragma omp parallel for
    for (int j = 0; j < size; j++) {

      result[i] += m[i][j] * v[j];
    }
  }
}

void mult_vector_by_vector(double* v1, double* v2, const size_t size, double &result) {

  result = 0.0;

#pragma omp parallel for 
  for (int i = 0; i < size; i++) {

    result += v1[i] * v2[i];
  }
}

void mult_double_by_vector(double x, double* v, const size_t size, double*  result) {

#pragma omp parallel for
  for (int i = 0; i < size; i++) {

    result[i] = x * v[i];
  }
}

void add_vector_to_vector(double* v1, double* v2, const size_t size, double*  result) {

#pragma omp parallel for
  for (int i = 0; i < size; i++) {

    result[i] = v1[i] + v2[i];
  }
}

void sub_vector_from_vector(double* v1, double* v2, const size_t size, double*  result) {

#pragma omp parallel for
  for (int i = 0; i < size; i++) {

    result[i] = v1[i] - v2[i];
  }
}

double norma(double* v, const size_t size) {

  double result = 0;

#pragma omp parallel for
  for (int i = 0; i < size; i++) {

    result = v[i] * v[i];
  }

  result = sqrt(result);
  return result;
}

void copy_vector_to_vector(double* from, double* to, const size_t size) {

#pragma omp parallel for
  for (int i = 0; i < size; i++) {

    to[i] = from[i];
  }
}

void solve(double** A, double* b, const size_t size, double*  result) {

  omp_set_num_threads(4);

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
  double** alpha_A = new double*[size];
  for (int i = 0; i < size; i++) {

    alpha_A[i] = new double[size];
  }
  double* alpha_A_z = new double[size];
  double* alpha_z = new double[size];
  double* beta_z = new double[size];
  double rr;
  double oldr_oldr;

  //r = b - A*x;
  mult_matrix_by_vector(A, result, size, Ax);
  sub_vector_from_vector(b, Ax, size, r);

  //z = r;
  copy_vector_to_vector(r, z, size);

  b_norm = norma(b, size);

  do {

    //alpha = (r*r) / (A * z * z);
    mult_vector_by_vector(r, r, size, rr);
    mult_matrix_by_vector(A, z, size, Az);
    mult_vector_by_vector(Az, z, size, Azz);
    alpha = rr / Azz;

    //x = x + alpha * z;
    mult_double_by_vector(alpha, z, size, alpha_z);
    add_vector_to_vector(result, alpha_z, size, result);//!
    old_r = r;

    //r = r - alpha * A *z;
    mult_double_by_matrix(alpha, A, size, alpha_A);
    mult_matrix_by_vector(alpha_A, z, size, alpha_A_z);
    sub_vector_from_vector(r, alpha_A_z, size, r);//!

    //beta = (r * r) / (old_r*old_r);
    mult_vector_by_vector(r, r, size, rr);
    mult_vector_by_vector(old_r, old_r, size, oldr_oldr);
    beta = rr / oldr_oldr;

    //z = r + beta * z;
    mult_double_by_vector(beta, z, size, beta_z);
    add_vector_to_vector(r, beta_z, size, z);

  } while (norma(r, size) / b_norm > epsilon);
  
  delete[] r;
  delete[] old_r;
  delete[] z;
  delete[] Ax;
  delete[] Az;

  for (int i = 0; i < size; i++) {

    delete[] alpha_A[i];
  }
  delete[] alpha_A;
  delete[] alpha_A_z;
  delete[] alpha_z;
  delete[] beta_z;
}
