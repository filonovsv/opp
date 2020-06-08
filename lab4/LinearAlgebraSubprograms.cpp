#include <mpi.h>
#include <iostream>
#include "ThreadManager.h"

constexpr auto X0 = -1;
constexpr auto Y0 = -1;
constexpr auto Z0 = -1;
constexpr auto X1 = 1;
constexpr auto Y1 = 1;
constexpr auto Z1 = 1;
constexpr auto PHI_0 = 0;
constexpr auto NX = 240;
constexpr auto NY = 240;
constexpr auto NZ = 240;
constexpr auto ALPHA = 1e+5;
constexpr auto EPSILON = 1e-8;

double phi(double x, double y, double z) {
  return pow(x, 2) + pow(y, 2) + pow(z, 2);
}

double ro(double x, double y, double z) {
  return 6.0 - ALPHA * phi(x, y, z);
}

void solve(int rank, int size, double &delta) {

  int nx = 0;
  int ny = NY;
  int nz = NZ;

  int shift = 0;

  thread_manager_init(rank, size, NX, nx, shift);

  double hx = (double)(abs(X1) + abs(X0)) / (double)(NX - 1);
  double hy = (double)(abs(Y1) + abs(Y0)) / (double)(NY - 1);
  double hz = (double)(abs(Z1) + abs(Z0)) / (double)(NZ - 1);

  double* prev_phi = new double[nx * ny * nz];
  double* next_phi = new double[nx * ny * nz];

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        if (0 != i + shift && 0 != j && 0 != k && NX - 1 != i + shift && NY - 1 != j && NZ - 1 != k) {
          prev_phi[i*ny*nz + j * nz + k] = 0;
          next_phi[i*ny*nz + j * nz + k] = 0;
        }
        else {
          prev_phi[i*ny*nz + j * nz + k] = phi(X0 + (i + shift)*hx, Y0 + j * hy, Z0 + k * hz);
          next_phi[i*ny*nz + j * nz + k] = phi(X0 + (i + shift)*hx, Y0 + j * hy, Z0 + k * hz);
        }
      }
    }
  }

  double denom = (2.0 / pow(hx, 2) + 2.0 / pow(hy, 2) + 2.0 / pow(hz, 2) + ALPHA);

  double* phi_x_lower_bound = new double[ny * nz];
  double* phi_x_upper_bound = new double[ny * nz];

  MPI_Request req_up_isend;
  MPI_Request req_up_irecv;
  MPI_Request req_down_isend;
  MPI_Request req_down_irecv;


  while (true) {

    if (size - 1 != rank) {
      MPI_Isend(&(prev_phi[(nx - 1)*ny*nz]), ny*nz, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req_down_isend);
      MPI_Irecv(phi_x_upper_bound, ny*nz, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req_up_irecv);
    }
    if (0 != rank) {
      MPI_Isend(&(prev_phi[0]), ny*nz, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req_up_isend);
      MPI_Irecv(phi_x_lower_bound, ny*nz, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req_down_irecv);
    }

    double max = 0.0;

    for (int i = 1; i < nx - 1; ++i) {
      for (int j = 1; j < ny - 1; ++j) {
        for (int k = 1; k < nz - 1; ++k) {
          next_phi[i*ny*nz + j * nz + k] =
            ((prev_phi[(i + 1)*ny*nz + j * nz + k] + prev_phi[(i - 1)*ny*nz + j * nz + k]) / pow(hx, 2) +
            (prev_phi[i*ny*nz + (j + 1)*nz + k] + prev_phi[i*ny*nz + (j - 1)*nz + k]) / pow(hy, 2) +
              (prev_phi[i*ny*nz + j * nz + (k + 1)] + prev_phi[i*ny*nz + j * nz + (k - 1)]
                ) / pow(hz, 2) - ro(X0 + (i + shift)*hx, Y0 + j * hy, Z0 + k * hz)
              ) / denom;
          if (abs(next_phi[i*ny*nz + j * nz + k] - prev_phi[i*ny*nz + j * nz + k]) > max) {
            max = abs(next_phi[i*ny*nz + j * nz + k] - prev_phi[i*ny*nz + j * nz + k]);
          }
        }
      }
    }

    if (size - 1 != rank) {
      MPI_Wait(&req_down_isend, MPI_STATUS_IGNORE);
      MPI_Wait(&req_up_irecv, MPI_STATUS_IGNORE);
    } if (0 != rank) {
      MPI_Wait(&req_up_isend, MPI_STATUS_IGNORE);
      MPI_Wait(&req_down_irecv, MPI_STATUS_IGNORE);
    }

    for (int j = 1; j < ny - 1; ++j) {
      for (int k = 1; k < nz - 1; ++k) {
        if (0 != rank) {
          next_phi[j*nz + k] =
            ((prev_phi[ny*nz + j * nz + k] + phi_x_lower_bound[j*nz + k]) / pow(hx, 2) +
            (prev_phi[(j + 1)*nz + k] + prev_phi[(j - 1)*nz + k]) / pow(hy, 2) +
              (prev_phi[j*nz + (k + 1)] + prev_phi[j*nz + (k - 1)]) / pow(hz, 2) - ro(X0 + shift * hx, Y0 + j * hy, Z0 + k * hz)
              ) / denom;
          if (abs(next_phi[j*nz + k] - prev_phi[j*nz + k]) > max) {
            max = abs(next_phi[j*nz + k] - prev_phi[j*nz + k]);
          }
        }

        if (size - 1 != rank) {
          next_phi[(nx - 1)*ny*nz + j * nz + k] =
            ((phi_x_upper_bound[j*nz + k] + prev_phi[(nx - 2)*ny*nz + j * nz + k]) / pow(hx, 2) +
            (prev_phi[(nx - 1)*ny*nz + (j + 1)*nz + k] + prev_phi[(nx - 1)*ny*nz + (j - 1)*nz + k]) / pow(hy, 2) +
              (prev_phi[(nx - 1)*ny*nz + j * nz + (k + 1)] + prev_phi[(nx - 1)*ny*nz + j * nz + (k - 1)]
                ) / pow(hz, 2) - ro(X0 + (nx - 1 + shift)*hx, Y0 + j * hy, Z0 + k * hz)
              ) / denom;
          if (abs(next_phi[(nx - 1)*ny*nz + j * nz + k] - prev_phi[(nx - 1)*ny*nz + j * nz + k]) > max) {
            max = abs(next_phi[(nx - 1)*ny*nz + j * nz + k] - prev_phi[(nx - 1)*ny*nz + j * nz + k]);
          }
        }
      }
    }

    std::swap(prev_phi, next_phi);

    double max_tmp = 0.0;
    MPI_Allreduce(&max, &max_tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (EPSILON > max_tmp) {
      break;
    }
  }

  double max = 0.0;

  for (int i = 0; i < nx; ++i) {
    for (int j = 1; j < ny - 1; ++j) {
      for (int k = 1; k < nz - 1; ++k) {
        if (i + shift != NX - 1 && i + shift != 0) {
          if (abs(prev_phi[i*ny*nz + j * nz + k] - phi(X0 + (i + shift)*hx, Y0 + j * hy, Z0 + k * hz)) > max) {
            max = abs(prev_phi[i*ny*nz + j * nz + k] - phi(X0 + (i + shift)*hx, Y0 + j * hy, Z0 + k * hz));
          }
        }
      }
    }
  }

  MPI_Reduce(&max, &delta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  delete[] prev_phi;
  delete[] next_phi;
  delete[] phi_x_upper_bound;
  delete[] phi_x_lower_bound;
}
