#include "ThreadManager.h"

void thread_manager(int &row_per_proc, int &col_per_proc, int n1, int n3, int *dims, int *coords) {

  if (coords[0] < n1 % dims[0]) {
    row_per_proc = n1 / dims[0] + 1;
  }
  else {
    row_per_proc = n1 / dims[0];
  }

  if (coords[1] < n3 % dims[1]) {
    col_per_proc = n3 / dims[1] + 1;
  }
  else {
    col_per_proc = n3 / dims[1];
  }
}
