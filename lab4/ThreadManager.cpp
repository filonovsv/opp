void thread_manager_init(int rank, int size, int NX, int &nx, int &shift) {
 
  if (rank < NX % size) {
    nx = NX / size + 1;
    shift = rank * nx;
  }
  else {
    nx = NX / size;
    shift = (NX % size) * (nx + 1) + (rank - (NX % size)) * nx;
  }
}
