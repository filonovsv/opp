#pragma once

void thread_manager_init(int size, int mpi_size, int* &revcounts, int* &displs);

void thread_manager_finalize(int* revcounts, int* displs);
