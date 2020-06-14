#define HAVE_STRUCT_TIMESPEC
#include <iostream>
#include <pthread.h>
#include <mpi.h>
#include <queue>
#include <chrono>
#include <thread>

constexpr auto COUNT_ITER = 100;
constexpr auto LIST_SIZE = 100;
constexpr auto WEIGHT = 350;

std::queue<int> tasks;
pthread_mutex_t mutex;

void make_tasks(int rank, int size) {

  int sum_ranks = (size + 1) * (size) / 2;
  int count_tasks = (LIST_SIZE / sum_ranks) * (rank + 1);
  if (size - 1 == rank) {
    count_tasks += LIST_SIZE % sum_ranks;
  }
  for (int i = 0; i < count_tasks; ++i) {
    tasks.push(WEIGHT);
  }
}

void do_tasks(int rank, int size) {

  int send_rank = 0;
  int i = 0;
  while (i < size) {
    pthread_mutex_lock(&mutex);
    while (0 < tasks.size()) {
      int weight = tasks.front();
      tasks.pop();
      pthread_mutex_unlock(&mutex);
      std::this_thread::sleep_for(std::chrono::microseconds(weight));
      pthread_mutex_lock(&mutex);
    }
    pthread_mutex_unlock(&mutex);
    send_rank = (rank + i) % size;
    if (send_rank == rank) {
      ++i;
      continue;
    }
    int send_request = 1;
    int weight;
    MPI_Send(&send_request, 1, MPI_INT, send_rank, 1, MPI_COMM_WORLD);
    MPI_Recv(&weight, 1, MPI_INT, send_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (0 > weight) {
      ++i;
      continue;
    }
    pthread_mutex_lock(&mutex);
    tasks.push(weight);
    pthread_mutex_unlock(&mutex);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return;
}

void* wait_request(void* args) {

  while (true) {
    int recv_request;
    MPI_Status status;
    MPI_Recv(&recv_request, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
    if (0 == recv_request) {
      break;
    }
    int weight;
    pthread_mutex_lock(&mutex);
    if (0 == tasks.size()) {
      weight = -1;
      pthread_mutex_unlock(&mutex);
      MPI_Send(&weight, 1, MPI_INT, status.MPI_SOURCE, 2, MPI_COMM_WORLD);
      continue;
    }
    weight = tasks.front();
    tasks.pop();
    pthread_mutex_unlock(&mutex);
    MPI_Send(&weight, 1, MPI_INT, status.MPI_SOURCE, 2, MPI_COMM_WORLD);
  }
  return nullptr;
}

int main(int argc, char *argv[]) {

  int provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  if (MPI_THREAD_MULTIPLE != provided) {
    std::cerr << "required level was not provided" << std::endl;
    MPI_Finalize();
    abort();
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 != pthread_mutex_init(&mutex, NULL)) {
    std::cerr << "pthread_mutex_init";
    abort();
  }
  pthread_attr_t attr;
  if (0 != pthread_attr_init(&attr)) {
    std::cerr << "pthread_attr_init";
    abort();
  }
  if (0 != pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE)) {
    std::cerr << "pthread_attr_setdetachstate";
    abort();
  }
  double start = MPI_Wtime();

  pthread_t waiting_thread;

  if (0 != pthread_create(&waiting_thread, &attr, wait_request, NULL)) {
    std::cerr << "pthread_create";
    abort();
  }

  for (int i = 0; i < COUNT_ITER; ++i) {
    pthread_mutex_lock(&mutex);
    make_tasks(rank, size);
    pthread_mutex_unlock(&mutex);
    do_tasks(rank, size);
  }

  int send_request = 0;

  MPI_Send(&send_request, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
  double finish = MPI_Wtime();

  pthread_attr_destroy(&attr);

  if (0 != pthread_join(waiting_thread, NULL)) {
    std::cerr << "pthread_join";
    abort();
  }

  if (0 != pthread_mutex_destroy(&mutex)) {
    std::cerr << "pthread_mutex_destroy";
    abort();
  }

  if (0 == rank) {
    std::cout << "time: " << finish - start << std::endl;
  }

  MPI_Finalize();
  return 0;
}
