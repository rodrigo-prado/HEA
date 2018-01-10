#ifndef CUDA_CODE_H_
#define CUDA_CODE_H_

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Data.h"
#include "Chromosome.h"

using namespace std;

extern int four_node_num_moves;

void check_cudaError(cudaError_t error){
	if(error != cudaSuccess){
		cout << "Error: " << cudaGetErrorString(error) << endl;
		exit(-1);
	}
}


/**
  * Class that represents a swap
  */
struct Move {
	__device__ __host__ explicit Move()
		: i(0), j(0), solution(NULL)  {}

	/**
	  * Constructor that represents the swap of node i and j
	  */
	__device__ explicit Move(unsigned int i_, unsigned int j_, int* solution_)
		: i(i_), j(j_), solution(solution_) {}

	int* solution;
	unsigned int i,j;
};

struct One_node_move {
	__device__ __host__ explicit One_node_move()
		: i(0), solution(NULL)  {}

	__device__ explicit One_node_move(unsigned int i_, int *solution_)
		: i(i_), solution(solution_) {}

	int* solution;
	unsigned int i;
};

struct Four_node_move {
	__device__ __host__ explicit Four_node_move()
		: i(0), j(0), m(0), n(0), solution(NULL)  {}

	__device__ explicit Four_node_move(unsigned int i_, unsigned int j_, unsigned int m_, unsigned int n_, int* solution_)
		: i(i_), j(j_), m(m_), n(n_), solution(solution_) {}

	int* solution;
	unsigned int i, j, m, n;
};

struct Solution {
    int * allocation;
    double * runtime;
};

struct Environment {
	double * base;
	unsigned int * whatisit;

	double * slowdown;
	double * link;

	int num_vms;
	int size;
};

/**
 * The cost of swapping node i with node j.
 */
__device__ double cost(const Environment & env_, const Solution & solution_, const Move & move, const unsigned int & num_nodes_) {
	int i = move.i;
	int j = move.j;

	int vmi = solution_.allocation[i];
	int vmj = solution_.allocation[j];

	double initial_cost, final_cost;

	initial_cost =  final_cost = solution_.runtime[i] + solution_.runtime[j];


	if(vmi != vmj){
		double fiti = env_.base[i];
		double fitj = env_.base[j];

		if(env_.whatisit[i] == 1)
			fiti *= env_.slowdown[vmj];
		else
			fiti /= env_.link[vmj];

		if(env_.whatisit[i] == 1)
			fitj *= env_.slowdown[vmi];
		else
			fitj /= env_.link[vmi];

		final_cost = fiti + fitj;
	}

	// if(initial_cost - final_cost > 0)
	// 	printf("dif: %f initial: %f  final: %f i: %d j: %d\n", initial_cost - final_cost, initial_cost, final_cost, i, j);

	return initial_cost - final_cost; // não houve mudança no fitness

}

/**
 * Apply the move
 */
__device__ void apply_move(Move& move) {
	unsigned int tmp = move.solution[move.i];
	move.solution[move.i] = move.solution[move.j];
	move.solution[move.j] = tmp;

	// printf("Move applied: %d[%d] %d[%d]\n", move.solution[move.i], move.i, move.solution[move.j], move.j);
}

/**
 * @param move_number Move number to generate
 * @param num_nodes_ number of nodes
 */
__device__ Move generate_move(const unsigned int & move_number_, const unsigned int & num_nodes_, int * solution_) {
	float n = static_cast<float>(num_nodes_);
	float i = static_cast<float>(move_number_);

	//Generates move number i in the following series:
	//(1, 2), (1, 3), ..., (1, n-2), (1, n-1)
	//(2, 3), (2, 4), ..., (2, n-1)
	// ...
	//(n-2, n-1)
	float dx = n-2.0f-floor((sqrtf(4.0f*(n-1.0f)*(n-2.0f) - 8.0f*i - 7.0f)-1.0f)/2.0f);
	float dy = 2.0f+i-(dx-1)*(n-2.0f)+(dx-1.0f)*dx/2.0f;

	unsigned int x = static_cast<unsigned int>(dx);
	unsigned int y = static_cast<unsigned int>(dy);

   // printf("x: %d y: %d\n", x, y);

	return Move(x, y, solution_);
}


/**
 * CUDA kernel that executes local search on the GPU
 */
__global__ void evaluate_moves_kernel(const Environment env_, const Solution solution_, const double fitness, unsigned int *moves_, double * deltas_, unsigned int num_nodes_, unsigned int num_moves_per_thread_) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int num_moves = (static_cast<int>(num_nodes_)-2) * (static_cast<int>(num_nodes_)-1)/2;

	double max_delta =  -1 * fitness;

	// printf("Max delta : %f\n", max_delta);

	const unsigned int first_move = tid * num_moves_per_thread_;

	unsigned int best_move = first_move;

	for (int i = first_move; i < (first_move + num_moves_per_thread_); ++i) {
		if (i < num_moves) {
			Move move = generate_move(i, num_nodes_, solution_.allocation);
			double move_cost = cost(env_, solution_, move, num_nodes_);
			if (move_cost > max_delta) {
			 	max_delta = move_cost;
			 	best_move = i;
			}
		}
	}
	// printf("best_move: %d best delta: %f\n", best_move, max_delta);
	deltas_[tid] = max_delta;
	moves_[tid] = best_move;
}

template<unsigned int threads>
__device__ void reduce_to_minimum(const unsigned int & tid, double (&deltas_shmem_)[threads], unsigned int (&moves_shmem_)[threads]) {
	volatile double (&deltas_shmem_volatile_)[threads] = deltas_shmem_;
	volatile unsigned int (&moves_shmem_volatile_)[threads] = moves_shmem_;

	for (unsigned int i = 1; i < threads; i *= 2) {
		const unsigned int k = 2 * i *tid;
		if (k + i < threads) {
			if (deltas_shmem_volatile_[k+i] < deltas_shmem_volatile_[k]) {
				deltas_shmem_volatile_[k] = deltas_shmem_volatile_[k+i];
				moves_shmem_volatile_[k] = moves_shmem_volatile_[k+i];
			}
		}
		__syncthreads();
	}
}

template<unsigned int threads>
__device__ void reduce_to_maximum(const unsigned int & tid, double (&deltas_shmem_)[threads], unsigned int (&moves_shmem_)[threads]) {
	volatile double (&deltas_shmem_volatile_)[threads] = deltas_shmem_;
	volatile unsigned int (&moves_shmem_volatile_)[threads] = moves_shmem_;

	for (unsigned int i = 1; i < threads; i *= 2) {
		const unsigned int k = 2 * i *tid;
		if (k + i < threads) {
			if (deltas_shmem_volatile_[k+i] > deltas_shmem_volatile_[k]) {
				deltas_shmem_volatile_[k] = deltas_shmem_volatile_[k+i];
				moves_shmem_volatile_[k] = moves_shmem_volatile_[k+i];
			}
		}
		__syncthreads();
	}
}

/**
 * Kernel that reduces deltas_ into a single element, and then
 * applies this move
 */
template <unsigned int threads>
__global__ void apply_best_move_kernel(const Solution solution_, double* deltas_,  unsigned int* moves_, unsigned int num_nodes_, unsigned int deltas_size_) {
	// Thread id
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//Shared memory available to all threads to find the minimum delta
	__shared__ double deltas_shmem[threads];

	//Shared memory to find the index corresponding to the minimum delta
	__shared__ unsigned int moves_shmem[threads];

	//Reduce from num_nodes_ elements to blockDim.x elements
	deltas_shmem[tid] = 0;
	unsigned int reducePerThread = (deltas_size_ + threads-1) / threads; // ceiling(deltas_size_/threads)
	for (unsigned int i = tid * reducePerThread; i < (tid+1) * reducePerThread; ++i) {
		if (i < deltas_size_) {
			if (deltas_[i] > deltas_shmem[tid]) {
				deltas_shmem[tid] = deltas_[i];
				moves_shmem[tid] = moves_[i];
			}
		}
	}

	__syncthreads();

	//Now, reduce all elements into a single element
	// reduce_to_minimum<threads>(tid, deltas_shmem, moves_shmem);
	reduce_to_maximum<threads>(tid, deltas_shmem, moves_shmem);
	if (tid == 0) {
			Move move = generate_move(moves_shmem[0], num_nodes_, solution_.allocation);
			apply_move(move);
			deltas_[0] = deltas_shmem[0];
	 }
}

__device__ double compute_move_cost(const Environment& env_, const unsigned int& num_VMs_, int node_i, int current_allocation, double solution_cost)
{
	double temp;

	for (int new_allocation = 0; new_allocation < num_VMs_; ++new_allocation) {
		if(current_allocation != new_allocation) {
			double current_allocation_cost = env_.base[node_i];
			double new_allocation_cost = current_allocation_cost;
			if(env_.whatisit[node_i] == 1) {
				current_allocation_cost *= env_.slowdown[current_allocation];
				new_allocation_cost *= env_.slowdown[new_allocation];
			} else {
				current_allocation_cost /= env_.link[current_allocation];
				new_allocation_cost /= env_.link[new_allocation];
			}

			if (new_allocation_cost < solution_cost) {
				temp = new_allocation_cost;
				solution_cost = temp;
			}
		}
	}
	return solution_cost;
}

__device__ double one_node_move_cost(const Environment & env_, const Solution & solution_, const One_node_move & one_node_move, const unsigned int & num_VMs_)
{
	int node_i = one_node_move.i;

	int vm_i = solution_.allocation[node_i];
	double initial_cost, final_cost;

	initial_cost = solution_.runtime[node_i];

	final_cost = compute_move_cost(env_, num_VMs_, node_i, vm_i, initial_cost);

	return initial_cost - final_cost; // não houve mudança no fitness
}

__global__ void evaluate_one_node_moves_kernel(const Environment env_, const Solution solution_, unsigned int* one_node_moves_, double* deltas_, unsigned int num_moves,	unsigned int num_VMs_)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid <= num_moves) {

		One_node_move move = One_node_move(tid, solution_.allocation);

		double fitness_delta = one_node_move_cost(env_, solution_, move, num_VMs_);

		deltas_[tid] = fitness_delta;
	  one_node_moves_[tid] = tid;
	}
}

template <unsigned int threads>
__global__ void find_best_one_move_kernel(const Solution solution_, double *deltas_, unsigned int *moves_, unsigned int num_nodes_, unsigned int deltas_size_)
{
	// Thread id
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//Shared memory available to all threads to find the minimum delta
	__shared__ double deltas_shmem[threads];

	//Shared memory to find the index corresponding to the minimum delta
	__shared__ unsigned int moves_shmem[threads];

	//Reduce from num_nodes_ elements to blockDim.x elements
	deltas_shmem[tid] = 0;
	unsigned int reducePerThread = (deltas_size_ + threads-1) / threads; // ceiling(deltas_size_/threads)
	for (unsigned int i = tid * reducePerThread; i < (tid+1) * reducePerThread; ++i) {
		if (i < deltas_size_) {
			if (deltas_[i] > deltas_shmem[tid]) {
				deltas_shmem[tid] = deltas_[i];
				moves_shmem[tid] = moves_[i];
			}
		}
	}

	__syncthreads();

	//Now, reduce all elements into a single element
	reduce_to_maximum<threads>(tid, deltas_shmem, moves_shmem);

	if (tid == 0) {
		moves_[0] = moves_shmem[0];
	}
}

__device__ double compute_two_node_move_cost(const Environment& env_, const unsigned int& num_VMs_, int node_i, int node_j, int node_i_allocation, int node_j_allocation, double solution_cost)
{
	double temp, new_solution_cost;

	for (int new_allocation_j = 0; new_allocation_j < num_VMs_; ++new_allocation_j) {
		if (new_allocation_j != node_j_allocation) {
			double new_allocation_j_cost = env_.base[node_j];

			for (int new_allocation_i = 0; new_allocation_i < num_VMs_; ++new_allocation_i) {
				if (new_allocation_i != node_i_allocation) {
					double new_allocation_i_cost = env_.base[node_i];

					new_allocation_i_cost = (env_.whatisit[node_i] == 1) ? (new_allocation_i_cost * env_.slowdown[new_allocation_i]) : (new_allocation_i_cost / env_.link[new_allocation_i]);
					new_allocation_j_cost = (env_.whatisit[node_j] == 1) ? (new_allocation_j_cost * env_.slowdown[new_allocation_j]) : (new_allocation_j_cost / env_.link[new_allocation_j]);

					new_solution_cost = new_allocation_i_cost + new_allocation_j_cost;

					if (new_solution_cost + new_allocation_j_cost < solution_cost) {
						temp = new_solution_cost;
						solution_cost = temp;
					}
				}
			}
		}
	}
	return solution_cost;
}

__device__ double two_node_move_cost(const Environment & env_,	const Solution & solution_,	const Move & two_node_move, const unsigned int & num_VMs_)
{
	int node_i = two_node_move.i;
	int node_j = two_node_move.j;

	int vm_i = solution_.allocation[node_i];
	int vm_j = solution_.allocation[node_j];

	double initial_cost, final_cost, solution_i_cost, solution_j_cost;

	solution_i_cost = solution_.runtime[node_i];
	solution_j_cost = solution_.runtime[node_j];

	initial_cost = solution_i_cost + solution_j_cost;

	final_cost = compute_two_node_move_cost(env_, num_VMs_, node_i, node_j, vm_i, vm_j, initial_cost);

	return initial_cost - final_cost;
}

__global__ void evaluate_two_node_moves_kernel(const Environment env_, const Solution solution_, unsigned int *two_node_moves_, double *deltas_, unsigned int num_nodes_, unsigned int num_moves_per_thread_, const unsigned int num_moves_,	unsigned int num_VMs_)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const int num_moves = static_cast<int>(num_moves_);

	double max_delta = -1;

	const unsigned int first_move = tid * num_moves_per_thread_;

	unsigned int best_move = first_move;

	for (int i = first_move; i < (first_move + num_moves_per_thread_); ++i) {
		if (i < num_moves) {

			Move move = generate_move(i, num_nodes_, solution_.allocation);
			double move_cost = two_node_move_cost(env_, solution_, move, num_VMs_);

			if (move_cost > max_delta) {
			 	max_delta = move_cost;
			 	best_move = i;
			}
		}
	}
	deltas_[tid] = max_delta;
	two_node_moves_[tid] = best_move;
}

template <unsigned int threads>
__global__ void find_best_two_node_move_kernel(const Solution solution_, double *deltas_, unsigned int *moves_, unsigned int num_nodes_, unsigned int deltas_size_)
{
	// Thread id
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//Shared memory available to all threads to find the minimum delta
	__shared__ double deltas_shmem[threads];

	//Shared memory to find the index corresponding to the minimum delta
	__shared__ unsigned int moves_shmem[threads];

	//Reduce from num_nodes_ elements to blockDim.x elements
	deltas_shmem[tid] = 0;
	unsigned int reducePerThread = (deltas_size_ + threads-1) / threads; // ceiling(deltas_size_/threads)
	for (unsigned int i = tid * reducePerThread; i < (tid+1) * reducePerThread; ++i) {
		if (i < deltas_size_) {
			if (deltas_[i] > deltas_shmem[tid]) {
				deltas_shmem[tid] = deltas_[i];
				moves_shmem[tid] = moves_[i];
			}
		}
	}

	__syncthreads();

	//Now, reduce all elements into a single element
	reduce_to_maximum<threads>(tid, deltas_shmem, moves_shmem);

	if (tid == 0) {
		Move move = generate_move(moves_shmem[0], num_nodes_, solution_.allocation);
		moves_[0] = move.i;
		moves_[1] = move.j;
	}
}

__device__ double four_node_move_cost(const Environment& env_,	const Solution& solution_,	const Four_node_move& four_node_move, const unsigned int& num_VMs_)
{
	int node_i = four_node_move.i;
	int node_j = four_node_move.j;
	int node_m = four_node_move.m;
	int node_n = four_node_move.n;

	int vm_i = solution_.allocation[node_i];
	int vm_j = solution_.allocation[node_j];
	int vm_m = solution_.allocation[node_m];
	int vm_n = solution_.allocation[node_n];

	double initial_cost, final_cost, partial_cost_1, partial_cost_2, solution_i_cost, solution_j_cost, solution_m_cost, solution_n_cost;

	solution_i_cost = solution_.runtime[node_i];
	solution_j_cost = solution_.runtime[node_j];
	solution_m_cost = solution_.runtime[node_m];
	solution_n_cost = solution_.runtime[node_n];

	partial_cost_1 = solution_i_cost + solution_j_cost;
	partial_cost_2 = solution_m_cost + solution_n_cost;

	initial_cost = partial_cost_1 + partial_cost_2;

	final_cost = compute_two_node_move_cost(env_, num_VMs_, node_i, node_j, vm_i, vm_j, partial_cost_1);
	final_cost += compute_two_node_move_cost(env_, num_VMs_, node_m, node_n, vm_m, vm_n, partial_cost_2);

	return initial_cost - final_cost;
}

__device__ Four_node_move generate_four_node_move(int* h_four_node_move_matrix, const unsigned int& move_number_, const unsigned int num_nodes_, int* solution_)
{
	int i = static_cast<int>(move_number_); // Move number to generate

	int dx = h_four_node_move_matrix[i * num_nodes_ + 0];
	int dy = h_four_node_move_matrix[i * num_nodes_ + 1];
	int dz = h_four_node_move_matrix[i * num_nodes_ + 2];
	int dk = h_four_node_move_matrix[i * num_nodes_ + 3];

	unsigned int x = static_cast<unsigned int>(dx);
	unsigned int y = static_cast<unsigned int>(dy);
	unsigned int z = static_cast<unsigned int>(dz);
	unsigned int k = static_cast<unsigned int>(dk);

	return Four_node_move(x, y, z, k, solution_);
}

__global__ void evaluate_four_node_moves_kernel(int* h_four_node_move_matrix, const Environment env_, const Solution solution_, unsigned int* four_node_moves_, double* deltas_, unsigned int num_nodes_, unsigned int num_moves_per_thread_, const unsigned int num_moves_, unsigned int num_VMs_)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const int num_moves = static_cast<int>(num_moves_);

	double max_delta = -1;

	const unsigned int first_move = tid * num_moves_per_thread_;

	unsigned int best_move = first_move;

	for (int i = first_move; i < (first_move + num_moves_per_thread_); ++i) {
		if (i < num_moves) {
			Four_node_move move = generate_four_node_move(h_four_node_move_matrix, i, num_nodes_, solution_.allocation);
			double move_cost = four_node_move_cost(env_, solution_, move, num_VMs_);

			if (move_cost > max_delta) {
			 	max_delta = move_cost;
			 	best_move = i;
			}
		}
	}
	deltas_[tid] = max_delta;
	four_node_moves_[tid] = best_move;
}

template <unsigned int threads>
__global__ void find_best_four_node_move_kernel(int* h_four_node_move_matrix, const Solution solution_, double* deltas_, unsigned int* moves_, unsigned int num_nodes_, unsigned int deltas_size_)
{
	// Thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Shared memory available to all threads to find the minimum delta
	__shared__ double deltas_shmem[threads];

	//Shared memory to find the index corresponding to the minimum delta
	__shared__ unsigned int moves_shmem[threads];

	//Reduce from num_nodes_ elements to blockDim.x elements
	deltas_shmem[tid] = 0;
	unsigned int reducePerThread = (deltas_size_ + threads-1) / threads; // ceiling(deltas_size_/threads)
	for (unsigned int i = tid * reducePerThread; i < (tid+1) * reducePerThread; ++i) {
		if (i < deltas_size_) {
			if (deltas_[i] > deltas_shmem[tid]) {
				deltas_shmem[tid] = deltas_[i];
				moves_shmem[tid] = moves_[i];
			}
		}
	}

	__syncthreads();

	//Now, reduce all elements into a single element
	reduce_to_maximum<threads>(tid, deltas_shmem, moves_shmem);

	if (tid == 0) {
		Four_node_move move = generate_four_node_move(h_four_node_move_matrix, moves_shmem[0], num_nodes_, solution_.allocation);
		moves_[0] = move.i;
		moves_[1] = move.j;
		moves_[2] = move.m;
		moves_[3] = move.n;
	}
}


inline Chromosome best_node_allocation(Chromosome solution, unsigned int node)
{
	Chromosome new_solution(solution);
	solution.computeFitness();

	int current_vm = solution.allocation[node];
	int num_VMs = solution.vm_queue.size();
	double menor = solution.fitness;

	for (int new_vm = 0; new_vm < num_VMs; ++new_vm) {
		if(current_vm != new_vm) {
			new_solution.allocation[node] = new_vm;
			new_solution.computeFitness();
			if (new_solution.fitness < menor) {
				menor = new_solution.fitness;
			}
		}
	}
	return new_solution.fitness < solution.fitness ? new_solution : solution;
}

inline Chromosome best_two_node_allocation(Chromosome solution, unsigned int node_1, unsigned int node_2)
{
	Chromosome new_solution(solution);
	solution.computeFitness();

	int current_vm_node_1 = solution.allocation[node_1];
	int current_vm_node_2 = solution.allocation[node_2];
	int num_VMs = solution.vm_queue.size();
	double menor = solution.fitness;

	for (int new_node_1_vm = 0; new_node_1_vm < num_VMs; ++new_node_1_vm) {
		if(current_vm_node_1 != new_node_1_vm) {

			for (int new_node_2_vm = 0; new_node_2_vm < num_VMs; ++new_node_2_vm) {
				if(current_vm_node_2 != new_node_2_vm) {

					new_solution.allocation[node_1] = new_node_1_vm;
					new_solution.allocation[node_2] = new_node_2_vm;
					new_solution.computeFitness();

					if (new_solution.fitness < menor)
						menor = new_solution.fitness;
				}
			}
		}
	}
	return new_solution.fitness < solution.fitness ? new_solution : solution;
}

inline Chromosome best_four_node_allocation(Chromosome solution, unsigned int node_1, unsigned int node_2, unsigned int node_3, unsigned int node_4)
{
	Chromosome new_solution(solution);
	solution.computeFitness();

	int current_vm_node_1 = solution.allocation[node_1];
	int current_vm_node_2 = solution.allocation[node_2];
	int current_vm_node_3 = solution.allocation[node_1];
	int current_vm_node_4 = solution.allocation[node_1];

	int num_VMs = solution.vm_queue.size();
	double menor = solution.fitness;

	for (int new_node_1_vm = 0; new_node_1_vm < num_VMs; ++new_node_1_vm) {
		if(current_vm_node_1 != new_node_1_vm) {

			for (int new_node_2_vm = 0; new_node_2_vm < num_VMs; ++new_node_2_vm) {
				if(current_vm_node_2 != new_node_2_vm) {

					for (int new_node_3_vm = 0; new_node_3_vm < num_VMs; ++new_node_3_vm) {
						if(current_vm_node_3 != new_node_3_vm) {

							for (int new_node_4_vm = 0; new_node_4_vm < num_VMs; ++new_node_4_vm) {
								if(current_vm_node_4 != new_node_4_vm) {

									new_solution.allocation[node_1] = new_node_1_vm;
									new_solution.allocation[node_2] = new_node_2_vm;
									new_solution.allocation[node_3] = new_node_3_vm;
									new_solution.allocation[node_4] = new_node_4_vm;

									new_solution.computeFitness();

									if (new_solution.fitness < menor)
										menor = new_solution.fitness;
								}
							}
						}
					}
				}
			}
		}
	}
	return new_solution.fitness < solution.fitness ? new_solution : solution;
}


/**
* Local Search with 'Best Accept' strategy (steepest descent)
* @param solution:     Current solution, will be changed during local search
* @param neighborhood: Collection of moves == potential neighbors
*/
Chromosome local_search(Chromosome solution_old,  Data * data, Environment h_env) {
	//Number of legal moves.
	//We do not want to swap the first node, which means we have n-1 nodes we can swap
	//For the first node, we then have n-2 nodes to swap with,
	//for the second node, we have n-3 nodes to swap with, etc.
	//and for the n-2'th node, we have 1 node to swap with.
	//which gives us sum_i=1^{n-2} i legal swaps,
	//or (n-2)(n-1)/2 legal swaps

	// cudaEvent_t start, stop;

	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, cudaEventDefault);


	Chromosome solution(solution_old);
	solution.computeFitness();

	const unsigned int num_nodes = solution.allocation.size() - 1;
	const unsigned int num_moves = (num_nodes-2)*(num_nodes-1)/2;

	// const unsigned int num_evaluate_threads = 8192;
	const unsigned int num_evaluate_threads = 1024;
	const dim3 evaluate_block(128);
	const dim3 evaluate_grid((num_evaluate_threads + evaluate_block.x-1) / evaluate_block.x);


	const unsigned int num_moves_per_thread = (num_moves+num_evaluate_threads-1)/num_evaluate_threads;


	const unsigned int num_apply_threads = 512;
	const dim3 apply_block(num_apply_threads);
	const dim3 apply_grid(1);


	// cout << "num_nodes: " << num_nodes << endl;
	// cout << "num_moves: "  << num_moves << endl;
	// cout << "num_evaluate_threads: " << num_evaluate_threads << endl;
	// cout << "num_moves_per_thread: " << num_moves_per_thread << endl;
	// cout << "num_apply_threads: " << num_apply_threads << endl;


	// cout << "Solution without local search: " << endl;
	// solution.print();

	//Pointer to memory on the GPU
	int *d_allocation;
	double *d_runtime;


    //Allocate GPU memory for solution
    check_cudaError(cudaMalloc(&(d_allocation), solution.allocation.size() * sizeof(int)));
    check_cudaError(cudaMalloc(&(d_runtime), solution.runtime_vector.size() * sizeof(double)));

    //Copy solution to GPU
	check_cudaError(cudaMemcpy(d_allocation, &(solution.allocation[0]),solution.allocation.size() * sizeof(int), cudaMemcpyHostToDevice));
    check_cudaError(cudaMemcpy(d_runtime, &(solution.runtime_vector[0]),solution.runtime_vector.size() * sizeof(double), cudaMemcpyHostToDevice));


	Solution h_solution;

    h_solution.allocation = d_allocation;
    h_solution.runtime = d_runtime;




   double * h_deltas;
   unsigned int * h_moves;

   //Allocate memory for deltas
   check_cudaError(cudaMalloc(&h_deltas, num_evaluate_threads * sizeof(double)));
   //Allocate memory for moves
   check_cudaError(cudaMalloc(&h_moves, num_evaluate_threads * sizeof(unsigned int)));

   evaluate_moves_kernel <<< evaluate_grid, evaluate_block >>>(h_env, h_solution, solution.fitness, h_moves, h_deltas, num_nodes, num_moves_per_thread);
   apply_best_move_kernel<num_apply_threads><<<apply_grid, apply_block>>>(h_solution, h_deltas, h_moves, num_nodes, num_evaluate_threads);


	double max_delta = 0.0;
	check_cudaError(cudaMemcpy(&max_delta, &h_deltas[0], sizeof(double), cudaMemcpyDeviceToHost));

	//Copy solution to CPU
	check_cudaError(cudaMemcpy(&(solution.allocation[0]), h_solution.allocation, solution.allocation.size() * sizeof(int), cudaMemcpyDeviceToHost));


	cudaFree(h_deltas);
	cudaFree(h_moves);

	cudaFree(d_allocation);
	cudaFree(d_runtime);

	solution.computeFitness();

	// cudaEventRecord(stop, cudaEventDefault);
	// cudaEventSynchronize(stop);
 	// cudaEventElapsedTime(&elapsedTime, start, stop);



	return solution.fitness < solution_old.fitness ? solution : solution_old;
}

Chromosome one_node_move_local_search(Chromosome solution_old, Data* data, Environment h_env) {

	Chromosome solution(solution_old);

	const unsigned int num_VMs = solution.vm_queue.size();
	const unsigned int num_nodes = solution.allocation.size() - 1;
	const unsigned int num_moves = num_nodes;solution.allocation.size() - 1;

	const unsigned int num_evaluate_threads = num_nodes;
	const dim3 evaluate_block(1);
	const dim3 evaluate_grid(num_evaluate_threads);

	const unsigned int num_find_threads = 512;
	const dim3 find_block(num_find_threads);
	const dim3 find_grid(1);

	//Pointer to memory on the GPU
	int* d_allocation;
	double* d_runtime;

	cudaSetDevice(0);
	//Allocate GPU memory for solution
	check_cudaError(cudaMalloc(&(d_allocation), solution.allocation.size() * sizeof(int)));
	check_cudaError(cudaMalloc(&(d_runtime), solution.runtime_vector.size() * sizeof(double)));

	//Copy solution to GPU
	check_cudaError(cudaMemcpy(d_allocation, &(solution.allocation[0]),solution.allocation.size() * sizeof(int), cudaMemcpyHostToDevice));
	check_cudaError(cudaMemcpy(d_runtime, &(solution.runtime_vector[0]),solution.runtime_vector.size() * sizeof(double), cudaMemcpyHostToDevice));

	Solution h_solution;
	h_solution.allocation = d_allocation;
	h_solution.runtime = d_runtime;

	double* h_deltas;
	unsigned int* h_moves;

	//Allocate memory for deltas
	check_cudaError(cudaMalloc(&h_deltas, num_evaluate_threads * sizeof(double)));
	//Allocate memory for moves
	check_cudaError(cudaMalloc(&h_moves, num_evaluate_threads * sizeof(unsigned int)));

	evaluate_one_node_moves_kernel <<< evaluate_grid, evaluate_block >>>(h_env, h_solution, h_moves, h_deltas, num_moves, num_VMs);
	find_best_one_move_kernel <num_find_threads> <<<find_grid, find_block>>>(h_solution, h_deltas, h_moves, num_nodes, num_evaluate_threads);

	unsigned int node_pos;

	// Copy best movement to the CPU
	check_cudaError(cudaMemcpy(&node_pos, &h_moves[0], sizeof(unsigned int), cudaMemcpyDeviceToHost));

	solution = best_node_allocation(solution, node_pos);

	cudaFree(h_deltas);
	cudaFree(h_moves);
	cudaFree(d_allocation);
	cudaFree(d_runtime);

	return solution.fitness < solution_old.fitness ? solution : solution_old;
}

Chromosome two_node_move_local_search(Chromosome solution_old, Data* data, Environment h_env) {
	Chromosome solution(solution_old);

	const unsigned int num_VMs = solution.vm_queue.size();
	const unsigned int num_nodes = solution.allocation.size() - 1;
	const unsigned int num_moves = (num_nodes-2)*(num_nodes-1)/2;

	const unsigned int num_evaluate_threads = 1024;
	const dim3 evaluate_block(128);
	const dim3 evaluate_grid((num_evaluate_threads + evaluate_block.x-1) / evaluate_block.x);

	const unsigned int num_moves_per_thread = (num_moves+num_evaluate_threads-1)/num_evaluate_threads;
	const unsigned int num_find_threads = 512;
	const dim3 find_block(num_find_threads);
	const dim3 find_grid(1);

	//Pointer to memory on the GPU
	int* d_allocation;
	double* d_runtime;
	cudaSetDevice(0);
	//Allocate GPU memory for solution
	check_cudaError(cudaMalloc(&(d_allocation), solution.allocation.size() * sizeof(int)));
	check_cudaError(cudaMalloc(&(d_runtime), solution.runtime_vector.size() * sizeof(double)));

	//Copy solution to GPU
	check_cudaError(cudaMemcpy(d_allocation, &(solution.allocation[0]),solution.allocation.size() * sizeof(int), cudaMemcpyHostToDevice));
	check_cudaError(cudaMemcpy(d_runtime, &(solution.runtime_vector[0]),solution.runtime_vector.size() * sizeof(double), cudaMemcpyHostToDevice));

	Solution h_solution;
	h_solution.allocation = d_allocation;
	h_solution.runtime = d_runtime;

	double* h_deltas;
	unsigned int* h_moves;

	//Allocate memory for deltas
	check_cudaError(cudaMalloc(&h_deltas, num_evaluate_threads * sizeof(double)));
	//Allocate memory for moves
	check_cudaError(cudaMalloc(&h_moves, num_evaluate_threads * sizeof(unsigned int)));

	evaluate_two_node_moves_kernel <<< evaluate_grid, evaluate_block >>>(h_env, h_solution, h_moves, h_deltas, num_nodes, num_moves_per_thread, num_moves, num_VMs);
	find_best_two_node_move_kernel<num_find_threads><<<find_grid, find_block>>>(h_solution, h_deltas, h_moves, num_nodes, num_evaluate_threads);

	unsigned int node_1_pos;
	unsigned int node_2_pos;

	// Copy best two moves to the CPU
	check_cudaError(cudaMemcpy(&node_1_pos, &h_moves[0], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	check_cudaError(cudaMemcpy(&node_2_pos, &h_moves[1], sizeof(unsigned int), cudaMemcpyDeviceToHost));

	solution = best_two_node_allocation(solution, node_1_pos, node_2_pos);

	cudaFree(h_deltas);
	cudaFree(h_moves);
	cudaFree(d_allocation);
	cudaFree(d_runtime);

	return solution.fitness < solution_old.fitness ? solution : solution_old;
}

Chromosome four_node_move_local_search(Chromosome solution_old, Data* data, Environment h_env, int* h_four_node_move_matrix) {
	Chromosome solution(solution_old);

	const unsigned int num_VMs = solution.vm_queue.size();
	const unsigned int num_nodes = solution.allocation.size() - 1;
	const unsigned int num_moves = four_node_num_moves;

	const unsigned int num_evaluate_threads = 2048;
	const dim3 evaluate_block(128);
	const dim3 evaluate_grid(((num_evaluate_threads) + evaluate_block.x-1) / evaluate_block.x);

	const unsigned int num_moves_per_thread = (four_node_num_moves + num_evaluate_threads-1)/num_evaluate_threads;
	const unsigned int num_find_threads = 1024;
	const dim3 find_block(num_find_threads);
	const dim3 find_grid(1);

	//Pointer to memory on the GPU
	int* d_allocation;
	double* d_runtime;

	//Allocate GPU memory for solution
	check_cudaError(cudaMalloc(&(d_allocation), solution.allocation.size() * sizeof(int)));
	check_cudaError(cudaMalloc(&(d_runtime), solution.runtime_vector.size() * sizeof(double)));

	//Copy solution to GPU
	check_cudaError(cudaMemcpy(d_allocation, &(solution.allocation[0]),solution.allocation.size() * sizeof(int), cudaMemcpyHostToDevice));
	check_cudaError(cudaMemcpy(d_runtime, &(solution.runtime_vector[0]),solution.runtime_vector.size() * sizeof(double), cudaMemcpyHostToDevice));

	Solution h_solution;
	h_solution.allocation = d_allocation;
	h_solution.runtime = d_runtime;

	double* h_deltas;
	unsigned int* h_moves;

	//Allocate memory for deltas
	check_cudaError(cudaMalloc(&h_deltas, num_evaluate_threads * sizeof(double)));
	//Allocate memory for moves
	check_cudaError(cudaMalloc(&h_moves, num_evaluate_threads * sizeof(unsigned int)));

	evaluate_four_node_moves_kernel <<< evaluate_grid, evaluate_block >>>(h_four_node_move_matrix, h_env, h_solution, h_moves, h_deltas, num_nodes, num_moves_per_thread, num_moves, num_VMs);
	find_best_four_node_move_kernel <num_find_threads> <<<find_grid, find_block>>>(h_four_node_move_matrix, h_solution, h_deltas, h_moves, num_nodes, num_evaluate_threads);

	unsigned int node_1_pos;
	unsigned int node_2_pos;
	unsigned int node_3_pos;
	unsigned int node_4_pos;

	// Copy best 4 movements to the CPU
	check_cudaError(cudaMemcpy(&node_1_pos, &h_moves[0], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	check_cudaError(cudaMemcpy(&node_2_pos, &h_moves[1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	check_cudaError(cudaMemcpy(&node_3_pos, &h_moves[2], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	check_cudaError(cudaMemcpy(&node_4_pos, &h_moves[3], sizeof(unsigned int), cudaMemcpyDeviceToHost));

	solution = best_four_node_allocation(solution, node_1_pos, node_2_pos, node_3_pos, node_4_pos);

	cudaFree(h_deltas);
	cudaFree(h_moves);
	cudaFree(d_allocation);
	cudaFree(d_runtime);

	return solution.fitness < solution_old.fitness ? solution : solution_old;
}

#endif /* CUDA_CODE_H_ */
