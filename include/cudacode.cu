
#ifndef CUDA_CODE_H_
#define CUDA_CODE_H_


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "Data.h"
#include "Chromosome.h"


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>


#include "SimpleRNG.h"
#include "reference_solution.h"

using namespace std;


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

struct Solution {
    int * allocation;
    double * runtime;
};

struct Environment {
	double * base;
	double * slowdown;
	double * link;
	unsigned int * whatisit;
	int num_vms;
	int size;
};

/**
 * The cost of swapping node i with node j.
 */
__device__ double cost(const Environment env_, const Solution solution_, const Move & move, const unsigned int & num_nodes_) {
	int i = move.i;
	int j = move.j;

	int vmi = solution_.allocation[i];
	int vmj = solution_.allocation[j];

	double icost = solution_.runtime[i] + solution_.runtime[j]; 
	double cost =  icost;

	if(vmi != vmj){
		double fiti, fitj;
		if(env_.whatisit[i] == 1)
			fiti =  env_.base[i] *  env_.slowdown[vmj];
		else 
			fiti = env_.base[i] / env_.link[vmj];

		if(env_.whatisit[i] == 1)
			fitj =  env_.base[j] *  env_.slowdown[vmi];
		else 
			fitj = env_.base[j] / env_.link[vmi];

		cost = fiti + fitj;
	}
	printf("icost: %f  Cost: %f i: %d j: %d\n", icost, cost, i, j);
	return cost; // não houve mudança no fitness

}

/**
 * Apply the move
 */
__device__ void apply_move(Move& move) {
	unsigned int tmp = move.solution[move.i];
	move.solution[move.i] = move.solution[move.j];
	move.solution[move.j] = tmp;

	printf("Move applied: %d[%d] %d[%d]\n", move.solution[move.i], move.i, move.solution[move.j], move.j);
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

	double min_delta =  fitness;

	const unsigned int first_move = tid * num_moves_per_thread_;

	unsigned int best_move = first_move;

	for (int i = first_move; i < (first_move + num_moves_per_thread_); ++i) {
		if (i < num_moves) {
			Move move = generate_move(i, num_nodes_, solution_.allocation);
			double move_cost = cost(env_, solution_, move, num_nodes_);
			if (move_cost < min_delta) {
			 	min_delta = move_cost;
			 	best_move = i;
			}
		}
	}
	// printf("best_move: %d best delta: %f\n", best_move, max_delta);
	deltas_[tid] = min_delta;
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
	reduce_to_minimum<threads>(tid, deltas_shmem, moves_shmem);
	if (tid == 0) {
			Move move = generate_move(moves_shmem[0], num_nodes_, solution_.allocation);
			apply_move(move);
			deltas_[0] = deltas_shmem[0];
	 }
}



/**
* Local Search with 'Best Accept' strategy (steepest descent)
* @param solution:     Current solution, will be changed during local search
* @param neighborhood: Collection of moves == potential neighbors
*/
unsigned int local_search(Chromosome & solution, Data * data) {
	//Number of legal moves. 
	//We do not want to swap the first node, which means we have n-1 nodes we can swap
	//For the first node, we then have n-2 nodes to swap with, 
	//for the second node, we have n-3 nodes to swap with, etc.
	//and for the n-2'th node, we have 1 node to swap with.
	//which gives us sum_i=1^{n-2} i legal swaps,
	//or (n-2)(n-1)/2 legal swaps

	const unsigned int num_nodes = solution.allocation.size() - 1;
	const unsigned int num_moves = (num_nodes-2)*(num_nodes-1)/2;

	// const unsigned int num_evaluate_threads = 8192;
	const unsigned int num_evaluate_threads = 2048;
	const dim3 evaluate_block(128);
	const dim3 evaluate_grid((num_evaluate_threads + evaluate_block.x-1) / evaluate_block.x);
	

	const unsigned int num_moves_per_thread = (num_moves+num_evaluate_threads-1)/num_evaluate_threads;
	
	
	const unsigned int num_apply_threads = 512;
	const dim3 apply_block(num_apply_threads);
	const dim3 apply_grid(1);


	cout << "num_nodes: " << num_nodes << endl;
	cout << "num_moves: "  << num_moves << endl;
	cout << "num_evaluate_threads: " << num_evaluate_threads << endl;
	cout << "num_moves_per_thread: " << num_moves_per_thread << endl;
	cout << "num_apply_threads: " << num_apply_threads << endl;


	// cout << "Solution without local search: " << endl;
	// solution.print();

	//Pointer to memory on the GPU
	int *d_allocation;
	double *d_runtime;

	double *d_slowdown;
	double *d_base;
	double *d_link;
	unsigned int *d_whatisit;

	//Allocate GPU memory for environmet's information
    check_cudaError(cudaMalloc(&(d_slowdown), data->vm_slowdown.size() * sizeof(double)));
    check_cudaError(cudaMalloc(&(d_base), data->base.size() * sizeof(double)));
    check_cudaError(cudaMalloc(&(d_link), data->link.size() * sizeof(double)));
    check_cudaError(cudaMalloc(&(d_whatisit), data->whatisit.size() * sizeof(unsigned int)));



    // CPU -> GPU //Copy informations to GPU
    check_cudaError(cudaMemcpy(d_slowdown, &(data->vm_slowdown[0]),data->vm_slowdown.size() * sizeof(double), cudaMemcpyHostToDevice));
    check_cudaError(cudaMemcpy(d_base, &(data->base[0]),data->base.size() * sizeof(double), cudaMemcpyHostToDevice));
    check_cudaError(cudaMemcpy(d_link, &(data->link[0]),data->link.size() * sizeof(double), cudaMemcpyHostToDevice));
    check_cudaError(cudaMemcpy(d_whatisit, &(data->whatisit[0]),data->whatisit.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Environment h_env;

    h_env.slowdown = d_slowdown;
    h_env.base = d_base;
    h_env.whatisit = d_whatisit;
    h_env.link = d_link;


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


	double min_delta = 0.0;
	check_cudaError(cudaMemcpy(&min_delta, &h_deltas[0], sizeof(double), cudaMemcpyDeviceToHost));

	//Copy solution to CPU
	check_cudaError(cudaMemcpy(&(solution.allocation[0]), h_solution.allocation, solution.allocation.size() * sizeof(int), cudaMemcpyDeviceToHost));

	
	// cudaFree(h_deltas);
	// cudaFree(h_moves);

	// cudaFree(base_gpu);
	// cudaFree(vm_slowdown_gpu);
	// cudaFree(whatisit_gpu);

	check_cudaError(cudaDeviceReset());

	cout << "Print after local search: " << endl;
	cout << "max_delta: " << min_delta << endl;

	solution.computeFitness();
	solution.print();

	return 0;
}



#endif /* CUDA_CODE_H_ */

