
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



/**
  * The cost of swapping node i with node j. 
  */
__device__ double cost(const Move& move, const unsigned int & num_nodes_) {
	const unsigned int i = move.i;
	const unsigned int j = move.j;
	const int* ptr = move.solution;


    double cost = 0.0;

	if(ptr[i] == 0 && ptr[j] != 0)  return -1;
	if(ptr[i] ==  0 && ptr[j] == 0) return 1.0;
	if(ptr[i] != 0 && ptr[j] == 0 ) return 2.0;

}

/**
  * Apply the move
  */
__device__ void apply_move(Move& move) {
	unsigned int tmp = move.solution[move.i];
	move.solution[move.i] = move.solution[move.j];
	move.solution[move.j] = tmp;

	printf("%d %d move\n", move.i, move.j);
}

/**
  * @param move_number Move number to generate
  * @param num_nodes_ number of nodes
  */
__device__ Move generate_move(const unsigned int& move_number_, const unsigned int & num_nodes_, int * solution_) {
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
__global__ void evaluate_moves_kernel(int * solution_, unsigned int *moves_, double * deltas_, const double  fitness, unsigned int num_nodes_, unsigned int num_moves_per_thread_) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	printf("%d\n", tid);

	const unsigned int num_moves = (static_cast<int>(num_nodes_)-2) * (static_cast<int>(num_nodes_)-1)/2;

	float max_delta = -2.0;

	const unsigned int first_move = tid * num_moves_per_thread_;
	
	unsigned int best_move = first_move;

	for (int i = first_move; i < (first_move + num_moves_per_thread_); ++i) {
		if (i < num_moves) {
			Move move = generate_move(i, num_nodes_, solution_);
			double move_cost = cost(move, num_nodes_);
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
__global__ void apply_best_move_kernel(int* solution_, const double fitness, double* deltas_, unsigned int* moves_, unsigned int num_nodes_, unsigned int deltas_size_) {
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
			Move move = generate_move(moves_shmem[0], num_nodes_, solution_);
			apply_move(move);
			deltas_[0] = deltas_shmem[0];
	 }
}

__global__ void test(int * dependency_, const int unsigned size_){

	for(int i = 0; i < size_ ; i++){
		for(int j = 0; j < size_; j++)
			dependency_[i * size_ + j] += 1;
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

	// solution.print();

	//Pointer to memory on the GPU 
	int * solution_gpu;
    double * deltas_gpu;
    unsigned int * moves_gpu;

    /* problem structures */
    double * base_gpu;
    double * vm_slowdown_gpu;
    unsigned int * whatisit_gpu;
    int * dependency_gpu;
    unsigned int static_vm = data->static_vm;

   

	//Allocate GPU memory for solution
    check_cudaError(cudaMalloc(&solution_gpu, 10 * sizeof(int)));
     cout << "solution" << endl;
	//Copy solution to GPU
	check_cudaError(cudaMemcpy(solution_gpu, &(solution.allocation[0]), solution.allocation.size() * sizeof(int), cudaMemcpyHostToDevice));

	test <<< 1, 1 >>>(solution_gpu,  solution.allocation.size());

	cudaFree(solution_gpu);

 //    //Allocate memory for deltas
 //    check_cudaError(cudaMalloc(&deltas_gpu, num_evaluate_threads * sizeof(double)));

 //    //Allocate memory for moves
 //    check_cudaError(cudaMalloc(&moves_gpu, num_evaluate_threads * sizeof(unsigned int)));

 //    solution.fitness = 0.0;
 //    for(int i = 0; i < solution.allocation.size(); i++)
 //    	if(solution.allocation[i] == 0) solution.fitness += 1;


	// //Loop through all possible moves and find best (steepest descent)
	// evaluate_moves_kernel <<< evaluate_grid, evaluate_block >>>(solution_gpu, moves_gpu, deltas_gpu, solution.fitness, num_nodes, num_moves_per_thread);
	// apply_best_move_kernel<num_apply_threads><<<apply_grid, apply_block>>>(solution_gpu, solution.fitness, deltas_gpu, moves_gpu, num_nodes, num_evaluate_threads);
	
	
	// double max_delta = 0.0; 
	// check_cudaError(cudaMemcpy(&max_delta, &deltas_gpu[0], sizeof(double), cudaMemcpyDeviceToHost));

	// //Copy solution to CPU
	// check_cudaError(cudaMemcpy(&(solution.allocation[0]), solution_gpu, solution.allocation.size() * sizeof(int), cudaMemcpyDeviceToHost));

	// cudaFree(solution_gpu);
	// cudaFree(deltas_gpu);
	// cudaFree(moves_gpu);


	// check_cudaError(cudaDeviceReset());

	// cout << "Print after: " << endl;
	// cout << "max_delta: " << max_delta << endl;
    
 // 	solution.computeFitness();
	

	return 0;
}



//int gpu_test( ) {
//
//	//Number of nodes in our solution
//	unsigned int num_nodes = 4;
//	// Maximal number of iterations performed in local search
//	unsigned int max_iterations = 5000;
//
//	// choose number of nodes and maximal number of iterations according to command line argument
//
//
//	// info output
//	std::cout << "problem size:       " << num_nodes << " nodes" << std::endl;
//	std::cout << "maximal iterations: " << max_iterations << std::endl;
//
//
//	// generate random coordinates
//	unsigned int seed = 12345;
//	srand(seed);
//	std::vector<float> city_coordinates(2*num_nodes);
//	for(unsigned int i = 0; i < num_nodes; ++i)
//	{
//		city_coordinates[2*i] = ((float)rand())/((float)RAND_MAX);
//		city_coordinates[2*i+1] = ((float)rand())/((float)RAND_MAX);
//	}
//
//	//Generate our circular solution vector (last node equal first)
//	std::vector<unsigned int> gpu_solution(num_nodes+1);
//	for(unsigned int i = 0; i < num_nodes; ++i) {
//		gpu_solution[i] = i;
//	}
//
//	SimpleRNG rng(54321);
//	std::random_shuffle(gpu_solution.begin()+1, gpu_solution.end()-1, rng);
//	//Dummy node for more readable code
//	gpu_solution[num_nodes] = gpu_solution[0];
//
//
//	unsigned int gpu_num_iterations = local_search(gpu_solution, city_coordinates, max_iterations);
//
//	cout << "solution: " << endl;
//	for(auto i : gpu_solution)
//		cout << i << ", ";
//	cout << endl;
//
//	float gpu_cost = solution_cost(gpu_solution, city_coordinates);
//	std::cout << "GPU completed " << gpu_num_iterations << " iterations in " << std::endl;
//	std::cout << " Solution has a cost of " << gpu_cost << std::endl;
//	//check_results(gpu_solution, max_iterations, false);
//
//	return 0;
//}



#endif /* CUDA_CODE_H_ */

