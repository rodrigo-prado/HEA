
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

using namespace std;





void check_cudaError(cudaError_t error){
	if(error != cudaSuccess){
		cout << "Error: " << cudaGetErrorString(error) << endl;
		exit(-1);
	}
}


// class CudaChromosome{

// public:
// 	int *allocation, *ordering;
// 	int size, task_size, vm_size;

// 	float fitness, prob;

// 	unsigned int seed;
	


// 	__host__ CudaChromosome(int *ordering, int size, int task_size, int vm_size, float prob = 0.4) {
// 		this->size = size;
// 		this->task_size = task_size;
// 		this->vm_size = vm_size;
// 		this->prob = prob;
// 		this->seed = seed;

// 		this->fitness = 0.0;

// 		allocation = (int*) malloc (sizeof(int) * this->size);
// 		this->ordering = ordering;

// 		this->encode();
// 	}

// 	__host__ CudaChromosome();




// 	__host__  void encode() {
//         //Encode allocation chromosome
//         for (int i = 0; i < this->size; i++)
//             allocation[i] = random() % this->vm_size;
//     }



// 	// __device__ __host__ CudaChromosome(const CudaChromosome & other) : 
// 	// 		allocation(other.allocation), ordering(other.ordering), size(other.size), 
// 	// 		task_size(other.task_size), vm_size(other.vm_size), fitness(other.fitness), prob(other.prob), seed(other.seed) {}


// 	__device__ __host__ ~CudaChromosome() {};


// 	__device__ __host__ void print(int threadId=0){
// 		printf("%d\n", threadId);
// 		for(int i = 0; i < this->size; i++)
// 			printf("%d ", allocation[i]);
// 		printf("\n");
// 		for(int i = 0; i < this->task_size; i++)
// 			printf("%d ", ordering[i]);
// 		printf("\n");
// 		printf("Fitness: %f\n", this->fitness);
// 	}

// 	// __device__ void encode(){

// 	// 	curandState_t state;

// 	// 	/* we have to initialize the state */
// 	// 	curand_init(this->seed, /* the seed controls the sequence of random values that are produced */
//  //              0, /* the sequence number is only important with multiple cores */
//  //              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
// 	// 	&state);
// 	// 	/* curand works like rand - except that it takes a state as a parameter */

// 	// 	for(int i = 0; i < this->size; i++)
// 	// 		allocation[i] = curand(&state) % this->vm_size;

// 	// }


// 	// __device__ void mutate( ){
// 	// 	curandState_t state;

// 	// 	/* we have to initialize the state */
// 	// 	curand_init(this->seed, /* the seed controls the sequence of random values that are produced */
//  //              0, /* the sequence number is only important with multiple cores */
//  //              0,  the offset is how much extra we advance in the sequence for each call, can be 0 
// 	// 	&state);
// 	// 	/* curand works like rand - except that it takes a state as a parameter */

// 	// 	for(int i = 0; i < this->size; i++)
// 	// 		if(((float) curand(&state) / (float) RAND_MAX) <= prob )
// 	// 			allocation[i] = curand(&state) % this->vm_size;
// 	// }


// 	// __device__ __host__ void compute_fitness() {

// 	// 	this->fitness = 0.0;
// 	// 	for(int i = 0; i < this->size; i++)
// 	// 		if(allocation[i] == 1)
// 	// 			this->fitness = this->fitness + 1;
// 	// }
// };




//  ================== KERNEL ============================================ //


#define CUDA_POPULATION_SIZE 50


// __global__ void genetic(CudaChromosome **Population, int *ordering, unsigned int seed, int size, int task_size, int vm_size, CudaChromosome *best){

// 	Population[threadIdx.x] = new CudaChromosome(ordering, size, task_size, vm_size, seed);

// 	Population[threadIdx.x]->encode();
// 	Population[threadIdx.x]->mutate();
// 	Population[threadIdx.x]->compute_fitness();
	
// 	__syncthreads();

// 	//get the best
// 	if(threadIdx.x == 0){

// 		CudaChromosome *best_local = Population[0];

// 		for(int i = 1; i < CUDA_POPULATION_SIZE; i++){

// 			if(Population[i]->fitness > best_local->fitness)
// 				best_local = Population[i];

// 		}

// 		best = new CudaChromosome(* best_local);
// 		best->size = 1;

// 		printf("BEST:\n");
// 		best->print();


// 	}

// 	__syncthreads();

// 	delete Population[threadIdx.x];

// 	// if(threadIdx.x == 0){
// 	// 	for(int i = 0; i < CUDA_POPULATION_SIZE; i++)
// 	// 		// Population[i]->print();
// 	// }
	

// }


// int cuda_call(Data *data, Chromosome *seed_chrom){

// 	int *ordering;
// 	// int *d_ordering;

	
// 	CudaChromosome *cuda_chrom;


// 	ordering = (int *) malloc (seed_chrom->ordering.size() * sizeof(int));
// 	for(int i = 0; i < seed_chrom->ordering.size(); i++){
// 		ordering[i] = seed_chrom->ordering[i];
// 	}

// 	CudaChromosome *c_population = new CudaChromosome[CUDA_POPULATION_SIZE];

// 	// for(int i = 0; i < CUDA_POPULATION_SIZE; i++)
// 	// 	c_population[i] = new CudaChromosome(ordering, data->size, data->task_size, data->vm_size);

// 	for(int i = 0; i < CUDA_POPULATION_SIZE; i++)
// 		c_population[i].print();


	
// 	// CudaChromosome *best = new CudaChromosome(ordering, data->size, data->task_size, data->vm_size, time(NULL));

// 	// for(int i = 0; i < data->size; i++)
// 	// 	best->allocation[i] = -1;

// 	// check_cudaError(cudaMalloc(&d_ordering, initial_solution.ordering.size() * sizeof(int)));	

// 	// check_cudaError(cudaMemcpy(d_ordering, ordering, initial_solution.ordering.size() * sizeof(int), cudaMemcpyHostToDevice));


// 	// CudaChromosome **Population;

// 	// check_cudaError( cudaMalloc(&Population, sizeof(CudaChromosome*) * CUDA_POPULATION_SIZE));


// 	// genetic<<<1, CUDA_POPULATION_SIZE>>>(Population, d_ordering, time(NULL), data->size, data->task_size, data->vm_size, d_best);


// 	// cudaMemcpy(&best, &d_best, sizeof(CudaChromosome), cudaMemcpyDeviceToHost);
// 	// cudaMemcpy(&(best->allocation), &(d_best->allocation), data->size * sizeof(int), cudaMemcpyDeviceToHost);
// 	// cudaMemcpy(&(best->ordering), &(d_best->ordering), data->task_size * sizeof(int), cudaMemcpyDeviceToHost);


// 	// cout << "HOST BEST: " << endl;
// 	// best->print();

// 	// check_cudaError(cudaFree(d_ordering));
// 	// check_cudaError(cudaFree(Population));
// 	// free(ordering);


// 	return 0;
// }



// class CudaClass{
// public:
// 	int* data;
// 	CudaClass(int x) {
// 		data = new int[1]; data[0] = x;
// 	}
// };

// __global__ void useClass(CudaClass *cudaClass){
// 	printf("%d\n", cudaClass->data[0]);
// 	cudaClass->data[0] = 8;
// };




// int cuda_call(){
// 	CudaClass c(1);
//     // create class storage on device and copy top level class
//     CudaClass *d_c;
//     cudaMalloc((void **)&d_c, sizeof(CudaClass));
//     cudaMemcpy(d_c, &c, sizeof(CudaClass), cudaMemcpyHostToDevice);
//     // make an allocated region on device for use by pointer in class
//     int *hostdata;
//     cudaMalloc((void **)&hostdata, sizeof(int));
//     cudaMemcpy(hostdata, c.data, sizeof(int), cudaMemcpyHostToDevice);
//     // copy pointer to allocated device storage to device class
//     cudaMemcpy(&(d_c->data), &hostdata, sizeof(int *), cudaMemcpyHostToDevice);
//     useClass<<<1,1>>>(d_c);
//     cudaDeviceSynchronize();

//     int *data;
//     data = (int *) malloc(sizeof(int));
//     cudaMemcpy(data, d_c->data, sizeof(int), cudaMemcpyDeviceToHost);

//     cout << data[0] << endl;

//     cudaFree(d_c);
//     return 0;
// }

#endif /* CUDA_CODE_H_ */

