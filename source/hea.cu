#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include "HEFT.h"
#include "MinMin.h"
#include "Data.h"
#include "Chromosome.h"
#include "cudacode.cu"
#include <omp.h>
#include <random>
#include <chrono>
#include <time.h>
#include <math.h>

#define FOUR_OPT_MOVES 4

using namespace TCLAP;
using namespace std;

int n_threads = 16;

int four_node_num_moves = 0;

int n_chromosomes = 50;
int n_elite_set = n_chromosomes / 2;

int omp_iter_max = ceil((n_chromosomes * 0.6) / n_threads);

int** OMP_matrix;
unsigned int OMP_matrix_row_pos = 0;

int* h_four_node_move_matrix;
int** four_node_move_matrix;
int four_node_move_matrix_cols;
unsigned int four_node_move_matrix_rows;

double run_time2 = 0;
int tempo_ls = 0;
int tempo_do_next_pop = 0;

typedef vector<Chromosome> vect_chrom_type;

struct Settings_struct{
	int num_chromosomes;        // Number of chromosomes
	int num_generations;        // Number of generations
  int num_elite_set;          // Max size of Elite-set

  float mutation_probability; // Probability of mutation (for each gene)
	float elitism_rate;  // Rate of generated solutions
  int howMany_elistism;
	float alpha; // percentage of chromosomes send to local search procedures
	float localSearch_probability; // Probability of local search

	float time_limit;                 // Run time limite in seconds
  int print_gen;                   // What generation will be printed
	bool verbose, start_heuristic;  // verbose = print type, start_heuristic = Initial Population type

	long seed;      // Random Seed

  double delta = 0.0; // acceptance criteria for Elite-Set (based on distance)

    Settings_struct() {
        // Default Settings
        //num_chromosomes = 50;
        num_chromosomes = n_chromosomes;

        //num_generations = 500;
        num_generations = 100;
        num_elite_set = n_elite_set;

        mutation_probability = 0.10;
        elitism_rate = 0.10;
        alpha = 0.3; // Local Search Rate

        localSearch_probability = 0.3;
        time_limit = 7200; // time limit 30 minutes

        print_gen = 10;

        verbose = false;
        start_heuristic = true;

        howMany_elistism =  (int) ceil(num_chromosomes * elitism_rate);
    }
};

Settings_struct *setting;

/*  Call HEFT */
Chromosome HEFT(Data *data) {
    //orders
    event_map orders;

    //building and ordering the seqOfTasks
    vector<int> seqOftasks;
    boost::copy(data->task_map | boost::adaptors::map_keys, std::back_inserter(seqOftasks));

    vector<double> ranku_vet(seqOftasks.size(), 0.0);
    vector<double> ranku_aux(seqOftasks.size(), -1);

    for_each(seqOftasks.begin(), seqOftasks.end(), [&](const int &idA) {
        ranku_vet[idA] = ranku(idA, data, ranku_aux);
    });

    sort(seqOftasks.begin(), seqOftasks.end(), [&](const int &idA, const int &idB) {
        return ranku_vet[idA] < ranku_vet[idB];
    });

    //get all vm keys
    vector<int> vm_keys;
    boost::copy(data->vm_map | boost::adaptors::map_keys, std::back_inserter(vm_keys));

    //build orders struct (event_map)
    for (auto vm_key : vm_keys)
        orders.insert(make_pair(vm_key, vector<Event>()));


    vector<int> taskOn(data->task_size, -1);
    vector<double> end_time(data->task_size, 0);
    for (auto id_task = seqOftasks.rbegin(); id_task != seqOftasks.rend(); ++id_task) { // reverse vector
        allocate(*id_task, taskOn, vm_keys, orders, end_time, data);
    }


    // == build chromosome == //

    Chromosome heft_chr(data);

    // build allocation
    for (auto info : orders) {
        auto id_vm = info.first;
        for (auto event : info.second) {
            auto task = data->task_map.find(event.id)->second;
            heft_chr.allocation[task.id] = id_vm;
            // update output files;
            for (auto out : task.output)
                heft_chr.allocation[out] = id_vm;
        }
    }


    // build ordering
    heft_chr.ordering.clear();
    // add root
    heft_chr.ordering.push_back(data->id_root);
    int task_id = -1;
    do {
        task_id = get_next_task(orders);
        if (task_id != -1 && task_id != data->id_root && task_id != data->id_sink)
            heft_chr.ordering.push_back(task_id);
    } while (task_id != -1);
    // add sink
    heft_chr.ordering.push_back(data->id_sink);

    heft_chr.computeFitness(true, true);

    return heft_chr;
}

/* Call MinMin */
Chromosome minMinHeuristic(Data* data) {
    list<int> task_list;
    // start task list
    for (auto info : data->task_map)
        task_list.push_back(info.second.id);
    task_list.sort([&](const int &a, const int &b) { return data->height[a] < data->height[b]; });

    list<int> avail_tasks;

    vector<double> ft_vector(data->size, 0);
    vector<double> queue(data->vm_size, 0);
    vector<int> file_place(data->size, 0);
    list<int> task_ordering(0);


    //the task_list is sorted by the height(t). While task_list is not empty do
    while (!task_list.empty()) {
        auto task = task_list.front();//get the first task
        avail_tasks.clear();
        while (!task_list.empty() && data->height[task] == data->height[task_list.front()]) {
            //build list of ready tasks, that is the tasks which the predecessor was finish
            avail_tasks.push_back(task_list.front());
            task_list.pop_front();
        }

        schedule(data, avail_tasks, ft_vector, queue, file_place, task_ordering);//Schedule the ready tasks
    }

    Chromosome minMin_chrom(data);

    for (int i = 0; i < data->size; ++i)
        minMin_chrom.allocation[i] = file_place[i];
    minMin_chrom.ordering.clear();

    minMin_chrom.ordering.insert(minMin_chrom.ordering.end(), task_ordering.begin(), task_ordering.end());
    minMin_chrom.computeFitness(true, true);

    //if(setting->verbose)
    //    cout << "MinMIn fitness: " << minMin_chrom.fitness << endl;

    return minMin_chrom;
}


// ========== CUDA Functions ============ //

Environment load_cuda(Data* data) {

    double* d_slowdown;
    double* d_base;
    double* d_link;
    unsigned int* d_whatisit;

		cudaSetDevice(0);
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

    return h_env;
}

int* load_GPU_matrix(int** matrix, unsigned int rows, int cols) {

    int* h_matrix;
    int* d_matrix;

		cudaSetDevice(0);

    check_cudaError(cudaMalloc((void**)&d_matrix, rows * cols * sizeof(int)));
    check_cudaError(cudaMemcpy(d_matrix, matrix[0], rows * cols * sizeof(int), cudaMemcpyHostToDevice));

    h_matrix = d_matrix;

    return h_matrix;
}

// ========== Path Relinking ============ //
Chromosome pathRelinking(vect_chrom_type Elite_set, const Chromosome& dest, Data* data){

    Chromosome best(dest);

    // For each chromosome on Elite Set, do:
    for(unsigned i = 0; i < Elite_set.size(); ++i){
        auto src = Elite_set[i];

        // Copy ordering from dest chromosome
        src.ordering.clear();
        src.ordering.insert(src.ordering.end(), dest.ordering.begin(), dest.ordering.end());

        for(int el = 0; el < data->size; ++el){
            if(src.allocation[el] != dest.allocation[el]){
                src.allocation[el] = dest.allocation[el];
                src.computeFitness(true, true);
                if(best.fitness > src.fitness)
                    best = src;
            }
        }
    }

    return best;
}

// Get the best chromosome
inline int getBest(vect_chrom_type Population){
	auto best = Population[0];
	auto pos = 0;
	for(int i = 0; i < setting->num_chromosomes; ++i)
		if(best.fitness > Population[i].fitness){
			best = Population[i];
			pos = i;
		}
	return pos;
}

// Tournament Selection
inline int tournamentSelection(vect_chrom_type Population){

	int a = random() % Population.size();
	int b = random() % Population.size();

	while (b == a)
		b = random() % Population.size();

	return Population[a].fitness < Population[b].fitness ? a : b;
}


// =========== Local search functions  ========= //

// N1 - Swap-vm
inline Chromosome localSearchN1(const Data* data, Chromosome ch) {
    Chromosome old_ch(ch);

    for (int i = 0; i < data->size; ++i) {
        for (int j = i + 1; j < data->size; ++j) {
            if (ch.allocation[i] != ch.allocation[j]) {
                //do the swap
                iter_swap(ch.allocation.begin() + i, ch.allocation.begin() + j);
                ch.computeFitness();
                if (ch.fitness < old_ch.fitness) {
                    return ch;
                }
                //return elements
                iter_swap(ch.allocation.begin() + i, ch.allocation.begin() + j);
            }
        }
    }
    return old_ch;
}

// N2 - Swap position
inline Chromosome localSearchN2(const Data* data, Chromosome ch) {
		Chromosome old_ch(ch);
    // for each task, do
    for (int i = 0; i < data->task_size; ++i) {
        auto task_i = ch.ordering[i];
        for (int j = i + 1; j < data->task_size; ++j) {
            auto task_j = ch.ordering[j];
            if (ch.height_soft[task_i] == ch.height_soft[task_j]) {
                //do the swap
                iter_swap(ch.ordering.begin() + i, ch.ordering.begin() + j);
                ch.computeFitness(false, true);
                if (ch.fitness < old_ch.fitness) {
                    return ch;
                }
                //return elements
                iter_swap(ch.ordering.begin() + i, ch.ordering.begin() + j);
            } else
                break;
        }
    }
    return old_ch;
}

// N3 = Move-1 Element
inline Chromosome localSearchN3(const Data* data, Chromosome ch) {
    Chromosome old_ch(ch);

    for (int i = 0; i < data->size; ++i) {
        int old_vm = ch.allocation[i];
        // for (int j = 0; j < data->vm_size; ++j) {
				for (int j = 0; j < ch.vm_queue.size(); ++j) {
            if (old_vm != j) {
                ch.allocation[i] = j;
                ch.computeFitness();
                if (ch.fitness < old_ch.fitness) {
                    return ch;
                }
            }
        }
        ch.allocation[i] = old_vm;
    }
    return old_ch;
}

// N4 = move task - reorganize files
inline Chromosome localSearchN4(const Data* data, Chromosome ch){

    // for each task, do:
		// for(int i = 0; i < data->task_size; ++i){
    for(int i = 0; i < data->task_size; ++i){
        if(i != data->id_sink && i != data->id_root) {
            Chromosome aux_ch(ch);
            auto task = data->task_map.find(i)->second;

            int old_vm = ch.allocation[task.id];

						// for (int j = 0; j < data->vm_size; ++j) {
            for (int j = 0; j < data->vm_size; ++j) {
                if( j != old_vm) {
                    aux_ch.allocation[task.id] = j;
                    for (auto out_file : task.output)
                        aux_ch.allocation[out_file] = j;
                    aux_ch.computeFitness(true, true);
                    if (aux_ch.fitness < ch.fitness) {
                        return aux_ch;
                    }
                }
            }
        }
    }
    return ch;
}

// N5 File best local place.
inline Chromosome localSearchN5(const Data* data, Chromosome chr){
    Chromosome best_chr(chr);

    for(int i = data->task_size; i < data->size; ++i){//for each dynamic file, do:
        //get file
        auto file = data->file_map.find(i)->second;

        vector<int> aux_vet(data->vm_size, -1);


        for(auto task_id : file.all_tasks){//for each task
            auto vm_id = chr.allocation[task_id];
            if(aux_vet[vm_id] == -1){//vm not checked
                aux_vet[vm_id] = 0;
                chr.allocation[file.id] = vm_id;
                chr.computeFitness(true, true);
                if(best_chr.fitness > chr.fitness)
                     //return best_chr = chr;
                    return chr;
            }
        }
    }
    return best_chr;
}

// N6 MOVE Vm Queue Objects.
inline Chromosome localSearchN6(const Data* data, Chromosome chr){
    uniform_int_distribution<> dis (0, data->vm_size-1);
    Chromosome best(chr);
    int vm1, vm2;

    vm1 = dis(engine_chr);
     do { vm2 = dis(engine_chr); }while(vm1 == vm2);

    auto f = chr.vm_queue.find(vm1);
    if(f != chr.vm_queue.end()){
        vector<int> aux(f->second);
        for (auto i : aux) {//for each element, do:
            chr.allocation[i] = vm2;
            chr.computeFitness(true, true);
            if (best.fitness > chr.fitness)
                //best = chr;
                return chr;
        }
    }
    return best;
}


// ========== Main Functions ========== //

inline void doNextPopulation(vect_chrom_type & Population){
	//int how_many =  (int) ceil(setting->num_chromosomes * setting->elitism_rate);

	vector<Chromosome> children_pool;

	// === do offsprings === //
	for(int i = 0; i < ceil(setting->num_chromosomes/2.0); ++i){

     	// select our two parents with tournament Selection
		int posA, posB;
		posA = tournamentSelection(Population);
		do{ posB = tournamentSelection(Population);	}while(posA == posB);

        // get the parents
		auto parentA = Population[posA];
		auto parentB = Population[posB];

		// cross their genes
		auto child = parentA.crossover(parentB);
		// mutate the child
		child.mutate(setting->mutation_probability);

		// recompute fitness
    child.computeFitness();

    // Add solution on children_pool
		children_pool.push_back(child);

	}

	// === update population === //

  // add all solutions to the children_pool
  children_pool.insert(children_pool.end(), Population.begin(), Population.end());

  // Delete old population
  Population.clear();

  // Elitisme operator - the best is always on the population
  //auto posBest = getBest(children_pool);
  //Population.push_back(children_pool[posBest]);
  sort(children_pool.begin(), children_pool.end(), [&](const Chromosome & chr1, const Chromosome & chr2){
      return chr1.fitness < chr2.fitness;
  });

    for(int i = 0; i < setting->howMany_elistism; ++i){
        Population.push_back(children_pool[0]);
        children_pool.erase(children_pool.begin());
    }

    // Selected the solutions to build the new population
    while(Population.size() < static_cast<unsigned int>(setting->num_chromosomes)){
        auto pos = tournamentSelection(children_pool);
        //auto pos = random() % children_pool.size();
        Population.push_back(Chromosome(children_pool[pos]));
        children_pool.erase(children_pool.begin() + pos);
    }
    random_shuffle(Population.begin(), Population.end());
}

// Call all the Local Search Functions
inline vect_chrom_type localSearch(vect_chrom_type& Population, Data* data, Environment& h_env, int* h_four_node_move_matrix) {
		std::chrono::steady_clock::time_point begin_ls = std::chrono::steady_clock::now();

		int tid;
		int ch_pos;

		for (int omp_iter = 0; omp_iter < omp_iter_max; ++omp_iter) {
			#pragma omp parallel private(tid, ch_pos) num_threads(n_threads)
			{
				tid = omp_get_thread_num();
				ch_pos = OMP_matrix[OMP_matrix_row_pos][tid];

	      Population[ch_pos] = local_search(Population[ch_pos], data, h_env);
	      Population[ch_pos] = one_node_move_local_search(Population[ch_pos], data, h_env);
			  Population[ch_pos] = two_node_move_local_search(Population[ch_pos], data, h_env);
				if (tid % 2 == 0)
  	    	Population[ch_pos] = four_node_move_local_search(Population[ch_pos], data, h_env, h_four_node_move_matrix);
				Population[ch_pos] = localSearchN3(data, Population[ch_pos]);
				Population[ch_pos] = localSearchN5(data, Population[ch_pos]);
			}
			OMP_matrix_row_pos += 1;
		}

		std::chrono::steady_clock::time_point end_ls = std::chrono::steady_clock::now();
		tempo_ls += std::chrono::duration_cast<std::chrono::milliseconds>(end_ls - begin_ls).count();

		return Population;
}


Chromosome run(string name_workflow, string name_cluster) {

		Data* data = new Data(name_workflow, name_cluster);

		// CUDA - Load data on the GPU
    Environment h_env = load_cuda(data);
    h_four_node_move_matrix = load_GPU_matrix(four_node_move_matrix, four_node_move_matrix_rows, four_node_move_matrix_cols);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    vector<Chromosome> Population;
    vector<Chromosome> Elite_set;

    // Set Delta
    setting->delta = data->size / 4.0;
    // check distance
    auto check_distance = [&](Chromosome  chr, const vector<Chromosome>& Set){
        for(auto set_ch : Set) {
            if (chr.getDistance(set_ch) < setting->delta) {
                return false;
            }

        }
       return true;
    };

	// == Start initial population == //
    Chromosome minminChr(minMinHeuristic(data));
    Chromosome heftChr (HEFT(data));

    Population.push_back(minminChr);
    Population.push_back(heftChr);

    double mut = 0.5;

    for(int i = 0; i < ceil(setting->num_chromosomes*0.8); ++i) {
        Chromosome chr1(minminChr);
        chr1.mutate(mut);
        chr1.computeFitness();

        Chromosome chr2(heftChr);
        chr2.mutate(mut);
        chr2.computeFitness();

        Population.push_back(chr1);
        Population.push_back(chr2);
        mut += 0.05;
    }

    // 10% of random solutions
    for(int i = 0; i < (setting->num_chromosomes*0.10); ++i) {
        Population.push_back(Chromosome(data));
    }

    Chromosome best(Population[getBest(Population)]);

    // Do generation
	  int i = 0;
    // start stop clock

	  while(i < setting->num_generations ) {
        // Do local Search ?
        float doit = (float) random() / (float) RAND_MAX;
        if (doit <= (setting->localSearch_probability))
					Population = localSearch(Population, data, h_env, h_four_node_move_matrix);

        // Update best
        auto pos = getBest(Population);

        if (best.fitness > Population[pos].fitness) {

            best = Population[pos];

            // Apply path Relinking
            if (!Elite_set.empty())
                best = pathRelinking(Elite_set, best, data);

            // Update Elite-set
            if(check_distance(best, Elite_set))
                Elite_set.push_back(best);  // Push all best' solutions on Elite-set

            // check elite set size
            if (Elite_set.size() > static_cast<unsigned int>(setting->num_elite_set))
                Elite_set.erase(Elite_set.begin());

            // Apply Local Search
						best = local_search(best, data, h_env);
						best = one_node_move_local_search(best, data, h_env);
						best = two_node_move_local_search(best, data, h_env);
            best = four_node_move_local_search(best, data, h_env, h_four_node_move_matrix);
            best = localSearchN3(data, best);
            best = localSearchN4(data, best);
						best = localSearchN5(data, best);
						best = localSearchN6(data, best);

            Population[pos] = best;
            i = 0;
        }

				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				run_time2 = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
			  if (setting->verbose && (i % setting->print_gen) == 0)
            cout << "Gen: " << i << " Fitness: " << best.fitness / 60.0 << " run time(min): "
                 << run_time2 / 60.0 << " Tempo LS = "<<tempo_ls/1000<<endl;

        i += 1;

    }

		cudaDeviceReset();

		// return the global best
		return best;
}

// Read command line parameters (input files)
void setupCmd(int argc, char** argv, string & name_workflow, string & name_cluster) {
	try {
			// Define the command line object.
			CmdLine cmd("Hybrid Evolutionary Algorithm", ' ', "1.0");

			// Define a value argument and add it to the command line.
			ValueArg<string> arg1("w", "workflow", "Name of workflow file", true,"file", "string");
			cmd.add(arg1);
			ValueArg<string> arg2("c", "cluster", "Name of virtual cluster file", true, "file", "string");
			cmd.add(arg2);
	    SwitchArg verbose_arg("v", "verbose", "Output info", cmd, false);

			// Parse the args.
			cmd.parse(argc, argv);

			// Get the value parsed by each arg.
			name_workflow = arg1.getValue();
			name_cluster = arg2.getValue();
			setting->verbose = verbose_arg.getValue();

    } catch(ArgException &e) {  // catch any exceptions
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		}
}

int n_choose_k(int n, int k)
{
  if (k > n) return 0;
  if (k * 2 > n) k = n-k;
  if (k == n) return 1;

  int result = n;
  for (int i = 2; i <= k; ++i) {
    result *= (n-i+1);
    result /= i;
  }
  return result;
}

void generate_shuffled_matrix(int** matrix, int matrix_cols, int row_size, int row_start_index, int chunk_size)
{
  random_device rd;
  mt19937 g(rd());

  srand(time(NULL));
  for (int i = row_start_index; i <= chunk_size + row_start_index; ++i) {
    int start = 0 + rand() % row_size + 1;
    if (start == row_size) {
      int cont = 0;
      for (int j = start - matrix_cols; j < row_size; ++j) {
        matrix[i][cont] = j;
        cont+=1;
      }
    	shuffle(&matrix[i][0], &matrix[i][matrix_cols], g);
	  } else {
        int cnt_max = matrix_cols;
        int cnt = 0;
        int j = 0;
        while (cnt < cnt_max) {
          matrix[i][j] = start + cnt;
          cnt += 1;
          j += 1;
          if (start + cnt == row_size)
            start = 0 - cnt;
        }
  	    shuffle(&matrix[i][0], &matrix[i][matrix_cols], g);
    }
  }
}

int main(int argc, char** argv) {
	std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();

	string name_workflow, name_cluster;

	setting = new Settings_struct();
	setupCmd(argc, argv, name_workflow, name_cluster);

	Data* temp_data = new Data(name_workflow, name_cluster);
	int num_nodes = temp_data->size;
	four_node_move_matrix_cols = num_nodes;
  four_node_move_matrix_rows = n_choose_k(num_nodes, FOUR_OPT_MOVES);
  unsigned int max_num_moves = n_threads * 50000;

  if (four_node_move_matrix_rows > max_num_moves) {
    four_node_move_matrix_rows = max_num_moves;
  } else {
    four_node_move_matrix_rows = floor(four_node_move_matrix_rows/n_threads) * n_threads;
 }

  four_node_num_moves = four_node_move_matrix_rows;

  four_node_move_matrix = new int* [four_node_move_matrix_rows];
  four_node_move_matrix[0] = new int [four_node_move_matrix_rows * four_node_move_matrix_cols];

  for (int i = 1; i < four_node_move_matrix_rows; ++i)
    four_node_move_matrix[i] = four_node_move_matrix[i-1] + four_node_move_matrix_cols;

	int OMP_matrix_cols = n_chromosomes;
	int OMP_matrix_rows = 50000;

  OMP_matrix = new int* [OMP_matrix_rows];
  OMP_matrix[0] = new int [OMP_matrix_rows * OMP_matrix_cols];

  for (int i = 1; i < OMP_matrix_rows; ++i)
    OMP_matrix[i] = OMP_matrix[i-1] + OMP_matrix_cols;

  int thread_work_chunk_size = four_node_move_matrix_rows/n_threads;
  int tid;
  int row_start_index;

	#pragma omp parallel private(tid, row_start_index) num_threads(n_threads)
	{
		tid = omp_get_thread_num();

		if (tid == 0) {
			row_start_index = 0;
		} else {
			row_start_index = (thread_work_chunk_size * tid) - 1;
		}
		generate_shuffled_matrix(four_node_move_matrix, four_node_move_matrix_cols, num_nodes, row_start_index, thread_work_chunk_size);
	}

	generate_shuffled_matrix(OMP_matrix, OMP_matrix_cols, n_chromosomes, 0, OMP_matrix_rows - 1);

	auto best = run(name_workflow, name_cluster);
	best.computeFitness(true, true);

	std::chrono::steady_clock::time_point end0= std::chrono::steady_clock::now();
	double elapseSecs = std::chrono::duration_cast<std::chrono::seconds>(end0 - begin0).count();

	if(setting->verbose) {
		cout << "\t **** HEA **** " << endl;
		// best.print();
	}

	cout << best.fitness / 60.0 <<  " " <<  "   total RunTime: " << elapseSecs << endl;

	delete [] OMP_matrix;
	delete [] four_node_move_matrix;
	//delete setting struct
	delete [] setting;
	return 0;
}
