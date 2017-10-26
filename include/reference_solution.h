/**
  * Copyright 2014, SINTEF ICT, Applied Mathematics.
  * Released under GPLv3
  */

#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#define PROFILING_ITER 10

/**
  * Cost function that calculates the cost of the edge 
  * from node i to j (e.g., in travelling salesman: the distance 
  * between the nodes)
  */
inline float cost_of_edge(const unsigned int& i, const unsigned int& j, const std::vector<unsigned int>& solution, const std::vector<float> &city_coordinates) {
	unsigned int a = solution[i];
	unsigned int b = solution[j];

	float x = city_coordinates[2*a] - city_coordinates[2*b];
	float y = city_coordinates[2*a+1] - city_coordinates[2*b+1];
	return sqrtf(x*x + y*y);
}

/**
  * Returns cost of solution
  */
inline float solution_cost(const std::vector<unsigned int>& solution, const std::vector<float> &city_coordinates) {
	float total_cost = 0;
	for(unsigned int i = 0; i < solution.size()-1; ++i) {
		total_cost += cost_of_edge(i, i+1, solution, city_coordinates);
	}
	return total_cost;
}

/**
  * Prints given solution in copyable format
  */
inline void print_solution(const std::vector<unsigned int>& solution)
{
	for(unsigned int i = 0; i < solution.size()-1; ++i) {
		if (i % 25 == 0)
			std::cout << std::endl;
		std::cout << solution[i] << ", ";
	}
	std::cout << std::endl;
}

/**
  * Checks whether solution equals reference solution.
  * If not, prints new solution.
  */
inline void check_results(const std::vector<unsigned int>& solution, const unsigned int max_iterations, 
	const bool printSolutionOnError=true, const bool cc_message=true, const bool storeSolutionIfReferenceNotValid = false) {

	/**
	  * choose correct reference solution
	  */
	size_t num_nodes = solution.size()-1;
	std::stringstream filename; 
	filename << "reference_solution_" << num_nodes << "_" << max_iterations;

	/**
	  * read reference solution
	  */
	std::vector<unsigned int> reference_solution;
	std::ifstream input(filename.str().c_str());  
	if (input.is_open()) {
		reference_solution.resize(num_nodes);
		for(size_t i = 0; i < num_nodes; ++i)
		{
			if (!input.good()) {
				reference_solution.resize(0);
				break;
			}
			input >> reference_solution[i];
		}

		input.close();
	}

	if (reference_solution.size() == 0 && !storeSolutionIfReferenceNotValid) {
		std::cout << "No reference solution found!" << std::endl;
		if (printSolutionOnError)
			print_solution(solution);
		return;
	}

	/**
	  * check for equality
	  */
	if (reference_solution.size() > 0)
	{
		size_t num_errors = 0;
		for(size_t i = 0; i < num_nodes; ++i) {
			if (solution[i] - reference_solution[i] != 0) {
				if (printSolutionOnError)
					std::cout << "Error at position " << i << ":" << solution[i] << "!=" << reference_solution[i] << std::endl;
				++num_errors;
			}
		}
		if (num_errors) {
			std::cout << "Solution does not equal reference solution from CPU!";
			if (cc_message)
				std::cout << " Did you compile with 2.0 compute capability?"
				//<< "  (-ftz=false -prec-div=true -prec-sqrt=true)"
				;
			std::cout << std::endl;
			if (printSolutionOnError)
				print_solution(solution);
		}
	}
	/**
	  * store solution as reference
	  */
	else {
		std::ofstream output(filename.str().c_str());
		if (!output.is_open()) {
			std::cout << "Could not open reference file '" << filename.str().c_str() << "' for storing reference solution" << std::endl;
		} else {
			std::cout << "Storing reference solution into '" << filename.str().c_str() << "'" << std::endl;
			for(size_t i = 0; i < num_nodes; ++i) {
				output << " " << solution[i];
			}
			output.close();
		}
	}

}