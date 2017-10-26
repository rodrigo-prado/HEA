/**
  * Copyright 2014, SINTEF ICT, Applied Mathematics.
  * Released under GPLv3
  */

class SimpleRNG {
public:
	SimpleRNG(const unsigned int seed) {
		srand(seed);
	}
	unsigned int operator()(unsigned int N) {
		double helper = ((double)RAND_MAX)+1;
		helper = ((double)N) * ( ((double)rand())/helper ); // helper in [0,N)
		return (unsigned int) helper;
	}
};