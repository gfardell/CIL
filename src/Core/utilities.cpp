#include "utilities.h"
#include "iostream"

#ifdef USE_IPP

#include "ippcore.h"
int initialise(void)
{
	//must be before other ipp calls, determines the processor and optimisation options
	ippInit();
	return 0;
}


#endif // USE_IPP

#ifndef USE_IPP

void initialise(void)
{}
	return 0;
}

#endif // !USE_IPP



void threads_setup(int nThreads_requested, int *nThreads_current)
{
#pragma omp parallel
	{
		if (omp_get_thread_num() == 0)
		{
			*nThreads_current = omp_get_num_threads();
		}
	}
	omp_set_num_threads(nThreads_requested);
}
