#include "omp.h"
#include "dll_export.h"


#ifdef __cplusplus
extern "C" {
#endif
	DLL_EXPORT int initialise();
#ifdef __cplusplus
}
#endif


void threads_setup(int nThreads_requested, int *nThreads_current);