#ifndef HIO_H
#define HIO_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "pwcdi.h"

#ifdef __cplusplus
extern "C" {
#endif

void hio(ssc_pwcdi_plan *workspace,
		 ssc_pwcdi_params *params,
		 int globalIteration,
		 int shrinkWrapSubiter,
		 int initialShrinkWrapSubiter,
		 int extra_constraint,
		 int extraConstraintIteration,
		 int initialExtraConstraintSubiter);

  
#ifdef __cplusplus
}
#endif
 
#endif //HIO_H
