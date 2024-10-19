#ifndef ER_H
#define ER_H

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

void er(ssc_pwcdi_plan *workspace,
        ssc_pwcdi_params *params,
        int global_iteration,
        int shrinkwrap_subiter,
        int initial_shrinkwrap_subiter,
        int extra_constraint,
        int extra_constraint_subiter,
        int initial_extra_constraint_subiter,
        float shrinkwrap_threshold,                      // novos parametros 
        int shrinkwrap_iter_filter,
        int shrinkwrap_mask_multiply,
        bool shrinkwrap_fftshift_gaussian,
        float sigma, 
        float sigma_mult, 
        float beta,
        float beta_update,
        int beta_reset_subiter);

  
#ifdef __cplusplus
}
#endif
 
#endif //ER_H