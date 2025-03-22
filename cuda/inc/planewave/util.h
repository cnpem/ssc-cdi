#ifndef PWUTILS_H
#define PWUTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>  // For uint8_t

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "pwcdi.h"

#ifdef __cplusplus
extern "C" {
#endif

  void alloc_workspace(ssc_pwcdi_plan *workspace,
                       ssc_pwcdi_params *params,
                       int *gpus,
                       int ngpu);
  
  void free_workspace(ssc_pwcdi_plan *workspace,
                      ssc_pwcdi_params *params);

  void set_input(ssc_pwcdi_plan *workspace,
                 ssc_pwcdi_params *params,
                 float *input,
                 cufftComplex* obj_input);
  
  void set_output(char *outpath,
                  ssc_pwcdi_params *params,
                  ssc_pwcdi_plan *workspace,
                  int variable,
                  int complex_part);

  // private functions

  void m_projection_M(cufftHandle& plan_C2C,
                      cudaLibXtDesc* d_y,
                      cudaLibXtDesc* d_x,  
                      float** d_signal, 
                      float eps,
                      size_t totalDim, 
                      size_t perGPUDim, 
                      ssc_gpus *gpus,
                      cufftComplex* host_swap,
                      bool timing);


  void m_projection_M_shuffleddata(cufftHandle& plan_C2C,
                                   cudaLibXtDesc* d_y,  
                                   cudaLibXtDesc* d_x, 
                                   float** d_signal,
                                   float eps,
                                   size_t totalDim, 
                                   size_t perGPUDim,
                                   ssc_gpus *gpus,
                                   // cudaLibXtDesc* device_swap,
                                   bool timing);
        
  void m_projection_S(cudaLibXtDesc* d_z,
                      cudaLibXtDesc* d_y,
                      uint8_t** d_support, 
                      int extra_constraint,
                      size_t totalDim, 
                      size_t perGPUDim, 
                      ssc_gpus *gpus);
  
  void m_projection_N(cudaLibXtDesc* d_x,
                      cudaLibXtDesc* d_z, 
                      size_t totalDim, 
                      size_t perGPUDim, 
                      ssc_gpus *gpus);

  void m_projection_S_only(cudaLibXtDesc* d_z,
                           cudaLibXtDesc* d_y,
                           uint8_t** d_support,
                           int extra_constraint,
                           size_t totalDim,
                           size_t perGPUDim,
                           ssc_gpus *gpus);
  
  void s_projection_M(cufftHandle& plan_input,
                      cufftComplex* d_y,
                      cufftComplex* d_x,
                      float* d_signal_host,
                      float eps,
                      int dimension);

  void s_projection_S(cufftHandle& plan_input,
                      cufftComplex* d_z,
                      cufftComplex* d_y,
                      uint8_t* d_support,
                      int extra_constraint,
                      int dimension);

  void s_projection_S_only(cufftHandle& plan_input,
                           cufftComplex* d_z,
                           cufftComplex* d_y,
                           uint8_t* d_support,
                           int extra_constraint,
                           int dimension);
  
  void s_projection_N(cufftComplex* d_x,
                      cufftComplex* d_z,
                      const int dimension);


  void permuted2natural(cudaLibXtDesc* var, 
                       cufftHandle& plan_input, 
                       size_t dim_size,
                       cufftComplex* host_var);
  


  void m_fftshift(uint8_t** data, 
                  size_t dimension, 
                  int dtype, 
                  uint8_t* host_swap_byte,
                  size_t perGPUDim,
                  ssc_gpus *gpus);



  void m_projection_extra_constraint(cudaLibXtDesc* d_x,
                                 cudaLibXtDesc* d_y,
                                 int extra_constraint,
                                 size_t totalDim,
                                 size_t perGPUDim,
                                 ssc_gpus *gpus);


  void s_projection_extra_constraint(cufftComplex* d_x,
                                    cufftComplex* d_y,
                                    int extra_constraint,
                                    int dimension);
 

  void get_output(cufftComplex* obj_output, 
                  uint8_t* finsup_output, 
                  ssc_pwcdi_plan* workspace, 
                  ssc_pwcdi_params* params);


#ifdef __cplusplus
}
#endif
 
#endif //PWUTILS_H
