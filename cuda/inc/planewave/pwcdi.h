
#ifndef SSC_PWCDI_H
#define SSC_PWCDI_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>  // For uint8_t

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "gpus.h"

#define tbx 16     // 16 // 
#define tby 8      // 4 // 
#define tbz 4      // 4 // 

#define SSC_MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define SSC_MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )

#define SSC_VARIABLE_ITER 0
#define SSC_VARIABLE_SUPP 1 

#define MULTIPLY_FULL   0
#define MULTIPLY_REAL   1
#define MULTIPLY_LEGACY 2

#define FILTER_FULL      0
#define FILTER_AMPLITUDE 1
#define FILTER_REAL_PART 2


#define AMPLITUDE     0
#define PHASE         1
#define REAL_PART     2
#define IMAG_PART     3


#define NO_EXTRA_CONSTRAINT 0 
#define LEFT_SEMIPLANE      1
#define RIGHT_SEMIPLANE     2
#define TOP_SEMIPLANE       3
#define BOTTOM_SEMIPLANE    4
#define FIRST_QUADRANT      5
#define SECOND_QUADRANT     6
#define THIRD_QUADRANT      7
#define FOURTH_QUADRANT     8

#define NO_ERR              0
#define ITER_DIFF           1



#define checkMyCudaErrors(call)                               \
  do{                                                         \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


#define BILLION 1E9
#define CLOCK  CLOCK_REALTIME
#define TIME(End,Start) (End.tv_sec - Start.tv_sec) + (End.tv_nsec-Start.tv_nsec)/BILLION

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct{
    cudaLibXtDesc *d_x, *d_y, *d_z; 
    cudaLibXtDesc *d_x_lasterr;                               // only allocated if err_type==ITER_DIFF 
    float **d_signal, **d_signal_host;  
    uint8_t **d_support, **d_support_host;
    cufftComplex **d_gaussian;
    cufftComplex *d_x_swap; 
  }multi_t;

  typedef struct{
    cufftComplex *d_x, *d_y, *d_z, *d_gaussian, *d_x_swap;
    cufftComplex *d_x_lasterr;                                // only allocated if err_type==ITER_DIFF 
    float *d_signal, *d_signal_host;
    uint8_t *d_support, *d_support_host;
  }single_t;
  
  typedef struct{
    int dimension;
    size_t nvoxels;        
    cufftHandle plan_C2C; 
    ssc_gpus *gpus;
    bool timing; 

    multi_t  mgpu;  //multi GPU
    single_t sgpu;  //single GPU

    float *errs;

    cufftComplex *host_swap; // host swap variable for multi-gpu
    uint8_t* host_swap_byte;
    
  }ssc_pwcdi_plan;

  typedef struct{
    int timing;
    int N;
    int sthreads;
    int pnorm;
    float radius;
    uint8_t* sup_data; 
    float eps_zeroamp;

    int err_type;
    int err_subiter;

    // memory parameters 
    bool map_d_signal;
    bool map_d_support;
    bool swap_d_x;
  }ssc_pwcdi_params;
  
  typedef struct{
    char *name;
    int iteration;
    int shrinkwrap_subiter; 
    int initial_shrinkwrap_subiter;
    int extra_constraint;
    int extra_constraint_subiter;
    int initial_extra_constraint_subiter;
    float shrinkwrap_threshold;                      
    int shrinkwrap_iter_filter;
    int shrinkwrap_mask_multiply;
    bool shrinkwrap_fftshift_gaussian;
    float sigma;
    float sigma_mult; 
    float beta;
    float beta_update;
    int beta_reset_subiter;
  }ssc_pwcdi_method;
  
  
  void pwcdi(cufftComplex* obj_output,  
             uint8_t* finsup_output,        
             float* data_input,        // inital data
             cufftComplex* obj_input,  // initial object
             int* gpu,
             int ngpu,
             int nalgorithms,
             ssc_pwcdi_params params,
             ssc_pwcdi_method* algorithms);

    
#ifdef __cplusplus
}
#endif


#endif // #ifndef SSC_PWCDI_H
