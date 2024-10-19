#ifndef FFT_H
#define FFT_H

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>  // For uint8_t


#define SSC_DTYPE_CUFFTCOMPLEX 0
#define SSC_DTYPE_FLOAT 1
#define SSC_DTYPE_BYTE 2

#ifdef __cplusplus
extern "C" {
#endif


  __global__ void synth_support_data_mgpu(uint8_t* support, 
                                          int p, 
                                          float radius, 
                                          float x0, 
                                          float y0, 
                                          float z0,
          			                          int dimension, 
                                          int n_gpus, 
                                          int gpuId, 
                                          bool isNaturalOrder=true);

  __global__ void update_with_phase(cufftComplex *a, 
                                    cufftComplex *b, 
                                    float *m, 
                                    float eps, 
                                    int dimension);
    
  __global__ void update_with_phase_mgpu(cufftComplex *a, 
                                         cufftComplex *b, 
                                         float *m, 
                                         float eps, 
                                         size_t totalDim, 
                                         size_t perGPUDim);

  __global__ void update_with_support_extra(cufftComplex *a,
                                            cufftComplex *b,
                                            uint8_t *c,
                                            int extra_constraint,
                                            int dimension);

  __global__ void update_with_support_extra_mgpu(cufftComplex *a,
                                                 cufftComplex *b,
                                                 uint8_t *c,
                                                 int extra_constraint,
                                                 size_t perGPUDim); 

  __global__ void multiply_support_extra_mgpu(cufftComplex *a, 
                                              cufftComplex *b, 
                                              uint8_t *c, 
                                              int extra_constraint, 
                                              size_t perGPUDim);

  __global__ void multiply_support_extra(cufftComplex *a, 
                                         cufftComplex *b, 
                                         uint8_t *c, 
                                         int extra_constraint, 
                                         int dimension);

  __global__ void update_with_support(cufftComplex *a, 
                                      cufftComplex *b, 
                                      uint8_t *c, 
                                      int dimension);
    
  __global__ void update_with_support_mgpu(cufftComplex *a, 
                                           cufftComplex *b, 
                                           uint8_t *c, 
                                           size_t perGPUDim);

  __global__ void multiply_support_mgpu(cufftComplex *a, 
                                        cufftComplex *b, 
                                        uint8_t *c, 
                                        size_t perGPUDim);

  __global__ void multiply_support(cufftComplex *a, 
                                   cufftComplex *b, 
                                   uint8_t *c, 
                                   int dimension);
  
  __global__ void set_dx(cufftComplex *a, 
                         uint8_t *b, 
                         float *m, 
                         int dimension);
    
  __global__ void set_dx_mgpu(cufftComplex *a, 
                              uint8_t *b, 
                              float *m, 
                              size_t perGPUdim);

  __global__ void set_initial_mgpu(cufftComplex *a, 
                                   float *m, 
                                   size_t perGPUDim);

  __global__ void set_initial(cufftComplex *a, 
                              float *m, 
                              size_t perGPUDim);
 
 

  __global__ void gaussian1(cufftComplex *d_gaussian, 
                            float sigma, 
                            int dimension);

 
  __global__ void gaussian1_mgpu(cufftComplex *d_x, 
                                 float sigma, 
                                 int dimension, 
                                 int n_gpus, 
                                 int gpuId);
 

  __global__ void gaussian1_freq_fftshift_mgpu(cufftComplex *d_x, 
                                               float sigma, 
                                               int dimension, 
                                               int n_gpus, 
                                               int gpuId);
  
  __global__ void set_difference(cufftComplex *a, 
                                 cufftComplex *b, 
                                 cufftComplex *c, 
                                 float alpha, 
                                 float beta, 
                                 int dimension);
    
  __global__ void set_difference_mgpu(cufftComplex *a, 
                                      cufftComplex *b, 
                                      cufftComplex *c, 
                                      float alpha, 
                                      float beta, 
                                      size_t perGPUDim);

 
    
  __global__ void multiply(cufftComplex *a, 
                           cufftComplex *b, 
                           cufftComplex *c, 
                           int dimension);
    
  __global__ void multiply_rc(cufftComplex *a, 
                              cufftComplex *b, 
                              cufftComplex *c, 
                              int dimension);

  __global__ void multiply_legacy(cufftComplex *a, 
                                  cufftComplex *b, 
                                  cufftComplex *c, 
                                  int dimension);

  __global__ void multiply_mgpu(cufftComplex *a, 
                                cufftComplex *b, 
                                cufftComplex *c, 
                                size_t perGPUDim);
    
  __global__ void multiply_rc_mgpu(cufftComplex *a, 
                                   cufftComplex *b, 
                                   cufftComplex *c, 
                                   size_t perGPUDim);
    
  __global__ void multiply_legacy_mgpu(cufftComplex *a, 
                                       cufftComplex *b, 
                                       cufftComplex *c, 
                                       size_t perGPUDim);

  __global__ void mask(cufftComplex *a, 
                       cufftComplex *b, 
                       int dimension);
    
  __global__ void mask_mgpu(cufftComplex *a, 
                            cufftComplex *b, 
                            size_t perGPUDim);
    
  __global__ void get_max(cufftComplex *a, 
                          int *result, 
                          int dimension);
    
  __global__ void fftshift(void *input, 
                           int N, 
                           int dtype);

  __global__ void normalize(cufftComplex *a, 
                            int dimension);
    
  __global__ void normalize_mgpu(cufftComplex *a, 
                                 size_t totalDim,
                                 size_t perGPUDim);

  __global__ void update_support_sw_mgpu(uint8_t* supp, 
                                         cufftComplex *iter, 
                                         float threshold, 
                                         float globalMax, 
                                         float globalMin, 
                                         size_t perGPUDim);

  __global__ void update_support_sw(uint8_t* supp, 
                                    cufftComplex *iter, 
                                    float threshold, 
                                    float globalMax, 
                                    float globalMin, 
                                    size_t idxMax, 
                                    size_t idxMin, 
                                    size_t dimension, 
                                    bool which);
  
  __global__ void absolute_value_mgpu(cufftComplex *a, 
                                       cufftComplex *b, 
                                       size_t totalDim, 
                                       size_t perGPUDim);

  __global__ void absolute_value(cufftComplex *a, 
                                  cufftComplex *b, 
                                  int dimension);

  __global__ void real_part(cufftComplex *a, 
                            cufftComplex *b, 
                            int dimension);


  __global__ void clip_to_zero(cufftComplex *a,
                    cufftComplex *b,
                    float eps,
                    int dimension);
 

  __global__ void update_extra_constraint(cufftComplex *a,
                                         cufftComplex *b,
                                         int extra_constraint, 
                                         int dimension);


  __global__ void update_extra_constraint_mgpu(cufftComplex *a,
                                              cufftComplex *b,
                                              int extra_constraint,
                                              size_t perGPUDim);
 

#ifdef __cplusplus
}
#endif 

#endif // #ifndef FFT_H