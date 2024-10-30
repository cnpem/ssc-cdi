#include <stdint.h>  // For uint8_t

#include "fft.h"
#include "pwcdi.h"

extern "C" {

  __global__ void synth_support_data_mgpu(uint8_t* support,
                                          int p,
                                          float radius,
                                          float x0,
                                          float y0,
                                          float z0,
                                          int dimension,
                                          int n_gpus,
                                          int gpuId,
                                          bool isNaturalOrder){

    const size_t index = blockIdx.x*blockDim.x + threadIdx.x; 

    int ndimx, ndimy;  
    if (isNaturalOrder){
      ndimx = dimension/n_gpus; 
      ndimy = dimension;        
    }else{
      ndimx = dimension;                
      ndimy = dimension/n_gpus;          
    }

    const int ndimz =  dimension;               

    const size_t perGPUDim = (size_t) ndimx*ndimy*ndimz;   
  
    const int ix = (index/(ndimy*ndimz)) ;                   
    const int iy = ((index - ix*ndimy*ndimz)/ndimz);       
    const int iz = (index - ix*ndimy*ndimz - iy*ndimz);      

    float x, y, z;
    if (index<perGPUDim){        
      const float rp = powf(radius, p);                                  
      const float a = 1.0;
      const float delta = (2.0*a)/dimension;                            
      
      if (isNaturalOrder){
        x = -a + (ix + gpuId*ndimx)*delta;            
        y = -a + iy*delta;                             
      }else{
        x = -a + ix*delta;                          
        y = -a + (iy + gpuId*ndimy)*delta;          
      }

      z = -a + iz*delta;                    
      float norm = powf((x-x0), p) + powf((y-y0), p) + powf((z-z0), p);
      
      if(norm<rp){
        support[index] = 1;
      }else{
        support[index] = 0;
      }
    } 
  }


__global__ void clip_to_zero(cufftComplex *a,
                         cufftComplex *b,
                         float eps,
                         int dimension){

  const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
  const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
  const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

  const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

  if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
    const float absolute_value = sqrt(powf(fabs(b[index].x),2.0)+powf(fabs(b[index].y),2.0));
    if (absolute_value <= eps){
      a[index].x = 0.0f;
      a[index].y = 0.0f;
    }
  }
}




__global__ void update_extra_constraint(cufftComplex *a,
                                       cufftComplex *b,
                                       int extra_constraint, 
                                       int dimension){

  const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
  const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
  const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

  const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

  if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
    // impose real(b)>0 or real(b)<0 
    if (extra_constraint==RIGHT_SEMIPLANE || extra_constraint==FIRST_QUADRANT || extra_constraint==FOURTH_QUADRANT){
      if (b[index].x<0){        
        a[index].x = 0.0f;
      }
    }else if(extra_constraint==LEFT_SEMIPLANE || extra_constraint==SECOND_QUADRANT || extra_constraint==THIRD_QUADRANT){
      if (b[index].x>0){        
        a[index].x = 0.0f;
      }
    }

    // impose imag(b)>0 or imag(b)<0
    if(extra_constraint==TOP_SEMIPLANE || extra_constraint==FIRST_QUADRANT || extra_constraint==SECOND_QUADRANT){
      if (b[index].y<0){
        a[index].y = 0.0f;
      }
    }else if(extra_constraint==BOTTOM_SEMIPLANE || extra_constraint==THIRD_QUADRANT || extra_constraint==FOURTH_QUADRANT){
      if (b[index].y>0){
        a[index].y = 0.0f;
      }
    }

  }
}

__global__ void update_extra_constraint_mgpu(cufftComplex *a,
                                            cufftComplex *b,
                                            int extra_constraint,
                                            size_t perGPUDim){
 
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < perGPUDim){
    // impose real(b)>0 or real(b)<0 
    if (extra_constraint==RIGHT_SEMIPLANE || extra_constraint==FIRST_QUADRANT || extra_constraint==FOURTH_QUADRANT){
      if (b[index].x<0){        
        a[index].x = 0.0f;
      }
    }else if(extra_constraint==LEFT_SEMIPLANE || extra_constraint==SECOND_QUADRANT || extra_constraint==THIRD_QUADRANT){
      if (b[index].x>0){        
        a[index].x = 0.0f;
      }
    }

    // impose imag(b)>0 or imag(b)<0
    if(extra_constraint==TOP_SEMIPLANE || extra_constraint==FIRST_QUADRANT || extra_constraint==SECOND_QUADRANT){
      if (b[index].y<0){
        a[index].y = 0.0f;
      }
    }else if(extra_constraint==BOTTOM_SEMIPLANE || extra_constraint==THIRD_QUADRANT || extra_constraint==FOURTH_QUADRANT){
      if (b[index].y>0){
        a[index].y = 0.0f;
      }
    }
  }

}


 
  // Kernel access positions was modified
  __global__ void update_with_phase(cufftComplex *a,
                                    cufftComplex *b,
                                    float *m,
                                    float eps,
                                    int dimension){
    // 
    // This function receives b = fft(x) and m = measured signal (real), computes  
    //      a = m * exp(i * phase(b))/nvoxels  
    // that is, storing the result in a, the first pointer in the arguments. 
    //  
    // Some of the values of m are probably missing, and they are flagged as -1. 
    // This kernel handles the -1 values by substituting them with the computed fft(x)
    // values stored in b, the second pointer in the arguments.
    // 
    float cos, sin, theta;
    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    // if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
    //   if (m[index]<0){
    //     // handle -1 values
    //     a[index].x = b[index].x/((float)(dimension*dimension*dimension));
    //     a[index].y = b[index].y/((float)(dimension*dimension*dimension));
    //     // a[index] = b[index];
    //   }else if(eps>=0 && sqrtf(powf(fabs(b[index].x),2.0)+powf(fabs(b[index].y),2.0))<=eps){
    //     a[index].x = m[index]/((float) (dimension*dimension*dimension));
    //     a[index].y = 0.0f;
    //   }else{
    //     theta = atan2f(b[index].y, b[index].x);
    //     __sincosf(theta, &sin, &cos);
    //     a[index].x = (m[index]*cos)/((float) (dimension*dimension*dimension));
    //     a[index].y = (m[index]*sin)/((float) (dimension*dimension*dimension));
    //   }
    // }

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      if (m[index]<0){
        // handle -1 values
        a[index].x = b[index].x/((float)(dimension*dimension*dimension));
        a[index].y = b[index].y/((float)(dimension*dimension*dimension));
        // a[index] = b[index];
      }else{
        const float fft_abs = sqrtf(powf(fabs(b[index].x),2.0)+powf(fabs(b[index].y),2.0));
        if(eps>=0 && fft_abs<=eps){
          a[index].x = m[index]/((float) (dimension*dimension*dimension));
          a[index].y = 0.0f;
        }else{
          a[index].x = (m[index]*b[index].x)/((float) fft_abs*(dimension*dimension*dimension));
          a[index].y = (m[index]*b[index].y)/((float) fft_abs*(dimension*dimension*dimension));
        }
      }
    }


  }

  // for multiple GPU 
  __global__ void update_with_phase_mgpu(cufftComplex *a,
                                         cufftComplex *b,
                                         float *m,
                                         float eps,
                                         size_t totalDim,
                                         size_t perGPUDim){
    //
    // a = m * exp(i * phase(b))/nvoxels :  m is real
    // All variables a,b and m should have the same ordering.
    //
    float cos, sin, theta;
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < perGPUDim){
      if (m[index]<0){
        // handle -1 values
        a[index].x = b[index].x/((float) totalDim);
        a[index].y = b[index].y/((float) totalDim);
        // a[index] = b[index];
      }else{
        const float fft_abs = sqrtf(powf(fabs(b[index].x),2.0)+powf(fabs(b[index].y),2.0));
        if(eps>=0 && fft_abs<=eps){ 
        a[index].x = m[index]/((float) totalDim);
        a[index].y = 0.0f;
        }else{
          // theta = atan2f(b[index].y, b[index].x);
          // __sincosf(theta, &sin, &cos);
          // a[index].x = (m[index]*cos)/((float) totalDim);
          // a[index].y = (m[index]*sin)/((float) totalDim);
          a[index].x = (m[index]*b[index].x)/((float) fft_abs*totalDim);
          a[index].y = (m[index]*b[index].y)/((float) fft_abs*totalDim);
        }
      }
    }
  }




  __global__ void update_with_support_extra(cufftComplex *a,
                                            cufftComplex *b,
                                            uint8_t *c,
                                            int extra_constraint,
                                            int dimension){


    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx < dimension && threadIDy < dimension && threadIDz < dimension){
      const float supp = (float) c[index];
      float supp_mod;

      // resolve extra_constraint
      switch (extra_constraint){
        case NO_EXTRA_CONSTRAINT:
          supp_mod = 1.0f;
          break;
        case LEFT_SEMIPLANE:
          supp_mod = (b[index].x > 0) ? 1.0f : 0.0f;
          break;
        case RIGHT_SEMIPLANE:
          supp_mod = (b[index].x < 0) ? 1.0f : 0.0f;
          break;
        case TOP_SEMIPLANE:
          supp_mod = (b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case BOTTOM_SEMIPLANE:
          supp_mod = (b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FIRST_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case SECOND_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case THIRD_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FOURTH_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
      }

      // project onto the support AND the extra phase constraint 
      a[index].x = b[index].x*supp*supp_mod + a[index].x*(1.0f - supp*supp_mod);
      a[index].y = b[index].y*supp*supp_mod + a[index].y*(1.0f - supp*supp_mod);
  }
}


  __global__ void update_with_support_extra_mgpu(cufftComplex *a,
                                                 cufftComplex *b,
                                                 uint8_t *c,
                                                 int extra_constraint,
                                                 size_t perGPUDim){


    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;
  
    if (index < perGPUDim){  
      float supp = (float) c[index];
      float supp_mod;

      // resolve extra_constraint
      switch (extra_constraint){
        case NO_EXTRA_CONSTRAINT:
          supp_mod = 1.0f;
          break;
        case LEFT_SEMIPLANE:
          supp_mod = (b[index].x > 0) ? 1.0f : 0.0f;
          break;
        case RIGHT_SEMIPLANE:
          supp_mod = (b[index].x < 0) ? 1.0f : 0.0f;
          break;
        case TOP_SEMIPLANE:
          supp_mod = (b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case BOTTOM_SEMIPLANE:
          supp_mod = (b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FIRST_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case SECOND_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case THIRD_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FOURTH_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
      }

      // project onto the support AND the extra phase constraint 
      a[index].x = b[index].x*supp*supp_mod + a[index].x*(1.0f - supp*supp_mod);
      a[index].y = b[index].y*supp*supp_mod + a[index].y*(1.0f - supp*supp_mod);


  }
}


  __global__ void multiply_support_extra_mgpu(cufftComplex *a,
                                              cufftComplex *b,
                                              uint8_t *c,
                                              int extra_constraint,
                                              size_t perGPUDim){
  /**
   * @brief Updates a complex array `a` by performing a constrained projection of the array `b` onto a support array `c`, with additional constraints if specified. 
   * This version is designed for execution across multiple GPUs.
   *
   * This CUDA kernel function updates a complex array `a` by performing a projection of the array `b` onto the support array `c`, 
   * with an additional optional constraint specified by `extra_constraint`. The projection is performed based on the support array `c` 
   * and an additional modulation factor `supp_mod` determined by `extra_constraint`.
   *
   * The update rule is given by:
   * 
   *  a[index].x = b[index].x * supp * supp_mod
   *  a[index].y = b[index].y * supp * supp_mod
   * 
   * where `index` is the linear index within the arrays `a`, `b`, and `c`, and `perGPUDim` is the total number of elements per GPU segment.
   * `supp` is the support value from `c`, and `supp_mod` is a modulation factor that depends on `extra_constraint`.
   *
   * The `extra_constraint` parameter determines how `supp_mod` is calculated:
   * - `NO_EXTRA_CONSTRAINT`: No additional constraint (supp_mod = 1.0f).
   * - `LEFT_SEMIPLANE`, `RIGHT_SEMIPLANE`, `TOP_SEMIPLANE`, `BOTTOM_SEMIPLANE`: Constraints based on the sign of `b[index].x` or `b[index].y`.
   * - `FIRST_QUADRANT`, `SECOND_QUADRANT`, `THIRD_QUADRANT`, `FOURTH_QUADRANT`: Constraints based on the quadrant of the complex number `b[index]`.
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`) after the update.
   * @param b Pointer to the input array of complex values (`cufftComplex`) to be projected and updated.
   * @param c Pointer to the input array of support constraints (`uint8_t`) defining which elements to update and blend.
   * @param extra_constraint Additional constraint type (0 for no extra constraint, other values for quadrant/semiplane constraints).
   * @param perGPUDim Size of the array segment allocated per GPU (total elements per GPU).
   *
   * @note This function assumes that `a`, `b`, and `c` are preallocated and have dimensions aligned for execution across multiple GPUs.
   *       It computes the update along the CUDA grid and block dimensions for each GPU segment.
   */
 
    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    
    if (index<perGPUDim){  
      float supp = (float) c[index];      
      float supp_mod;

      // resolve extra_constraint
      switch (extra_constraint){
        case NO_EXTRA_CONSTRAINT:
          supp_mod = 1.0f;
          break;
        case LEFT_SEMIPLANE:
          supp_mod = (b[index].x > 0) ? 1.0f : 0.0f;
          break;
        case RIGHT_SEMIPLANE:
          supp_mod = (b[index].x < 0) ? 1.0f : 0.0f;
          break;
        case TOP_SEMIPLANE:
          supp_mod = (b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case BOTTOM_SEMIPLANE:
          supp_mod = (b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FIRST_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case SECOND_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case THIRD_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FOURTH_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
      }

      // project
      a[index].x = (float) b[index].x*supp*supp_mod;
      a[index].y = (float) b[index].y*supp*supp_mod;

    }
  }

  __global__ void multiply_support_extra(cufftComplex *a,
                                         cufftComplex *b,
                                         uint8_t *c,
                                         int extra_constraint,
                                         int dimension){
  /**
   * @brief Updates a complex array `a` by performing a constrained projection of the array `b` onto a support array `c`, with additional constraints if specified.
   *
   * This CUDA kernel function updates a complex array `a` by performing a projection of the array `b` onto the support array `c`, with an additional optional constraint specified by `extra_constraint`.
   * The projection is performed based on the support array `c` and an additional modulation factor `supp_mod` determined by `extra_constraint`.
   *
   * The update rule is given by:
   * 
   *  a[index].x = b[index].x * supp * supp_mod
   *  a[index].y = b[index].y * supp * supp_mod
   * 
   * where `index` is the linear index within the arrays `a`, `b`, and `c`, and `dimension` is the total dimension size of each dimension of the cube.
   * `supp` is the support value from `c`, and `supp_mod` is a modulation factor that depends on `extra_constraint`.
   *
   * The `extra_constraint` parameter determines how `supp_mod` is calculated:
   * - `NO_EXTRA_CONSTRAINT`: No additional constraint (supp_mod = 1.0f).
   * - `LEFT_SEMIPLANE`, `RIGHT_SEMIPLANE`, `TOP_SEMIPLANE`, `BOTTOM_SEMIPLANE`: Constraints based on the sign of `b[index].x` or `b[index].y`.
   * - `FIRST_QUADRANT`, `SECOND_QUADRANT`, `THIRD_QUADRANT`, `FOURTH_QUADRANT`: Constraints based on the quadrant of the complex number `b[index]`.
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`) after the update.
   * @param b Pointer to the input array of complex values (`cufftComplex`) to be projected and updated.
   * @param c Pointer to the input array of support constraints (`uint8_t`) defining which elements to update and blend.
   * @param extra_constraint Additional constraint type (0 for no extra constraint, other values for quadrant/semiplane constraints).
   * @param dimension Size of each dimension of the cubic array.
   *
   * @note This function assumes that `a`, `b`, and `c` are preallocated and have dimensions aligned for the single GPU execution.
   *       It computes the update along the CUDA grid and block dimensions for the single GPU execution.
   */
 
    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx < dimension && threadIDy < dimension && threadIDz < dimension){
      float supp = (float) c[index];      
      float supp_mod;

      // resolve extra_constraint
      switch (extra_constraint){
        case NO_EXTRA_CONSTRAINT:
          supp_mod = 1.0f;
          break;
        case LEFT_SEMIPLANE:
          supp_mod = (b[index].x > 0) ? 1.0f : 0.0f;
          break;
        case RIGHT_SEMIPLANE:
          supp_mod = (b[index].x < 0) ? 1.0f : 0.0f;
          break;
        case TOP_SEMIPLANE:
          supp_mod = (b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case BOTTOM_SEMIPLANE:
          supp_mod = (b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FIRST_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case SECOND_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y > 0) ? 1.0f : 0.0f;
          break;
        case THIRD_QUADRANT:
          supp_mod = (b[index].x < 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
        case FOURTH_QUADRANT:
          supp_mod = (b[index].x > 0 && b[index].y < 0) ? 1.0f : 0.0f;
          break;
      }
       
      // project the real and imaginary parts onto the support constrained domain 
      a[index].x = (float) b[index].x*supp*supp_mod;
      a[index].y = (float) b[index].y*supp*supp_mod;
    }
  }




  
  __global__ void update_with_support(cufftComplex *a,
                                      cufftComplex *b,
                                      uint8_t *c,
                                      int dimension){ 
   /**
   * @brief Updates a complex array `a` by performing the projection of the array `b` onto a support array `c`. This procedure is part of the 
   * Hybrid Input-Output (HIO) algorithm, here performed using a single GPU.
   *
   * This CUDA kernel function updates a complex array `a` by performing the projection onto the support array `c`, which is part of the Hybrid 
   * Input-Output (HIO) algorithm. The update is performed by projecting the real and imaginary parts of `b` onto the support constrained domain defined by `c`, and then
   * blending with the existing values in `a` based on the support constraint. The update rule is given by:
   * 
   *  a[index].x = b[index].x * supp + a[index].x * (1.0f - supp)
   *  a[index].y = b[index].y * supp + a[index].y * (1.0f - supp)
   * 
   * where `index` is the linear index within the arrays `a`, `b`, and `c`, and `dimension` is the total dimension size of each dimension of the cube.
   * This was implemented to be used with multiple GPUs, that is, the data is spread accross multiple GPUs. 
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`) after the update.
   * @param b Pointer to the input array of complex values (`cufftComplex`) to be projected and updated.
   * @param c Pointer to the input array of support constraints (`uint8_t`) defining which elements to update and blend.
   * @param dimension Size of each dimension of the cubic array.
   *
   * @note This function assumes that `a`, `b`, and `c` are preallocated and have dimensions aligned for the single GPU execution.
   *       It computes the update along the CUDA grid and block dimensions for the single GPU execution.
   */

    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx < dimension && threadIDy < dimension && threadIDz < dimension){
      const float supp = (float) c[index];
   
      // project the real and imaginary parts onto the support constrained domain 
      a[index].x = b[index].x*supp + a[index].x*(1.0f - supp);
      a[index].y = b[index].y*supp + a[index].y*(1.0f - supp);

    }
  }

  __global__ void update_with_support_mgpu(cufftComplex *a,
                                           cufftComplex *b,
                                           uint8_t *c,
                                           size_t perGPUDim){

  /**
   * @brief Updates a complex array `a` by performing the projection of the array `b` onto a support array `c`. This procedure is part of the 
   * Hybrid Input-Output (HIO) algorithm, here performed using multiple GPUs.
   *
   * This CUDA kernel function updates a complex array `a` by performing the projection onto the support array `c`, which is part of the Hybrid 
   * Input-Output (HIO) algorithm. The update is performed by projecting the real and imaginary parts of `b` onto the support constrained domain defined by `c`, and then
   * blending with the existing values in `a` based on the support constraint. The update rule is given by:
   * 
   *  a[index].x = b[index].x * supp + a[index].x * (1.0f - supp)
   *  a[index].y = b[index].y * supp + a[index].y * (1.0f - supp)
   * 
   * where `index` is the linear index within the portion of arrays `a`, `b`, and `c` allocated in a given GPU, identified by `perGPUDim`. This
   * was implemented to be used with multiple GPUs, that is, the data is spread accross multiple GPUs. 
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`) after the update.
   * @param b Pointer to the input array of complex values (`cufftComplex`) to be projected and updated.
   * @param c Pointer to the input array of support constraints (`uint8_t`) defining which elements to update and blend.
   * @param perGPUDim Size of the array segment allocated per GPU (total elements per GPU).
   *
   * @note This function assumes that `a`, `b`, and `c` are preallocated and have dimensions aligned for the multiple GPU execution.
   *       It computes the update along the CUDA grid and block dimensions for each GPU segment.
   */

    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;
  
    if (index < perGPUDim){  
      float supp = (float) c[index];
       
      // project the real and imaginary parts onto the support constrained domain 
      a[index].x = b[index].x*supp + a[index].x*(1.0f - supp);
      a[index].y = b[index].y*supp + a[index].y*(1.0f - supp);
    }
  }


  __global__ void multiply_support_mgpu(cufftComplex *a,
                                        cufftComplex *b,
                                        uint8_t *c,
                                        size_t perGPUDim){
  /**
   * @brief Multiplies a complex array `b` by a support constraint array `c` and stores the result in array `a` for multiple GPUs.
   *
   * This CUDA kernel function computes the element-wise multiplication of a complex array `b` with a support constraint array `c`,
   * and stores the result in array `a`. The support constraint array `c` determines which elements of `b` are multiplied by their
   * corresponding elements in `c`. The complex multiplication is performed such that:
   * 
   *  a[index].x = b[index].x * c[index]
   *  a[index].y = b[index].y * c[index]
   * 
   * where `index` is the linear index within the portion of arrays `a`, `b`, and `c` allocated for a single GPU, identified by `perGPUDim`.
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`) after multiplication.
   * @param b Pointer to the input array of complex values (`cufftComplex`) to be multiplied.
   * @param c Pointer to the input array of support constraints (`uint8_t`) defining which elements to multiply.
   * @param perGPUDim Size of the array segment allocated per GPU (total elements per GPU).
   *
   * @note This function assumes that `a`, `b`, and `c` are preallocated and have dimensions aligned for the multiple GPU execution.
   *       It computes the multiplication along the CUDA grid and block dimensions for each GPU segment.
   */

    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    
    if (index<perGPUDim){  
      float supp = (float) c[index];

      a[index].x = (float) b[index].x*supp;
      a[index].y = (float) b[index].y*supp;

    }
  }

  __global__ void multiply_support(cufftComplex *a,
                                   cufftComplex *b,
                                   uint8_t *c,
                                   int dimension){
  /**
   * @brief Multiplies a complex array `b` by a support constraint array `c` and stores the result in array `a`.
   *
   * This CUDA kernel function computes the element-wise multiplication of a complex array `b` with a support constraint
   * array `c`, and stores the result in array `a`. The support constraint array `c` determines which elements of `b` are
   * multiplied by their corresponding elements in `c`. The complex multiplication is performed such that:
   * 
   *  a[index].x = b[index].x * c[index]
   *  a[index].y = b[index].y * c[index]
   * 
   * where `index` represents the linear index in the 3D volume defined by `dimension`.
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`) after multiplication.
   * @param b Pointer to the input array of complex values (`cufftComplex`) to be multiplied.
   * @param c Pointer to the input array of support constraints (`uint8_t`) defining which elements to multiply.
   * @param dimension Size of each dimension of the 3D volume represented by `a`, `b`, and `c` (total elements in each dimension).
   *
   * @note This function assumes that `a`, `b`, and `c` are preallocated and have dimensions aligned for the single GPU execution.
   *       It computes the multiplication along the CUDA grid and block dimensions.
   */
    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx < dimension && threadIDy < dimension && threadIDz < dimension){
      float supp = (float) c[index];
       
      // project the real and imaginary parts onto the support constrained domain 
      a[index].x = (float) b[index].x*supp;
      a[index].y = (float) b[index].y*supp;
    }
  }


  __global__ void set_dx(cufftComplex *a,
                         uint8_t *b,
                         float *m,
                         int dimension){
    //
    // a = m * exp(1j * b) : m is real
    //
    float sin, cos, theta;
    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      theta = (float) b[index];

      __sincosf(theta, &sin, &cos);

      a[index].x = m[index]*cos;
      a[index].y = m[index]*sin;
    }
  }


  // for multiple GPU 
  __global__ void set_dx_mgpu(cufftComplex *a,
                              uint8_t *b,
                              float *m,
                              size_t perGPUDim){
    //
    // a = m * exp(1j * b) : m is real
    //

    float sin, cos, theta;
    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;
  
    if (index < perGPUDim){
      theta = (float) b[index];

      __sincosf(theta, &sin, &cos);

      a[index].x = m[index]*cos;
      a[index].y = m[index]*sin;
    }
  }

  __global__ void set_initial(cufftComplex *a,
                              float *m,
                              size_t dimension){
  /**
   * @brief Sets initial values for a complex array based on a given real scalar amplitude and a computed random phase.
   *
   * This CUDA kernel function sets initial values for a complex array `a` based on a real scalar `m`
   * and a random phase angle. The complex values are computed as:
   * 
   *  a = m * exp(1j * random), 
   * 
   * where `m` is real, and `random` is uniform noise. If the amplitude is less than or equal to zero in a given point,
   * then the amplitude is assumed to be zero.
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`).
   * @param m Pointer to the input array of real scalars.
   * @param dimension Size of each dimension of the 3D volume represented by `a` (total elements in each dimension).
   *
   * @note This function assumes that `a` and `m` are preallocated and have dimensions aligned for the single GPU execution.
   *       It computes the initial complex values along the CUDA grid and block dimensions.
   */
    float sin, cos, theta;

    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;


    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      curandState state;
      // float r = 0.0f;
      float r = curand_uniform_double(&state); // 0.0f 
      theta = r;
      __sincosf(theta, &sin, &cos);

      // handle -1 values in measured data
      if (m[index]<=0){
        a[index].x = cos; //0.0f;
        a[index].y = sin; //0.0f;
      }else{
        // a[index].x = m[index]
        a[index].x = m[index]*cos;
        a[index].y = m[index]*sin;
      }
    }    
  }

 
  __global__ void set_initial_mgpu(cufftComplex *a,
                                   float *m,
                                   size_t perGPUDim){
  /**
   * @brief Sets initial values for a complex array based on a given real scalar amplitude and a computed random phase.
   * This was implemented to be used for data stored in multiple GPUs.
   *
   * This CUDA kernel function sets initial values for a complex array `a` based on a real scalar `m`
   * and a random phase angle, assuming that the data is stored in multiple GPU. The complex values are computed as:
   * 
   *  a = m * exp(1j * random), 
   * 
   * where `m` is real, and `random` is uniform noise. If the amplitude is less than or equal to zero in a given point,
   * then the amplitude is assumed to be zero.
   *
   * @param a Pointer to the output array of complex values (`cufftComplex`).
   * @param m Pointer to the input array of real scalars.
   * @param perGPUDim Size of the array per GPU (total elements per GPU).
   *
   * @note This function assumes that `a` and `m` are preallocated and have dimensions aligned for the number of GPUs.
   *       It computes the initial complex values along the CUDA grid and block dimensions.
   */
    float sin, cos, theta;
    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < perGPUDim ){
      curandState state;
      // float r = 0.0f;
      float r = curand_uniform_double(&state); // 0.0f
      theta = r;
      __sincosf(theta, &sin, &cos);

      if (m[index]<=0){
        a[index].x = cos; //0.0f; // cos;
        a[index].y = sin; // 0.0f; // sin;
      }else{
        a[index].x = m[index]*cos;
        a[index].y = m[index]*sin;
      }
    }
  }

  /*
  __global__ void get_support_mgpu(uint8_t *m,
                                   uint8_t *a,
                                   size_t perGPUDim){
      //
      // m = a 
      //
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < perGPUDim ){
      m[index] = a[index];
    }
  }
  */
  


  __global__ void gaussian1(cufftComplex *d_gaussian,
                            float sigma,
                            int dimension){
  /**
   * @brief Computes a 3D Gaussian distribution in Fourier space.
   *
   * This CUDA kernel function computes a 3D Gaussian distribution in Fourier space.
   * The Gaussian function is given by:
   *   (1 / (sigma * sqrt(2 * pi))) * exp( -( (xx**2 + yy**2 + zz**2) / (2 * sigma ** 2) ) )
   *
   * The resulting complex values are stored in `d_gaussian`.
   *
   * @param d_gaussian Pointer to the output array (`cufftComplex`) where the Gaussian values will be stored.
   * @param sigma Standard deviation of the Gaussian distribution.
   * @param dimension Total dimension size of the data cube along each dimension (assuming cubic).
   *
   * @note This function assumes that `d_gaussian` is preallocated and has dimensions aligned for cubic data.
   *       It computes the Gaussian distribution along the x, y, and z dimensions of the 3D volume.
   */
    const float PI_F = 3.141592654f;
    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const float a     = 1.0;
    const float wc    = (float) dimension/(4*a); //nyquist
    const float delta = (2.0*wc)/(dimension);
 
    
    float x = -wc + threadIDx*delta;
    float y = -wc + threadIDy*delta;
    float z = -wc + threadIDz*delta;
 
    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      d_gaussian[index].x = exp((-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z))*(1/(sigma*sqrt(2*PI_F)));
      d_gaussian[index].y = 0.0f;
    }
  }











  __global__ void gaussian1_mgpu(cufftComplex *d_x,
                                 float sigma,
                                 int dimension,
                                 int n_gpus,
                                 int gpuId){
    /**
     * @brief Computes a 3D Gaussian distribution in Fourier space for multiple GPUs.
     *
     * This CUDA kernel function computes a 3D Gaussian distribution in Fourier space for multiple GPUs.
     * The Gaussian function is given by:
     *   (1 / (sigma * sqrt(2 * pi))) * exp( -( (xx**2 + yy**2 + zz**2) / (2 * sigma**2) )
     *
     * The resulting complex values are stored in `d_x`.
     *
     * @param d_x Pointer to the output array (`cufftComplex`) where the Gaussian values will be stored.
     * @param sigma Standard deviation of the Gaussian distribution.
     * @param dimension Total dimension size of the data cube along each dimension (assuming cubic).
     * @param n_gpus Total number of GPUs being used.
     * @param gpuId GPU identifier (0-indexed) indicating the current GPU's position.
     *
     * @note This function assumes that `d_x` is preallocated and has dimensions aligned such that `dimension / n_gpus` is the slowest dimension.
     *       It computes the Gaussian distribution along the x, y, and z dimensions of the 3D volume, distributed across multiple GPUs.
     */
    const float PI_F = 3.141592654f;
    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    
    const int ndimx = dimension/n_gpus; //slowest dimension
    const int ndimy = dimension;
    const int ndimz = dimension;
    
    const size_t perGPUDim = (size_t) ndimx*ndimy*ndimz;
    
    const float a = 1.0;

    const float wc    = (float) dimension/(4*a); //nyquist 
    const float delta = (2.0*wc)/(dimension); // dimension-1?
    
    const int ix = index/(ndimy*ndimz) ;
    const int iy = (index - ix*ndimy*ndimz)/ndimz;
    const int iz = index - ix*ndimy*ndimz - iy*ndimz;
 
    
    if (index < perGPUDim){
      float x = -wc + (ix + 1*gpuId*ndimx)*delta;      // -a 
      float y = -wc + (iy )*delta;                     // -a
      float z = -wc + iz*delta;                        // -a 
 
      d_x[index].x = exp((-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z))*(1/(sigma*sqrt(2*PI_F)));
      d_x[index].y = 0.0f;

 
    }
  }
  
  typedef struct{
    int ix;
    int iy;
    int iz;
  }index3;

  // __device__ index3 get3Dindex(const size_t index,
  //                              const int ndimy,
  //                              const int ndimz){
  //   index3 id3;
  //   id3.ix = index/ndimy/ndimz;
  //   id3.iy = (index - id3.ix*ndimy*ndimz)/ndimz;
  //   id3.iz = index - id3.ix*ndimy*ndimz - id3.iy*ndimz;   
  //   return id3;
  // }

  __device__ index3 get3Dindex(const size_t index,
                             const int ndimy,
                             const int ndimz){
  index3 id3;
  id3.ix = index / (ndimy * ndimz);  // Compute the x-coordinate
  id3.iy = (index % (ndimy * ndimz)) / ndimz;  // Compute the y-coordinate
  id3.iz = index % ndimz;  // Compute the z-coordinate
  return id3;
  }


  __global__ void gaussian1_freq_fftshift_mgpu(cufftComplex *d_x,
                                              float sigma,
                                              int dimension,
                                              int n_gpus,
                                              int gpuId){
  /**
   * @brief Computes a 3D Gaussian distribution in Fourier space with frequency shift for multiple GPUs.
   *
   * This CUDA kernel function computes a 3D Gaussian distribution in Fourier space with frequency shift.
   * The Gaussian function is given by:
   *   exp( -( (xx**2 + yy**2 + zz**2) * (2 * PI * sigma ** 2) ) )
   *
   * The resulting complex values are stored in `d_x`.
   *
   * @param d_x Pointer to the output array (`cufftComplex`) where the Gaussian values will be stored.
   * @param sigma Standard deviation of the Gaussian distribution.
   * @param dimension Total dimension size of the data cube along each dimension (assuming cubic).
   * @param n_gpus Total number of GPUs being used.
   * @param gpuId GPU identifier (0-indexed) indicating the current GPU's position.
   *
   * @note This function assumes that `d_x` is preallocated and has dimensions aligned such that `dimension / n_gpus` is the slowest dimension.
   *       It computes the Gaussian distribution along the x, y, and z dimensions of the 3D volume, distributed across multiple GPUs,
   *       with frequency shift applied to the output.
   * 
   * @reference This function is inspired by the principles described in:
   * High performance multi-dimensional (2D/3D) FFT-Shift implementation on Graphics Processing Units (GPUs)
   * Author(s): Marwan Abdellah; Salah Saleh; Ayman Eldeib; Amr Shaarawi
   * Published in: 2012 Cairo International Biomedical Engineering Conference (CIBEC) 
   * URL: https://ieeexplore.ieee.org/abstract/document/6473306
   */

    const float PI_F = 3.141592654f;

    const int ndimx = dimension/n_gpus; //slowest dimension
    const int ndimy = dimension;
    const int ndimz = dimension;
    const size_t perGPUDim = (size_t) ndimx * ndimy * ndimz;

    // it was like that 
    // const float a = 1.0;
    // const float delta = (2.0*a)/(dimension-1);
    const float a     = 1.0;
    const float wc    = (float) dimension/(4*a); //nyquist
    const float delta = (2.0*wc)/(dimension);


    // Transformations Equations
    const size_t sVolume= (size_t)dimension*dimension*dimension;
    const size_t sSlice = (size_t)dimension*dimension;
    const size_t sLine = (size_t)dimension;
    const size_t sEq1 = (sVolume + sSlice + sLine)/2;
    const size_t sEq2 = (sVolume + sSlice - sLine)/2;
    const size_t sEq3 = (sVolume - sSlice + sLine)/2;
    const size_t sEq4 = (sVolume - sSlice - sLine)/2;     


    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    const size_t globalIndex = index + gpuId*perGPUDim;

    // map 1D index to 3D
    index3 i, ii;
    float x, y, z;
  
    i = get3Dindex(globalIndex, ndimy, ndimz);
  
    // https://ieeexplore.ieee.org/abstract/document/6473306  

    if (index<perGPUDim){
      if ( i.ix < dimension/2){
        if (i.iy < dimension/2){
          if (i.iz < dimension/2){ // Q1 
            // data[index] <-> data[index + sEq1]
            ii = get3Dindex(globalIndex + sEq1, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;
          
            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp((-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z)) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }else{  // Q2
            ii = get3Dindex(globalIndex + sEq2, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;
              
            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp((-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z)) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }
        }else{ // i.iy > dimension/2
          if (i.iz < dimension/2){ // Q3
            ii = get3Dindex(globalIndex + sEq3, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;

            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp((-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z)) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }else{  // Q4
            ii = get3Dindex(globalIndex + sEq4, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;

            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp( (-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z)) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }
        }
      }else{// ix > dimension/2 
        if (i.iy < dimension/2){
          if (i.iz < dimension/2){ // Q1
            // 
            ii = get3Dindex(globalIndex - sEq4, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;

            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp( (-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z) ) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }else{  // Q2
            ii = get3Dindex(globalIndex - sEq3, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;

            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp( (-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z) ) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }
        }else{ // i.iy > dimension/2
          if (i.iz < dimension/2){ // Q3
            ii = get3Dindex(globalIndex - sEq2, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;

            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp( (-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z) ) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }else{  // Q4
            ii = get3Dindex(globalIndex - sEq1, ndimy, ndimz);
            x = -wc + ii.ix * delta;
            y = -wc + ii.iy * delta;
            z = -wc + ii.iz * delta;

            // d_x[index].x = exp( -( (x * x + y * y + z * z) * ( 2 * PI_F * sigma * sigma) ) );
            d_x[index].x = exp( (-1/(2*PI_F*sigma*sigma))*(x*x + y*y + z*z) ) * (1/(sigma*sqrt(2*PI_F)));
            d_x[index].y = 0.0f;
          }
        }
      }
    }
  }

  
  __global__ void set_difference(cufftComplex *a,
                                 cufftComplex *b,
                                 cufftComplex *c,
                                 float alpha,
                                 float beta,
                                 int dimension){
    /**
     * @brief Computes the difference between two `cufftComplex` arrays and stores the result in a third `cufftComplex` array.
     *
     * This kernel computes the difference between the elements of the input `cufftComplex` arrays `b` and `c`, 
     * scales them by the factors `alpha` and `beta` respectively, and stores the result in the output `cufftComplex` array `a`.
     *
     * The operation performed is:
     *     a = alpha * b - beta * c
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array.
     * @param c Pointer to the second input `cufftComplex` array.
     * @param alpha Scaling factor for the first input array `b`.
     * @param beta Scaling factor for the second input array `c`.
     * @param dimension The size of each dimension of the 3D arrays.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */

    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension) {
      a[index].x = alpha*b[index].x - beta*c[index].x;
      a[index].y = alpha*b[index].y - beta*c[index].y;
    }
  }





  __global__ void set_difference_mgpu(cufftComplex *a,
                                      cufftComplex *b,
                                      cufftComplex *c,
                                      float alpha,
                                      float beta,
                                      size_t perGPUDim){
    /**
     * @brief Computes the difference between two `cufftComplex` arrays for multi-GPU and stores the result in a third `cufftComplex` array.
     *
     * This kernel computes the difference between the elements of the input `cufftComplex` arrays `b` and `c`, 
     * scales them by the factors `alpha` and `beta` respectively, and stores the result in the output `cufftComplex` array `a`.
     *
     * The operation performed is:
     *     a = alpha * b - beta * c
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array.
     * @param c Pointer to the second input `cufftComplex` array.
     * @param alpha Scaling factor for the first input array `b`.
     * @param beta Scaling factor for the second input array `c`.
     * @param perGPUDim The number of elements each GPU handles.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `perGPUDim`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */

    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < perGPUDim){
      a[index].x = alpha*b[index].x - beta*c[index].x;
      a[index].y = alpha*b[index].y - beta*c[index].y;
    }
  }

 

  __global__ void multiply(cufftComplex *a,
                           cufftComplex *b,
                           cufftComplex *c,
                           int dimension){
    /**
     * @brief Computes the element-wise multiplication of two `cufftComplex` arrays and stores the result in a third `cufftComplex` array.
     *
     * This kernel performs pointwise complex multiplication between the elements of the input `cufftComplex` arrays `b` and `c`,
     * and stores the result in the output `cufftComplex` array `a`.
     *
     * The operation performed is:
     *     a = b * c
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array.
     * @param c Pointer to the second input `cufftComplex` array.
     * @param dimension The size of each dimension of the 3D arrays.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */
    const int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      // full implementation of pointwise complex multiplication
      a[index].x = b[index].x*c[index].x - b[index].y*c[index].y;
      a[index].y = b[index].x*c[index].y + b[index].y*c[index].x;  
    }
  }

  __global__ void multiply_legacy(cufftComplex *a,
                           cufftComplex *b,
                           cufftComplex *c,
                           int dimension){
   /**
     * @brief Computes the element-wise legacy multiplication of two `cufftComplex` arrays and stores the result in a third `cufftComplex` array.
     *
     * This kernel performs a legacy multiplication between the elements of the input `cufftComplex` arrays `b` and `c`,
     * and stores the result in the output `cufftComplex` array `a`.
     *
     *
     * The legacy multiplication of two complex numbers is defined as:
     *     (real(b) + i*imag(b)) (*_legacy) (real(c) + i*imag(c)) = (real(b)*real(c)) + i*(imag(b)*imag(c))
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array.
     * @param c Pointer to the second input `cufftComplex` array.
     * @param dimension The size of each dimension of the 3D arrays.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     *       Also, notice that this is included here only for retrocompatibility reasons. Will be depracted soon.
     */
    const int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      // legacy implementation 
      a[index].x = b[index].x * c[index].x;
      a[index].y = b[index].y * c[index].y;  
    }
  }


  __global__ void multiply_rc(cufftComplex *a,
                              cufftComplex *b,
                              cufftComplex *c,
                              int dimension){
    /**
     * @brief Computes the element-wise multiplication of two `cufftComplex` arrays using only the real component of the first array and stores the result in a third `cufftComplex` array.
     *
     * This kernel performs pointwise multiplication where only the real component of the input `cufftComplex` array `b` is used.
     * The real and imaginary parts of `b` are multiplied with the corresponding parts of the input `cufftComplex` array `c`,
     * and the results are stored in the output `cufftComplex` array `a`.
     *
     * The operation performed is:
     *     a = real(b) * c
     *
     * Specifically:
     *     a[index].x = b[index].x * c[index].x
     *     a[index].y = b[index].x * c[index].y
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array (only the real component is used for multiplication).
     * @param c Pointer to the second input `cufftComplex` array.
     * @param dimension The size of each dimension of the 3D arrays.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */
    const int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy * dimension + threadIDz * dimension * dimension;

    if (threadIDx < dimension && threadIDy < dimension && threadIDz < dimension) {
      a[index].x = b[index].x*c[index].x;
      a[index].y = b[index].x*c[index].y;  
    }
  }


  __global__ void multiply_mgpu(cufftComplex *a,
                                cufftComplex *b,
                                cufftComplex *c,
                                size_t perGPUDim){
    /**
     * @brief Computes the element-wise multiplication of two `cufftComplex` arrays and stores the result in a third `cufftComplex` array, designed for multi-GPU execution.
     *
     * This kernel performs pointwise complex multiplication between the elements of the input `cufftComplex` arrays `b` and `c`,
     * and stores the result in the output `cufftComplex` array `a`.
     *
     * The operation performed is:
     *     a = b * c
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array.
     * @param c Pointer to the second input `cufftComplex` array.
     * @param perGPUDim The size of the array portion assigned to each GPU.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `perGPUDim`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;  

    if (index < perGPUDim){
      // full implementation
      a[index].x = b[index].x*c[index].x - b[index].y*c[index].y;
      a[index].y = b[index].x*c[index].y + b[index].y*c[index].x;  
    }
  }

  __global__ void multiply_legacy_mgpu(cufftComplex *a,
                                       cufftComplex *b,
                                       cufftComplex *c,
                                       size_t perGPUDim){
   /**
     * @brief Computes the element-wise legacy multiplication of two `cufftComplex` arrays and stores the result in a third `cufftComplex` array, designed for multi-GPU execution.
     *
     * This kernel performs legacy pointwise multiplication between the elements of the input `cufftComplex` arrays `b` and `c`,
     * and stores the result in the output `cufftComplex` array `a`.
     *
     * The legacy multiplication of two complex numbers is defined as:
     *     (real(b) + i*imag(b)) (*_legacy) (real(c) + i*imag(c)) = (real(b)*real(c)) + i*(imag(b)*imag(c))
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array.
     * @param c Pointer to the second input `cufftComplex` array.
     * @param perGPUDim The size of the array portion assigned to each GPU.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `perGPUDim`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;  

    if (index < perGPUDim){
      // legacy implementation 
      a[index].x = b[index].x*c[index].x;
      a[index].y = b[index].y*c[index].y;
    }
  }


  __global__ void multiply_rc_mgpu(cufftComplex *a,
                                   cufftComplex *b,
                                   cufftComplex *c,
                                   size_t perGPUDim){
    /**
     * @brief Computes the element-wise multiplication of two `cufftComplex` arrays using only the real component of the first array and stores the result in a third `cufftComplex` array, designed for multi-GPU execution.
     *
     * This kernel performs pointwise multiplication where only the real component of the input `cufftComplex` array `b` is used.
     * The real and imaginary parts of `b` are multiplied with the corresponding parts of the input `cufftComplex` array `c`,
     * and the results are stored in the output `cufftComplex` array `a`.
     *
     * The operation performed is:
     *     a = real(b) * c
     *
     * Specifically:
     *     a[index].x = b[index].x * c[index].x
     *     a[index].y = b[index].x * c[index].y
     *
     * @param a Pointer to the output `cufftComplex` array where the result will be stored.
     * @param b Pointer to the first input `cufftComplex` array (only the real component is used for multiplication).
     * @param c Pointer to the second input `cufftComplex` array.
     * @param perGPUDim The size of the array portion assigned to each GPU.
     *
     * @note This kernel assumes that `a`, `b`, and `c` are all arrays of size `perGPUDim`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;  

    if (index < perGPUDim){
      a[index].x = b[index].x*c[index].x;
      a[index].y = b[index].x*c[index].y;
    }
  }


  __global__ void normalize(cufftComplex *a,
                            int dimension){
    /**
     * @brief Normalizes the elements of a `cufftComplex` array by dividing each element by the cube of the dimension size.
     *
     * This kernel normalizes each element of the input `cufftComplex` array `a` by dividing the real and imaginary parts
     * by the cube of the dimension size.
     *
     * The operation performed is:
     *     a = a / (dimension^3)
     *
     * Specifically:
     *     a[index].x = a[index].x / (dimension^3)
     *     a[index].y = a[index].y / (dimension^3)
     *
     * @param a Pointer to the input `cufftComplex` array that will be normalized.
     * @param dimension The size of each dimension of the 3D array.
     *
     * @note This kernel assumes that `a` is an array of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the array is properly allocated and initialized.
     */
    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension) {
      a[index].x = a[index].x/((float) (dimension*dimension*dimension));
      a[index].y = a[index].y/((float) (dimension*dimension*dimension));
    }
  }

  __global__ void normalize_mgpu(cufftComplex *a,
                                 size_t totalDim,
                                 size_t perGPUDim){
   /**
   * @brief Normalizes the elements of a `cufftComplex` array by dividing each element by the total dimension size, designed for multi-GPU execution.
   *
   * This kernel normalizes each element of the input `cufftComplex` array `a` by dividing the real and imaginary parts
   * by the total dimension size `totalDim`. It is designed for multi-GPU execution, where each GPU handles a portion
   * of the total array specified by `perGPUDim`.
   *
   * The operation performed is:
   *     a = a / totalDim
   *
   * Specifically:
   *     a[index].x = a[index].x / totalDim
   *     a[index].y = a[index].y / totalDim
   *
   * @param a Pointer to the input `cufftComplex` array that will be normalized.
   * @param totalDim The total dimension size used for normalization, typically `dimension^3`.
   * @param perGPUDim The size of the array portion assigned to each GPU.
   *
   * @note This kernel assumes that `a` is an array of size `perGPUDim`.
   *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
   */

    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index<perGPUDim){
      a[index].x = a[index].x/((float) (totalDim));
      a[index].y = a[index].y/((float) (totalDim));
    }
  }



  __global__ void absolute_value(cufftComplex *a,
                                 cufftComplex *b,
                                 int dimension){
    /**
     * @brief Computes the absolute value of each element in a `cufftComplex` array `b` and stores the result in array `a`.
     *
     * This kernel computes the absolute value of each element in the input `cufftComplex` array `b` by calculating the
     * magnitude of the complex number. The resulting real part is stored in array `a`, while the imaginary part is set to zero.
     *
     * Specifically:
     *     a[index].x = sqrt(b[index].x^2 + b[index].y^2)
     *     a[index].y = 0.0
     *
     * @param a Pointer to the output `cufftComplex` array where the absolute values will be stored.
     * @param b Pointer to the input `cufftComplex` array whose absolute values will be computed.
     * @param dimension The size of each dimension of the 3D arrays.
     *
     * @note This kernel assumes that `a` and `b` are arrays of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */

    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    cufftComplex absv;
    
    if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
      absv.x = sqrtf(powf(b[index].x,2.0) + powf(b[index].y,2.0));
      // absv.x = sqrtf(powf(fabs(b[index].x),2.0) + powf(fabs(b[index].y),2.0));
      // absv.x = b[index].x; 
      absv.y = 0.0f;
      a[index].x = absv.x;
      a[index].y = absv.y;
    }
  }


  __global__ void real_part(cufftComplex *a,
                            cufftComplex *b,
                            int dimension){
    /**
     * @brief Extracts the real part of each element in a `cufftComplex` array `b` and stores it in array `a`.
     *
     * This kernel extracts the real part of each element in the input `cufftComplex` array `b`. The extracted real part 
     * is stored in array `a`, while the imaginary part is set to zero.
     *
     * Specifically:
     *     a[index].x = b[index].x
     *     a[index].y = 0.0
     *
     * @param a Pointer to the output `cufftComplex` array where the real parts will be stored.
     * @param b Pointer to the input `cufftComplex` array from which the real parts will be extracted.
     * @param dimension The size of each dimension of the 3D arrays.
     *
     * @note This kernel assumes that `a` and `b` are arrays of size `dimension^3`.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */
    const int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t index = threadIDx + threadIDy * dimension + threadIDz * dimension * dimension;

    cufftComplex absv;
    
    if (threadIDx < dimension && threadIDy < dimension && threadIDz < dimension){
      absv.x = b[index].x;
      absv.y = 0.0f;

      a[index].x = absv.x;
      a[index].y = absv.y;
    }
  }



  __global__ void absolute_value_mgpu(cufftComplex *a,
                                      cufftComplex *b,
                                      size_t totalDim,
                                      size_t perGPUDim){
    /**
     * @brief Computes the absolute value of each element in a `cufftComplex` array `b` and stores the result in array `a`.
     *
     * This kernel computes the absolute value of each element in the input `cufftComplex` array `b` by calculating the
     * magnitude of the complex number. The resulting real part is stored in array `a`, while the imaginary part is set to zero.
     *
     * Specifically:
     *     a[index].x = sqrt(fabs(b[index].x)^2 + fabs(b[index].y)^2)
     *     a[index].y = 0.0
     *
     * @param a Pointer to the output `cufftComplex` array where the absolute values will be stored.
     * @param b Pointer to the input `cufftComplex` array whose absolute values will be computed.
     * @param totalDim Total size of the complex arrays `a` and `b`.
     * @param perGPUDim Size of the segment of `a` and `b` processed by this GPU.
     *
     * @note This kernel assumes that `a` and `b` are arrays where `totalDim` is the total number of elements,
     *       and `perGPUDim` is the number of elements processed by each GPU.
     *       It is the caller's responsibility to ensure that the arrays are properly allocated and initialized.
     */

    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex absv;
    
    if (index < perGPUDim){
      // absv.x = sqrtf( powf(b[index].x, 2) + powf(b[index].y,2));
      absv.x = sqrtf(powf(b[index].x, 2.0) + powf(b[index].y,2.0));
      absv.y = 0.0f;

      a[index].x = absv.x;
      a[index].y = absv.y; 
    }
  }


 

  __global__ void update_support_sw(uint8_t* supp,
                                    cufftComplex *iter,
                                    float threshold,
                                    float globalMax,
                                    float globalMin,
                                    size_t idxMax,
                                    size_t idxMin,
                                    size_t dimension,
                                    bool which){
    /**
     * @brief Updates a support array based on a threshold comparison with the magnitude of complex numbers.
     *
     * This kernel updates the support array `supp` based on a threshold comparison with the magnitude of complex numbers 
     * in the `iter` array. Two modes are supported:
     * - `which=true`: Each element in `supp` is set to 1 if the magnitude of `iter` at that index is greater than a 
     *                 percentage (`threshold/100`) of the maximum magnitude in `iter` (`maxv`).
     * - `which=false`: Each element in `supp` is set to 1 if the normalized magnitude (relative to `globalMax` and `globalMin`)
     *                  of `iter` at that index is greater than a percentage (`threshold/100`) of the range (`globalMax-globalMin`).
     *
     * @param supp Pointer to the support array to be updated. Values are set to 1 or 0 based on the threshold comparison.
     * @param iter Pointer to the input `cufftComplex` array containing complex numbers whose magnitudes are evaluated.
     * @param threshold Threshold value used for the comparison.
     * @param globalMax Maximum magnitude value in the `iter` array.
     * @param globalMin Minimum magnitude value in the `iter` array.
     * @param idxMax Index of the maximum magnitude value in the `iter` array.
     * @param idxMin Index of the minimum magnitude value in the `iter` array.
     * @param dimension Size of each dimension of the 3D arrays.
     * @param which Boolean flag indicating the mode of operation: true for comparison against `maxv`, false for normalized comparison.
     *
     * @note This kernel assumes that `sup` and `iter` arrays in which `dimension` is their total number of elements. Also,
     *       it assumes that everything is correctly allocated and initialized. It is the caller's responsibility to ensure that
     *       the arrays are properly allocated and initialized.
     */

    const int threadIDx = blockIdx.x*blockDim.x + threadIdx.x;
    const int threadIDy = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadIDz = blockIdx.z*blockDim.z + threadIdx.z;
    const size_t index = threadIDx + threadIDy*dimension + threadIDz*dimension*dimension;

    if(which){
      if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
        float maxv = sqrtf(powf(fabs(iter[idxMax].x),2.0) + powf(fabs(iter[idxMax].y),2.0));
        // float minv = sqrt(powf(fabs(iter[idxMin].x),2.0) + powf(fabs(iter[idxMin].y),2.0));
        float val = sqrtf(powf(fabs(iter[index].x),2.0) + powf(fabs(iter[index].y),2.0));
  
        if (val>(threshold/100.)*maxv){
          supp[index] = 1;
        }else{
          supp[index] = 0;
        }
      }

    }else{
      if (threadIDx<dimension && threadIDy<dimension && threadIDz<dimension){
        float val = sqrtf( powf( fabs(iter[index].x),2) + powf( fabs(iter[index].y),2));
        // float min = sqrtf(powf(fabs(iter[index].x),2) + powf(fabs(iter[index].y),2));
  
        // if ( val > (threshold/100.) * globalMax ){
        if (((val-globalMin)/(globalMax-globalMin)) > (threshold/100.0f)){
          supp[index] = 1;
        }else{
          supp[index] = 0;
        }
      }
    }
  }
  
  __global__ void update_support_sw_mgpu(uint8_t* supp,
                                         cufftComplex *iter,
                                         float threshold,
                                         float globalMax,
                                         float globalMin,
                                         size_t perGPUDim){
    /**
     * @brief Updates a support array `supp` based on the magnitude of elements in the `iter` array for multi-GPU setups.
     *
     * This kernel updates the support array `supp` based on the magnitude of elements in the `iter` array.
     * It uses a global threshold to determine if the magnitude of each element in `iter` exceeds a certain fraction
     * of the maximum magnitude (`globalMax - globalMin`). This is computed per GPU thread based on the `perGPUDim`.
     *
     * @param supp Pointer to the uint8_t array `supp` where the support information will be stored.
     * @param iter Pointer to the input `cufftComplex` array whose magnitude determines the support.
     * @param threshold Threshold percentage used to determine support activation.
     * @param globalMax Global maximum magnitude value for thresholding.
     * @param globalMin Global minimum magnitude value for thresholding.
     * @param perGPUDim Number of elements each GPU thread processes.
     *
     * @note This kernel assumes that `supp` and `iter` are properly allocated and initialized.
     *       It operates on arrays of size `perGPUDim`, and `globalMax` and `globalMin` provide the overall
     *       magnitude range for the thresholding calculation.
     */
    const size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index<perGPUDim){
      float val = sqrtf(powf(fabs(iter[index].x),2.0) + powf(fabs(iter[index].y),2.0));
      // float max = iter[index].x;
  
      // if ( max > (threshold/100.) * globalMax ){
      if (((val-globalMin)/(globalMax-globalMin)) > (threshold/100.0f)){
        supp[ index ] = 1;
      }else{
        supp[ index ] = 0;
      }
    }
  }



 
  __global__ void fftshift(void* input, int N, int dtype){
    /**
     * @brief Performs an in-place FFT shift operation on a 3D array of data.
     *
     * This CUDA kernel function performs an in-place FFT shift operation on a 3D array of data represented by `input`.
     * The operation reorganizes the data in the frequency domain to shift the zero-frequency components to the center of the spectrum.
     * It supports shifting for `cufftComplex` (complex), `uint8_t` (byte), and `float` (real) data types based on the parameter `dtype`.
     *
     * @param input Pointer to the 3D array (`cufftComplex`, `uint8_t`, or `float`) to be shifted.
     * @param N Size of one dimension of the 3D array (`N x N x N`).
     * @param dtype Integer indicating the data type: `SSC_DTYPE_CUFFTCOMPLEX` for `cufftComplex`, `SSC_DTYPE_BYTE` for `uint8_t`, or `SSC_DTYPE_FLOAT` for `float`.
     *
     * @note This function implements the FFT shift algorithm for CUDA, as adapted from the original implementation available at:
     *       https://github.com/marwan-abdellah/cufftShift/blob/master/Src/CUDA/Kernels/in-place/cufftShift_3D_IP.cu
     *       Ensure that `input` is properly allocated and initialized as a 3D array of the specified data type before invoking this kernel.
     */

    // 3D Volume & 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N*N;
    int sVolume = N*N*N;

    // Transformations Equations
    int sEq1 = (sVolume + sSlice + sLine)/2;
    int sEq2 = (sVolume + sSlice - sLine)/2;
    int sEq3 = (sVolume - sSlice + sLine)/2;
    int sEq4 = (sVolume - sSlice - sLine)/2;

    // Thread
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;
    int zThreadIdx = threadIdx.z;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int blockVolume = blockDim.z;

    // Thread Index 2D
    int xIndex = blockIdx.x*blockWidth + xThreadIdx;
    int yIndex = blockIdx.y*blockHeight + yThreadIdx;
    int zIndex = blockIdx.z*blockVolume + zThreadIdx;

    // Thread Index Converted into 1D Index
    int index = (zIndex*sSlice) + (yIndex*sLine) + xIndex;
  
    if(dtype==SSC_DTYPE_CUFFTCOMPLEX){
      cufftComplex regTemp;
      cufftComplex*data = (cufftComplex *) input;

      if (zIndex<N/2){
        if (xIndex<N/2){
          if (yIndex<N/2){
            regTemp = data[index];

            // First Quad
            data[index] = data[index+sEq1];

            // Fourth Quad
            data[index+sEq1] = regTemp;
          }else{
            regTemp = data[index];

            // Third Quad
            data[index] = data[index+sEq3];
 
            // Second Quad
            data[index+sEq3] = regTemp;
          }
        }else{
          if (yIndex<N/2){
            regTemp = data[index];

            // Second Quad
            data[index] = data[index+sEq2];

            // Third Quad
            data[index+sEq2] = regTemp;
          }else{
            regTemp = data[index];

            // Fourth Quad
            data[index] = data[index+sEq4];
  
            // First Quad
            data[index+sEq4] = regTemp;
          }
        }
      }
    }

    if (dtype==SSC_DTYPE_BYTE){
      uint8_t regTemp;
      uint8_t *data = (uint8_t*) input;
    
      if (zIndex<N/2){
        if (xIndex<N/2){
          if (yIndex<N/2){
            regTemp = data[index];

            // First Quad
            data[index] = data[index+sEq1];

            // Fourth Quad
            data[index+sEq1] = regTemp;
          }else{
            regTemp = data[index];

            // Third Quad
            data[index] = data[index+sEq3];

            // Second Quad
            data[index+sEq3] = regTemp;
          }
        }else{
          if (yIndex<N/2){
            regTemp = data[index];

            // Second Quad
            data[index] = data[index+sEq2];

            // Third Quad
            data[index+sEq2] = regTemp;
          }else{
            regTemp = data[index];

            // Fourth Quad
            data[index] = data[index+sEq4];

            // First Quad
            data[index+sEq4] = regTemp;
          }
        }
      }

    }

    if (dtype==SSC_DTYPE_FLOAT){
      float regTemp;
      float *data = (float *) input;

      if (zIndex<N/2){
        if (xIndex<N/2){
          if (yIndex<N/2){
            regTemp = data[index];

            // First Quad
            data[index] = data[index+sEq1];

            // Fourth Quad
            data[index+sEq1] = regTemp;
          }else{
            regTemp = data[index];

            // Third Quad
            data[index] = data[index+sEq3];

            // Second Quad
            data[index+sEq3] = regTemp;
          }
        }else{
          if (yIndex<N/2){
            regTemp = data[index];

            // Second Quad
            data[index] = data[index+sEq2];

            // Third Quad
            data[index+sEq2] = regTemp;
          }else{
            regTemp = data[index];

            // Fourth Quad
            data[index] = data[index+sEq4];

            // First Quad
            data[index+sEq4] = regTemp;
          }
        }
      }
    }
  }

} // extern "C"
