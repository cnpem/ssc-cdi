#include <iostream>
#include <math.h>
#include <stdint.h>  // For uint8_t

#include "pwcdi.h"
#include "compute.h"
#include "gpus.h"

extern "C" {
  void alloc_workspace(ssc_pwcdi_plan *workspace,
                       ssc_pwcdi_params *params,
                       int *gpus,
                       int ngpu){
  /**
   * @brief Allocates memory for the workspace and sets up GPU configurations for the ssc-pwcdi plan.
   *
   * This function allocates and initializes the necessary memory and GPU resources required 
   * for the ssc_pwcdi_plan plan. It sets up the3D workspace, allocates memory for various data 
   * arrays on the GPU, and creates a CUDA plan  for FFT operations using cuFFT. 
   *
   * The allocation process includes setting up workspace parameters, creating a cuFFT plan, 
   * allocating device memory for support and measured data, and setting timing flags for performance profiling.
   *
   * @param workspace Pointer to the `ssc_pwcdi_plan` structure, which will store the allocated workspace 
   *        information, including device memory allocations and FFT plan.
   * @param params Pointer to the `ssc_pwcdi_params` structure containing configuration parameters 
   *        such as grid size (N), GPU memory mapping options, and timing settings.
   * @param gpus Pointer to an array of GPU IDs used for multi-GPU processing.
   * @param ngpu The number of GPUs available for the computation. It can be 1 or more.
   *
   * @note 
   * - The function assumes that the workspace and parameter structures are properly initialized before calling.
   * - Memory for support arrays (`d_support`) and signal data (`d_x`, `d_y`) is allocated on the GPU.
   * - cuFFT plan is created to optimize 3D FFT operations on the allocated workspace.
   * - The timing flag is set based on the `params->timing` value, enabling performance profiling if needed.
   * 
   * @pre The number of available GPUs should be checked and provided in the `gpus` array.
   * @post After successful execution, the workspace will be fully allocated and ready for use in the 
   *       ssc-pwcdi algorithm.
   */

                                 
    const char *BSTATE = "before allocation";
    const char *ASTATE = "after allocation";
    cudaError_t hostalloc_status;
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    size_t N               = params->N;
    workspace->dimension   = params->N;
    workspace->nvoxels = (size_t) (workspace->dimension)*(workspace->dimension)*(workspace->dimension);
    

    // set timers
    if (params->timing==1){
      workspace->timing = true;
    }else{
      workspace->timing = false;
    } 
    
    // allocate gpu workspace
    if (ngpu==1){ 
      workspace->gpus = (ssc_gpus *)malloc(sizeof(ssc_gpus)); 
      workspace->gpus->gpus = (int *)malloc( sizeof(int)*ngpu);
      workspace->gpus->ngpus = ngpu;
      for (int k=0; k<ngpu; k++){
        workspace->gpus->gpus[k] = gpus[k];
      }
    
      // debug
      fprintf(stdout,"ssc-cdi: Available GPUs at this node: %d\n",deviceCount);
      fprintf(stdout,"ssc-cdi: Distributing data through GPUs: ");
      
      // allocate GPU
      for (int k=0; k < workspace->gpus->ngpus; k++)
        fprintf(stdout,"%d ",workspace->gpus->gpus[k]);
      fprintf(stdout,"\n");
          
      // profile GPU (before)
      ssc_gpus_get_info(workspace->gpus->ngpus, workspace->gpus->gpus, BSTATE);

      checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[0]));
          
      // create PLAN and estimate its size 
      checkCudaErrors(cufftCreate(&workspace->plan_C2C));
      size_t worksize[workspace->gpus->ngpus];  
      cufftGetSize3d(workspace->plan_C2C,
                     workspace->dimension,
                     workspace->dimension,
                     workspace->dimension, 
                     CUFFT_C2C, 
                     worksize);
      
 
      // make plan
      checkCudaErrors(cufftMakePlan3d(workspace->plan_C2C,
                                      workspace->dimension,
                                      workspace->dimension,
                                      workspace->dimension,
                                      CUFFT_C2C, worksize));
  
      // debug     
      printf("ssc-cdi: plan created.\n");
      printf("ssc-cdi: plan size: %lf GiB.\n", (double) worksize[0]/ (1024*1024*1024));
  

      // allocate d_support 
      if (params->map_d_support==false){
        // allocate d_support in VRAM GPU memory directly
        checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_support, sizeof(uint8_t)*pow(workspace->dimension, 3)));
      }else if (params->map_d_support==true){
        // allocate d_support as mapped host memory 
        checkCudaErrors(cudaMallocHost((void**) &workspace->sgpu.d_support_host, sizeof(uint8_t)*pow(workspace->dimension, 3),cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&workspace->sgpu.d_support, (void *)workspace->sgpu.d_support_host, 0));    
      }


      // malloc other variables
      checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_x, sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_y, sizeof(cufftComplex)*pow(workspace->dimension, 3))); 


      // allocate d_signal 
      if (params->map_d_signal==false){
        // malloc d_signal in device memory directly 
        checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_signal, sizeof(float)*pow(workspace->dimension, 3)));
      }else if (params->map_d_signal==true){
        // malloc d_signal in mapped pinned host memory, and pass a pointer to the device only
        checkCudaErrors(cudaMallocHost((void**) &workspace->sgpu.d_signal_host, sizeof(float)*pow(workspace->dimension, 3),cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&workspace->sgpu.d_signal, (void *)workspace->sgpu.d_signal_host, 0));    
      }

      // allocate d_x_swap (host) or d_gaussian (device). If swap_d_x is false, then d_gaussian is an extra pointer in VRAM GPU 
      // memory. Otherwise, d_x will be stored in d_x_swap during the shrinkWrap operation.
      if (params->swap_d_x==false){
        checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_gaussian, sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      }else if(params->swap_d_x==true){
        checkCudaErrors(cudaMallocHost((void**)&workspace->sgpu.d_x_swap,sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      }

      // malloc err swap variables 
      if (params->err_type==ITER_DIFF){
        checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_x_lasterr, sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      }
      
      // profile GPU (after)
      ssc_gpus_get_info(workspace->gpus->ngpus, workspace->gpus->gpus, ASTATE);


    // multi GPU case  
    }else if (ngpu>1){

      // allocate gpus workspace
      workspace->gpus = (ssc_gpus*) malloc(sizeof(ssc_gpus));
      workspace->gpus->gpus = (int*) malloc(sizeof(int)*ngpu);
      workspace->gpus->ngpus = ngpu;
      for(int k=0; k < ngpu; k++){
          workspace->gpus->gpus[k] = gpus[k];
      }
      
      // profile GPUs (before)
      fprintf(stdout,"ssc-cdi: Available GPUs at this node: %d\n",deviceCount);
      fprintf(stdout,"ssc-cdi: Distributing data through GPUs: ");
      for (int k=0; k < workspace->gpus->ngpus; k++){
        fprintf(stdout,"%d ",workspace->gpus->gpus[k]);
      }
      fprintf(stdout,"\n");
      ssc_gpus_get_info(workspace->gpus->ngpus, workspace->gpus->gpus, BSTATE);

      // allocate workspace
      int n_gpus = workspace->gpus->ngpus;

      // create cuda streams
      workspace->gpus->streams = new cudaStream_t[n_gpus];
      for (int i=0; i<n_gpus; i++){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
        checkCudaErrors(cudaStreamCreateWithFlags(&(workspace->gpus->streams[i]), cudaStreamNonBlocking));
     }

      // allocate d_signal 
      if (params->map_d_signal==true){
        // allocate d_signal pointer in mapped host memory
        hostalloc_status = cudaMallocHost((void**)&workspace->mgpu.d_signal,n_gpus*sizeof(float*));

        // allocate d_signal_host as pinned host memory. After that, 
        // allocate a device pointer for d_signal containing a reference to d_signal_host only. 
        hostalloc_status = cudaMallocHost((void **)&workspace->mgpu.d_signal_host, n_gpus*sizeof(float*));
        for (int i=0; i<n_gpus; i++){
          hostalloc_status = cudaMallocHost((void **)&workspace->mgpu.d_signal_host[i], (size_t) (N*N*N*sizeof(float)/n_gpus), cudaHostAllocMapped);
        }
        for (int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          cudaHostGetDevicePointer((void **)&workspace->mgpu.d_signal[i], (void *)workspace->mgpu.d_signal_host[i], 0);
        }
        for (int i=0; i<n_gpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
      }else if(params->map_d_signal==false){
        // allocate d_signal directly in VRAM GPU memory
        // hostalloc_status = cudaMalloc((void**)&workspace->mgpu.d_signal, n_gpus*sizeof(float*));
        workspace->mgpu.d_signal = (float**) calloc(n_gpus, sizeof(float*));

        for (int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          hostalloc_status = cudaMalloc((void**)&workspace->mgpu.d_signal[i], (size_t)(N*N*N*sizeof(float)/n_gpus));
        }
        for (int i=0; i<n_gpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
      }


      // allocate d_support related variables
      if (params->map_d_support==true){
 
        // allocate d_support in mapped host memory
        hostalloc_status = cudaMallocHost((void**)&workspace->mgpu.d_support, n_gpus*sizeof(uint8_t*));

        // allocate d_support_host as pinned host memory. After that, 
        // allocate a device pointer for d_support containing a reference to d_support_host only. 
        hostalloc_status = cudaMallocHost((void **)&workspace->mgpu.d_support_host, n_gpus*sizeof(uint8_t*));
        for (int i=0; i<n_gpus; i++){
           hostalloc_status =cudaMallocHost((void **)&workspace->mgpu.d_support_host[i], (size_t) (N*N*N*sizeof(uint8_t)/n_gpus), cudaHostAllocMapped);
        }
        for (int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          cudaHostGetDevicePointer((void **)&workspace->mgpu.d_support[i], (void *)workspace->mgpu.d_support_host[i], 0);
        }
        for (int i=0; i<n_gpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
      }else if(params->map_d_support==false){
        // allocate d_support directly in GPU memory
        hostalloc_status = cudaMallocHost((void**)&workspace->mgpu.d_support, (size_t) (n_gpus*sizeof(uint8_t*)));
        for (int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
          checkCudaErrors(cudaMalloc((void **)&(workspace->mgpu.d_support[i]),(size_t)N*N*N*sizeof(uint8_t)/n_gpus)); 
        }
        for (int i=0; i<n_gpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
      }
      // workspace->mgpu.d_support  = (uint8_t**)calloc(n_gpus, sizeof(uint8_t*));
      // workspace->mgpu.d_gaussian = (cufftComplex**)calloc(n_gpus, sizeof(cufftComplex*));
      // checkCudaErrors(cudaMalloc((void**)&workspace->mgpu.d_support, (size_t) (n_gpus*sizeof(uint8_t*))));
    
 
      //  allocate host swap variable 
      // workspace->host_swap = (cufftComplex*)malloc(N*N*N*sizeof(cufftComplex));
      hostalloc_status = cudaMallocHost((void **)&workspace->host_swap, N*N*N*sizeof(cufftComplex));

      // dynamic allocation: allocate host swap uint8_t variable 
      // workspace->host_swap_byte = (uint8_t*)malloc(N*N*N*sizeof(uint8_t));
      hostalloc_status = cudaMallocHost((void **)&workspace->host_swap_byte,N*N*N*sizeof(uint8_t));

      // allocate d_x_swap (host) or d_gaussian (device). If swap_d_x is false, then d_gaussian is an extra pointer in VRAM GPU 
      // memory. Otherwise, d_x will be stored in d_x_swap during the shrinkWrap operation. 
      if (params->swap_d_x==true){
        hostalloc_status = cudaMallocHost((void **)&workspace->mgpu.d_x_swap, N*N*N*sizeof(cufftComplex));
      }

       
      // CufftXt plan creation
      checkCudaErrors(cufftCreate(&workspace->plan_C2C));
      checkCudaErrors(cufftXtSetGPUs(workspace->plan_C2C, n_gpus, workspace->gpus->gpus));
      size_t worksize[n_gpus];
      cufftGetSize3d(workspace->plan_C2C, N, N, N, CUFFT_C2C, worksize);
      checkCudaErrors(cufftMakePlan3d(workspace->plan_C2C, N, N, N, CUFFT_C2C, worksize));
      // checkCudaErrors(cufftDestroy(workspace->plan_C2C));


      // debug     
      printf("ssc-cdi: plan created.\n");
      for (int i=0; i<n_gpus; i++){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
        printf("ssc-cdi: plan size[%d]: %lf GiB.\n", i, (double) worksize[i]/ (1024*1024*1024));
      }
   
      
      // allocate main variables: d_x, d_y and d_z
      checkCudaErrors(cufftXtMalloc(workspace->plan_C2C, (cudaLibXtDesc**) &workspace->mgpu.d_x, CUFFT_XT_FORMAT_INPLACE)); 
      checkCudaErrors(cufftXtMalloc(workspace->plan_C2C, (cudaLibXtDesc**) &workspace->mgpu.d_y, CUFFT_XT_FORMAT_INPLACE)); 
      
  

      if (params->swap_d_x==false){
        checkCudaErrors(cufftXtMalloc(workspace->plan_C2C, (cudaLibXtDesc**) &workspace->mgpu.d_z, CUFFT_XT_FORMAT_INPLACE)); 
      }

      // allocate d_x_lasterr, if needed
      if (params->err_type==ITER_DIFF){
        checkCudaErrors(cufftXtMalloc(workspace->plan_C2C, (cudaLibXtDesc**) &workspace->mgpu.d_x_lasterr, CUFFT_XT_FORMAT_INPLACE)); 
      }

      // profile GPUs (after allocations) 
      ssc_gpus_get_info(workspace->gpus->ngpus, workspace->gpus->gpus, ASTATE);
    }else{
      fprintf(stdout,"ssc-cdi: ngpu must be either 1, or a positive integer, exiting.\n");
      return;
    }
  }
  
  void free_workspace(ssc_pwcdi_plan *workspace,
                      ssc_pwcdi_params *params){
    /**
     * @brief Frees the allocated memory and releases GPU resources for the ssc_pwcdi_plan variable.
     *
     * This function frees the memory allocated for the workspace and associated data arrays, 
     * as well as releasing GPU resources that were used in the ssc_pwcdi_plan plan.
     * It ensures proper cleanup by releasing both host and device memory depending on the memory mapping settings.
     * The function supports single GPU execution, handling memory deallocation for variables such as support arrays, 
     * measured data, and auxiliary variables.
     *
     * The function performs the following tasks:
     * - Frees memory allocated for the support array (`d_support`), signal data (`d_signal`), and other workspace variables.
     * - Releases GPU memory used for intermediate arrays such as `d_x`, `d_y`, `d_x_swap`, and `d_gaussian`.
     * - Cleans up resources based on the parameters set in the `params` structure, such as whether data is mapped to host memory or not.
     *
     * @param workspace Pointer to the `ssc_pwcdi_plan` structure, which holds the allocated workspace information 
     *        and GPU memory references that need to be freed.
     * @param params Pointer to the `ssc_pwcdi_params` structure containing configuration parameters 
     *        that dictate how memory should be deallocated, such as memory mapping and variable options.
     *
     * @note 
     * - The function checks the `params` structure to determine if the memory is mapped to the host or device and frees accordingly.
     * - It ensures that all GPU memory allocations (including temporary arrays) are properly freed to avoid memory leaks.
     * - If the `params->swap_d_x` or `params->err_type` flags are set, corresponding arrays (`d_x_swap`, `d_x_lasterr`) are also freed.
     * 
     * @pre The function assumes that the workspace was properly allocated using `alloc_workspace`.
     * @post After successful execution, all memory used by the ssc-pwcdi plan is released, and GPU resources are cleaned up.
     */


    printf("ssc-cdi: freeing enviroment.\n");
    if (workspace->gpus->ngpus==1){
      // single GPU case

      // free d_support related variables
      if (params->map_d_support==true){
        // free only the host variable. No need to free the device pointer
        checkCudaErrors(cudaFreeHost(workspace->sgpu.d_support_host));
      }else if (params->map_d_support==false){
        checkCudaErrors(cudaFree(workspace->sgpu.d_support));
      }

      // free d_signal related variables 
      if (params->map_d_signal==true){
        // free only the host variable. No need to free the device pointer
        checkCudaErrors(cudaFreeHost(workspace->sgpu.d_signal_host));
      }else if (params->map_d_signal==false){
        checkCudaErrors(cudaFree(workspace->sgpu.d_signal));
      }

      // free other variables
      checkCudaErrors(cudaFree(workspace->sgpu.d_x));
      checkCudaErrors(cudaFree(workspace->sgpu.d_y)); 

      // free d_x_swap or d_gaussian, accordingly.
      if (params->swap_d_x==false){
        checkCudaErrors(cudaFree(workspace->sgpu.d_gaussian));
      }else if (params->swap_d_x==true){
        checkCudaErrors(cudaFreeHost(workspace->sgpu.d_x_swap));
      }
      
      if (params->err_type==ITER_DIFF){
        checkCudaErrors(cudaFree(workspace->sgpu.d_x_lasterr));
      }

      // destroy FFT plan.
      checkCudaErrors(cufftDestroy(workspace->plan_C2C));
      
      // reset device
      cudaDeviceReset();

    }else if (workspace->gpus->ngpus>1){
      // multi GPU case

      // free host swaps 
      cudaFreeHost(workspace->host_swap);
      cudaFreeHost(workspace->host_swap_byte);

      // free d_x_swap, if necessary 
      if (params->swap_d_x==true){
        cudaFreeHost(workspace->mgpu.d_x_swap);
      }

      // free d_signal related variables
      if (params->map_d_signal==true){
        // free d_signal_host 
        for (int i=0; i<workspace->gpus->ngpus; i++){
          cudaFreeHost(workspace->mgpu.d_signal_host[i]);
        } 
        cudaFreeHost(workspace->mgpu.d_signal_host);
        // free d_signal (no need to explicitely free it, as it is device pointer to mapped memory)
        // cudaFreeHost(workspace->mgpu.d_signal);
      }else if (params->map_d_signal==false){
        for (int i=0; i<workspace->gpus->ngpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          cudaFree(workspace->mgpu.d_signal[i]);
        }
       for (int i=0; i<workspace->gpus->ngpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
        free(workspace->mgpu.d_signal);
      }

      // free the support related variables 
      if (params->map_d_support==true){
        // free d_support_host 
        for (int i=0; i<workspace->gpus->ngpus; i++){
          cudaFreeHost(workspace->mgpu.d_support_host[i]);
        }
        cudaFreeHost(workspace->mgpu.d_support_host);
        // and then we sould free d_support, but there's no need to do it since it is a device pointer to mapped memory.
      }else if(params->map_d_support==false){
        for (int i=0; i<workspace->gpus->ngpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
          checkCudaErrors(cudaFree(workspace->mgpu.d_support[i]));  
        }
        for (int i=0; i<workspace->gpus->ngpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
        cudaFreeHost(workspace->mgpu.d_support);
      }

      // free main variables d_x, d_y and d_z (if needed)
      checkCudaErrors(cufftXtFree(workspace->mgpu.d_x));
      checkCudaErrors(cufftXtFree(workspace->mgpu.d_y)); 
      if (params->swap_d_x==false){
        checkCudaErrors(cufftXtFree(workspace->mgpu.d_z)); 
      }

      // free d_x_lasterr, if it was used
      if (params->err_type==ITER_DIFF){
        checkCudaErrors(cufftXtFree(workspace->mgpu.d_x_lasterr)); 
      }

      // static allocation: plancrecate
      checkCudaErrors(cufftDestroy(workspace->plan_C2C));
       
      // destroy the GPU streams
      for (int i=0; i<workspace->gpus->ngpus; i++){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
        checkCudaErrors(cudaStreamDestroy(workspace->gpus->streams[i]));
      }

      // free gpu streams 
      free(workspace->gpus->streams);

      // free gpus 
      free(workspace->gpus);
      free(workspace->gpus->gpus);
      
      // reset
      cudaDeviceReset();  
    }    
  }

  
  void synth_support_data(uint8_t *support, 
                          int dimension,
                          int p, 
                          float radius,  
                          float x0, 
                          float y0, 
                          float z0){
  /**
   * @brief Generates a 3D support array based on an Lp-norm distance constraint in CPU.
   *
   * This function synthesizes a 3D support array by evaluating each voxel in a cubic grid to determine 
   * whether it lies within a support region defined by an Lp-norm distance constraint. The distance is computed 
   * relative to a specified center `(x0, y0, z0)` with a given radius, and each voxel is assigned a value of 
   * `1` if it falls within the support region and `0` otherwise. The function supports both Euclidean (p=2) 
   * and general Lp-norm distance computations. Data is generated in host device, and should be copied to 
   * GPU afterwards. 
   *
   * The function works as follows:
   * - For `p = 2`, the function computes the squared Euclidean distance between each voxel and the center, 
   *   setting the voxel to `1` if the squared distance is less than `radius^2`.
   * - For general `p`, the function computes the Lp-norm using the formula:
   *    norm = |x - x0|^p + |y - y0|^p + |z - z0|^p
   *   If the norm is less than `radius^p`, the voxel is set to `1`; otherwise, it is set to `0`.
   *
   * The grid is generated by iterating over each voxel in a 3D grid with dimensions specified by `dimension`, 
   * where the coordinates of each voxel are computed relative to the center and the size of the grid.
   *
   * @param support Pointer to the output support array (`uint8_t`), where each element is either `1` (inside) or `0` (outside).
   * @param dimension Size of the 3D grid in each dimension.
   * @param p Power parameter for the Lp-norm computation.
   * @param radius Radius of the support region in Lp-norm.
   * @param x0 X-coordinate of the support center.
   * @param y0 Y-coordinate of the support center.
   * @param z0 Z-coordinate of the support center.
   *
   * @note 
   * - This function is currently implemented on the CPU and will be deprecated in future versions when a GPU implementation is available.
   * - The computation assumes that the support array is preallocated, and it iterates over all voxels in the 3D grid to compute the support region.
   * - This function will be deprecated in the future to generate the data in GPU directly.
   * 
   * @pre The `support` array must be preallocated with a size of `dimension^3`.
   * @post The `support` array is populated with `1` for voxels inside the support region and `0` for those outside.
   */

    if (p==2){
      const float rp = radius*radius;

      for (long int i=0; i<dimension; i++)
        for(long int j=0; j<dimension; j++)
          for(long int k=0; k<dimension; k++) {
            const float a=2.0;
            const float delta = (2.0*a)/(dimension);

            float x = -a + i*delta;
            float y = -a + j*delta;
            float z = -a + k*delta;

            long int idx = k*dimension*dimension + j*dimension + i;
                                                      
            float normx = (x-x0)*(x-x0);
            float normy = (y-y0)*(y-y0);
            float normz = (z-z0)*(z-z0);
            float norm = normx + normy + normz;

            if(norm<rp)
              support[idx] = 1;
            else
              support[idx] = 0;
          }
    }else{
      const float rp = pow(radius, p);

      for (long int i=0; i<dimension; i++)
        for(long int j=0; j<dimension ; j++)
          for(long int k=0; k<dimension; k++) {
            const float a=1.0;
            const float delta = (2.0*a)/(dimension);

            float x = -a*1 + i*delta;
            float y = -a*1 + j*delta;
            float z = -a*1 + k*delta;

            long int idx = k*dimension*dimension + j*dimension + i;
                                                      
            float normx = pow((x-x0), p);
            float normy = pow((y-y0), p);
            float normz = pow((z-z0), p);
            float norm = normx + normy + normz;

            if(norm<rp)
              support[idx] = 1;
            else
              support[idx] = 0;
          }
    }
  }






  void permuted2natural(cudaLibXtDesc* var, cufftHandle& plan_input, size_t dim_size, cufftComplex* host_var){
  /**
   * @brief  Fixes the ordering of a cudaLibXtDesc variable by copying it to host memory and then back to the device. 
   * This function is designed to correct the ordering when the variable's subFormat is CUFFT_XT_FORMAT_INPLACE_SHUFFLED, 
   * which indicates a permuted ordering.
   * 
   * This function fixes the ordering of a cudaLibXtDesc variable by copying it to host memory and then 
   * copying back to the device. It does what cufftXtMemcpy() with the flag CUFFT_COPY_DEVICE_TO_DEVICE should do, 
   * but this feature doesn't seem to be working in CUDA 10.2.89.
   * 
   * @param var Pointer to the cudaLibXtDesc variable to be corrected in terms of ordering.
   * @param plan_input CUFFT plan handle associated with the variable `var`.
   * @param dim_size Size of the variable `var`.
   * @param host_var Pointer to host memory where the data is temporarily stored during the correction process.
   *
   * @note This function assumes that `var` and `host_var` are properly allocated and initialized. It uses CUDA 
   * cufftXtMemcpy() functions for data transfer between host and device.
   * 
   */
   
    // only fix the ordering if the ordering is permuted
    if (var->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED){
      return;
      printf("ssc-cdi: permuted2natural being called for no reason.\n");
    }

    // copy to host
    checkCudaErrors(cufftXtMemcpy(plan_input,
                                  host_var,
                                  var,
                                  CUFFT_COPY_DEVICE_TO_HOST));

    // fix subFormat attribute
    var->subFormat = CUFFT_XT_FORMAT_INPLACE;

    // copy back to device
    checkCudaErrors(cufftXtMemcpy(plan_input,
                                  var,
                                  host_var,
                                  CUFFT_COPY_HOST_TO_DEVICE));

  
  }



  
  void set_input(ssc_pwcdi_plan *workspace,
                 ssc_pwcdi_params *params,
                 float *input,
                 cufftComplex *obj_input){

    /**
     * @brief Initializes the input object and support variables for the computational workspace (ssc_pwcdi_plan).
     *
     * This function sets up the complex object and support variables for the algorithm. It copies the input  
     * to the GPU memory and creates or loads the initial support data, depending on the provided parameters. 
     * The support data can either be synthesized using a specified norm and radius or loaded from pre-existing input.
     * 
     * The function operates in single- or Multi GPU setups.support data is transferred to the GPU memory after creation 
     * or loading. Additionally, timing is performed to track the memory transfer operations if needed. 
     *
     * @param workspace Pointer to the `ssc_pwcdi_plan` structure, which contains information about the computational workspace, 
     *        including GPU memory allocations.
     * @param params Pointer to the `ssc_pwcdi_params` structure, which contains parameters such as the synthetic support data, 
     *        norm type (Lp-norm), and radius for the support creation.
     * @param input Pointer to the host array containing the input signal data (floats).
     * @param obj_input Pointer to the object input data (cufftComplex), which will be used during the algorithm (currently not used in this function).
     *
     * @note 
     * - The function assumes that all variables are 3D arrays with dimensions defined by `workspace->dimension`.
     * - The synthetic support is created using the `synth_support_data` function when no initial support data is provided. 
     *   The synthetic support is created based on the `pnorm` and `radius` values passed in `params`.
     * - If the initial support data is provided in `params->sup_data`, it is directly loaded into the support array.
     * - The support data is copied to GPU memory for subsequent algorithm computations.
     * 
     * @pre The `workspace->gpus->ngpus` should be correctly configured.
     */
 
    // single GPU case 
    if (workspace->gpus->ngpus==1){ 
      float time;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
  
      checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[0]));
      
      cudaEventRecord(start);


      const dim3 threadsPerBlock(tbx, tby, tbz);
      const dim3 gridBlock (ceil((workspace->dimension + threadsPerBlock.x - 1)/threadsPerBlock.x),
                            ceil((workspace->dimension + threadsPerBlock.y - 1)/threadsPerBlock.y),
                            ceil((workspace->dimension + threadsPerBlock.z - 1)/threadsPerBlock.z));
      
      
      // copy input data to workspace->sgpu.d_signal
      checkCudaErrors(cudaMemcpy(workspace->sgpu.d_signal,
                                 input,
                                 pow(workspace->dimension, 3)*sizeof(float),
                                 cudaMemcpyHostToDevice));

      // TODO: speed-up here support_data() nested-loop function.
      // allocate host memory for the support  
      uint8_t *hsupport = (uint8_t*) malloc(sizeof(uint8_t)*pow(workspace->dimension,3));

      // choose whether to create a synthetic initial support or to load it from input
      if (params->sup_data == NULL){
        // create synthetic initial support 
        printf("ssc-cdi: Creating synthetic initial support\n");
        synth_support_data(hsupport, workspace->dimension, params->pnorm, params->radius, 0.0, 0.0, 0.0);
      }else{
        // load from input
        printf("ssc-cdi: Using the initial support passed by dic['supinfo']['data']\n");
        // synth_support_data(hsupport, workspace->dimension, params->pnorm, params->radius, 0.0, 0.0, 0.0);
        for (int i=0; i<pow(workspace->dimension,3); i++){
          *(hsupport+i) = (uint8_t) params->sup_data[i];
        }

      }

      // copy support data to GPU memory
      checkCudaErrors(cudaMemcpy(workspace->sgpu.d_support,
                                 hsupport,
                                 pow(workspace->dimension, 3)*sizeof(uint8_t),
                                 cudaMemcpyHostToDevice));

      // free host memory storing the support 
      free(hsupport);


      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      if(workspace->timing){
          fprintf(stdout,"ssc-cdi: setting the initial support - %lf ms\n",time);
      }
  

      // set initial object data. either using the initial object in params, or 
      // inverting the input diffraction pattern. 
      cudaEventRecord(start);

      // if (params->amplitude_obj_data==NULL || params->phase_obj_data==NULL){
      if (obj_input==NULL){
        printf("ssc-cdi: Creating initial object data from the diffraction pattern.\n");
        // d_signal is assumed to have the lowest frequency at (Nx/2,Ny/2,Nz/2) 
        // in the input, so we shift the lower frequencies to the borders.
        // fftshift<<<gridBlock, threadsPerBlock>>>((void *) workspace->sgpu.d_signal,
        //            workspace->dimension,
        //            SSC_DTYPE_FLOAT);
        // cudaDeviceSynchronize();
        // getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");


        // sets d_x = d_signal * exp(1j * random) 
        set_initial<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_x,
                                                   (float*) workspace->sgpu.d_signal,
                                                    workspace->dimension);
        cudaDeviceSynchronize();
        getLastCudaError("ssc-cdi: error / Kernel execution failed @ set_initial<<<.>>>\n");

      
        // perform IFFT on d_x
        checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                     workspace->sgpu.d_x,
                                     workspace->sgpu.d_x,
                                     CUFFT_INVERSE));

        // normalize the IFFT of d_x
        normalize<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_x,
                                                  workspace->dimension);
        cudaDeviceSynchronize();
        getLastCudaError("ssc-cdi: error / kernel execution failed @ normalize<<<.>>>\n");


        fftshift<<<gridBlock, threadsPerBlock>>>((void *) workspace->sgpu.d_x,
                                                 workspace->dimension,
                                                 SSC_DTYPE_CUFFTCOMPLEX);
        cudaDeviceSynchronize();
        getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");
      }else{
        printf("ssc-cdi: Using initial object data from input parameters. \n");

        // copy d_x data to GPU memory
        checkCudaErrors(cudaMemcpy(workspace->sgpu.d_x,
                                   obj_input,
                                   pow(workspace->dimension, 3)*sizeof(cufftComplex),
                                   cudaMemcpyHostToDevice));
        getLastCudaError("ssc-cdi: error / cudaMemcpyHostToDevice().\n");

        // free host memory storing the amplitude and phase initial data
        // free(d_x_inputswap);
        // cudaFreeHost(d_x_inputswap);
        // getLastCudaError("ssc-cdi: error / cudaFreeHost().\n");

      }

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      if(workspace->timing){
        fprintf(stdout,"ssc-cdi: setting inital data - %lf ms\n",time);
      } 
       

      
    // multi GPU case
    }else if (workspace->gpus->ngpus>1){

      float time;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaError_t hostalloc_status;

    
      const size_t dim = workspace->nvoxels;
      const int n_gpus = workspace->gpus->ngpus;
      const size_t perGPUDim = dim/n_gpus;

      const dim3 threadsPerBlock(tbx*tby*tbz);
      const dim3 gridBlock (ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));

      uint8_t* swap_support_byte = (uint8_t*) malloc(sizeof(uint8_t)*workspace->nvoxels); // perGPUDim instead of workspace->nvoxels

      if(workspace->timing){
        cudaEventRecord(start);
      }

      // load d_signal from input data 
      for (int i=0; i<n_gpus; i++){
        if (params->map_d_signal==true){
          // copy data to pinned host memory
          memcpy(workspace->mgpu.d_signal_host[i], &(input[i*perGPUDim]),perGPUDim*sizeof(float));
        }else if (params->map_d_signal==false){
          // Copy data directly to GPU memory
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          cudaMemcpy(workspace->mgpu.d_signal[i], &(input[i*perGPUDim]), perGPUDim*sizeof(float), cudaMemcpyHostToDevice);
        }
      }

      for (int i = 0; i <n_gpus; ++i){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
        checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
      }


      // output wheter the initial support is being created or loaded from the input arguments 
      if (params->sup_data==NULL){
          printf("ssc-cdi: Creating synthetic initial support\n");
      }else{
          printf("ssc-cdi: Using the initial support passed by dic['supinfo']['data']\n");
      }

      // load or create the support 
      for (int i=0; i<n_gpus; i++){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
        
        // Use the following to maintain d_signal in GPU the whole time 
        // checkCudaErrors( cudaMemcpyAsync(workspace->mgpu.d_signal[i], &(input[i*perGPUDim]),
        //          perGPUDim*sizeof(float), cudaMemcpyHostToDevice,
        //          workspace->gpus->streams[i]) );
 
        if (params->sup_data==NULL){
          // create synthetic initial support  
          synth_support_data_mgpu<<<gridBlock, threadsPerBlock>>>(workspace->mgpu.d_support[i],
                                                                  params->pnorm,
                                                                  params->radius, 0.0, 0.0, 0.0,
                                                                  workspace->dimension, 
                                                                  n_gpus, 
                                                                  i, 
                                                                  true);
        }else{
          // load from input
          // create swap data to temporarily store the sup data (i think this is not necessary, as 
          // the sup data already is in sup_data)

          // convert params->sup_data[i*perGPUDim] to params->sup_data[(i+1)*perGPUDim]  
          // (exclusive) from float to uint8_t
          for (int j=i*perGPUDim; j<(i+1)*perGPUDim; j++){
            swap_support_byte[j-i*perGPUDim] = (uint8_t) params->sup_data[j]; 
          }
          checkCudaErrors(cudaMemcpy(workspace->mgpu.d_support[i], 
                                     swap_support_byte,
                                     perGPUDim*sizeof(uint8_t),
                                     cudaMemcpyHostToDevice));
                                     // workspace->gpus->streams[i]));
        }
      }
      for (int i = 0; i <n_gpus; ++i){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
        checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
      }
    
      // free swap variable
      free(swap_support_byte);

      if(workspace->timing){
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        fprintf(stdout,"ssc-cdi: moving signal and support to device - %lf ms\n",time);
      }

      //--------------------------------------
      // initialize d_x
      //--------------------------------------
      // if (params->amplitude_obj_data == NULL || params->phase_obj_data == NULL){
      if (obj_input==NULL){
        // use the inverted data as initial point 
        printf("ssc-cdi: Creating initial object data from the diffraction pattern.\n");

        for(int i=0; i<n_gpus; i++){
          checkCudaErrors( cudaSetDevice(  workspace->gpus->gpus[i] ) );
          // to = from * exp(1j * random)
          set_initial_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                           workspace->mgpu.d_signal[i],    // from 
                                                           perGPUDim);
        }
    
        for(int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
          getLastCudaError("ssc-cdi: error / kernel execution failed @ set_dx_mgpu<<<.>>> ");
        }

 
        checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C,
                                                 workspace->mgpu.d_x,
                                                 workspace->mgpu.d_x,
                                                 CUFFT_INVERSE));
        for(int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
          normalize_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i],
                                                        dim, 
                                                        perGPUDim);
        }
        for(int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
          getLastCudaError("ssc-cdi: error / kernel execution failed @ normalize_mgpu<<<.>>>\n");
        }
 
        // fix the ordering of d_x
        permuted2natural(workspace->mgpu.d_x, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);
 

      }else{
        // load initial object data from input
        printf("ssc-cdi: Using initial object data from input parameters. \n");

        // float theta,sin,cos;
        // cufftComplex *d_x_inputswap = (cufftComplex*) malloc(sizeof(cufftComplex)*perGPUDim); // perGPUDim only. not pow(workspace->dimension,3));

        for (int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          
          // copy to device 
          checkCudaErrors(cudaMemcpy(workspace->mgpu.d_x->descriptor->data[i], 
                                    //  d_x_inputswap,
                                     &(obj_input[i*perGPUDim]),
                                     perGPUDim*sizeof(cufftComplex),
                                     cudaMemcpyHostToDevice));
                                     // workspace->gpus->streams[i]));
          getLastCudaError("ssc-cdi: error / cudaMemcpyHostToDevice().\n");
        }

        // free host memory storing the amplitude and phase initial data
        // free(d_x_inputswap);
      }
    } 
  }

  typedef struct{  
    void* input;
    float* output; 
    size_t start, end, size;
    int nthread;
    int variable; 
    int complex_part; 
  }sscPwcdiC2A_t;

  void *sscPwcdiC2A_loop(void *t){
    size_t        w;
    sscPwcdiC2A_t *param;
    
    param = (sscPwcdiC2A_t *)t;

    if (param->variable == SSC_VARIABLE_ITER){
      cufftComplex *input = (cufftComplex*) param->input;
  
      for(w = param->start; w < SSC_MIN(param->end, param->size); w++){
        if (param->complex_part == AMPLITUDE){
          param->output[w] =  sqrt(powf(input[w].x,2) + powf(input[w].y,2));
        }else if (param->complex_part == PHASE){
          // param->output[w] = atan2f(input[w].y,input[w].x);
          // param->output[w] = atan2f(powf(input[w].y,2),powf(input[w].x,2));
          param->output[w] =  atan2(cuCimagf(input[w]), cuCrealf(input[w])); // atanf(input[w].y/input[w].x);
        }else if (param->complex_part == REAL_PART){
          param->output[w] = input[w].x;
        }else if (param->complex_part == IMAG_PART){
          param->output[w] = input[w].y;
        }else{
          fprintf(stderr,"ssc-cdi: Unrecognized option in param[n].complex_part. Exiting");
        }
      }
    }

    if (param->variable == SSC_VARIABLE_SUPP){
      uint8_t *input = (uint8_t*) param->input;
      
      for(w=param->start; w<SSC_MIN(param->end, param->size); w++){
        param->output[w] = (uint8_t) input[w];
      }
    }
  
    pthread_exit(NULL);
  }

  
 

  void set_output(char *outpath,
                  ssc_pwcdi_params *params,
                  ssc_pwcdi_plan *workspace,
                  int variable,
                  int complex_part){
/**
 * @brief Transfers the computed variable (either the iterated object or support) from the GPU to the host and saves it.
 *
 * This function copies the result of a computational variable (either the iterated object `d_x` or the support data `d_support`) 
 * from the GPU to the host. Depending on the specified `variable` type, it handles two different types of data: complex-valued data 
 * (for the object) or binary support data. The transfer is performed using CUDA memory copy functions.
 *
 * The function supports both single-GPU and multi-GPU setups. In the single-GPU case, data is copied directly from GPU memory to host 
 * memory. In the multi-GPU case, the data is divided across multiple GPUs and copied to the host sequentially.
 *
 * After the data is copied, the function optionally saves the result to a specified output path if a corresponding file saving operation 
 * is implemented (though this part is not shown in the provided code).
 *
 * @param outpath Path to the output file where the result will be saved (not used in the provided code, but typically intended for saving results).
 * @param params Pointer to the `ssc_pwcdi_params` structure, which contains algorithm parameters, including information about the variable to be transferred.
 * @param workspace Pointer to the `ssc_pwcdi_plan` structure, which contains the workspace and GPU memory allocations.
 * @param variable The type of variable to be transferred: `SSC_VARIABLE_ITER` for the iterated object `d_x`, or `SSC_VARIABLE_SUPP` for the support data `d_support`.
 * @param complex_part An integer indicating which part of the complex result to use (not utilized in the provided code).
 *
 * @note 
 * - The function supports both single-GPU and multi-GPU configurations. In the multi-GPU case, data is transferred from each GPU sequentially.
 * - Timing for the memory transfer is recorded using CUDA events, allowing for performance monitoring of the transfer process.
 * - The resulting data is copied to either the `result_c` (complex data) or `result_s` (support data) arrays, depending on the type of variable.
 * - The result is stored in host memory after the transfer and is intended to be saved or processed further.
 * 
 * @pre The `workspace` should have valid pointers to the GPU memory (`sgpu.d_x` or `sgpu.d_support`), and the `variable` parameter should specify the correct variable.
 * @post The GPU data (either the iterated object or support) is copied to the host memory, and the resulting data is ready for further use or saving.
 */

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cufftComplex *result_c;
    uint8_t *result_s;
    
    if (variable == SSC_VARIABLE_ITER){
      result_c = (cufftComplex*) malloc(workspace->nvoxels*sizeof(cufftComplex));
  
      cudaEventRecord(start);
  
      if (workspace->gpus->ngpus==1){
        // single GPU case
          checkCudaErrors(cudaMemcpy(result_c,
                                     workspace->sgpu.d_x,
                                     workspace->nvoxels*sizeof(cufftComplex),
                                     cudaMemcpyDeviceToHost));
      }else{
        // Multi GPU case
        checkCudaErrors(cufftXtMemcpy(workspace->plan_C2C,
                                      result_c,
                                      workspace->mgpu.d_x,
                                      CUFFT_COPY_DEVICE_TO_HOST));
      }
    }else if (variable==SSC_VARIABLE_SUPP){
      result_s = (uint8_t*) malloc(workspace->nvoxels*sizeof(uint8_t));
      
      if (workspace->timing){ 
        cudaEventRecord(start);
      }
  
      if (workspace->gpus->ngpus == 1){
        // Single GPU case
        checkCudaErrors(cudaMemcpy(result_s,
                                   workspace->sgpu.d_support,
                                   workspace->nvoxels*sizeof(uint8_t),
                                   cudaMemcpyDeviceToHost));
      }else{
        // Multi GPU case
        const size_t perGPUDim = workspace->nvoxels/workspace->gpus->ngpus;

        for (int i=0; i<workspace->gpus->ngpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
    
          checkCudaErrors(cudaMemcpy(&(result_s[i*perGPUDim]),
                                     workspace->mgpu.d_support[i], 
                                     perGPUDim*sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < workspace->gpus->ngpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          cudaDeviceSynchronize();
        }
      }
    }
    
    if (workspace->timing){  
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      fprintf(stdout,"ssc-cdi: copying result to host - %lf ms\n", time);
    }


 
    // saving the absolute value of "result" to output. 
    // Indenpendently if it is SSC_VARIABLE_ITER or SSC_VARIABLE_SUPP
    float *output = (float*) malloc(workspace->nvoxels*sizeof(float));

    struct timespec TimeStart, TimeEnd;
    clock_gettime(CLOCK, &TimeStart);

    // serial  
    // if (variable == SSC_VARIABLE_ITER){
    //   for( size_t k = 0; k < workspace->nvoxels; k++)
    //     output[k] = sqrt( pow( result_c[k].x, 2) + pow(result_c[k].y, 2) );
    // }else if (variable == SSC_VARIABLE_SUPP){
    //   for (size_t k =0; k < workspace->nvoxels; k++)
    //     output[k] = (float) result_s[k];
    // }

    // parallell
    pthread_t *thread;
    pthread_attr_t attr;
    int e, n, rc;    
    sscPwcdiC2A_t *param;  
    void *status;

    thread = (pthread_t*) malloc(sizeof(pthread_t)*params->sthreads);
    param  = (sscPwcdiC2A_t*) malloc(sizeof(sscPwcdiC2A_t)*params->sthreads);
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  
    e = (int) floor((workspace->nvoxels)/params->sthreads);;
        
    for(n = 0; n < params->sthreads; n++){
      param[n].size = workspace->nvoxels;
      if (variable == SSC_VARIABLE_ITER){
        param[n].input  = result_c;
      }else if (variable == SSC_VARIABLE_SUPP){
        param[n].input = result_s;
      }
      
      param[n].output = output; 
      param[n].nthread = n;            
      param[n].start = e*n;
      param[n].end = (n+1)*e;
      param[n].variable = variable;
      param[n].complex_part = complex_part;
        
      rc = pthread_create(&thread[n], 
                          &attr, 
                          sscPwcdiC2A_loop, 
                          (void *)&param[n]);
    }

    pthread_attr_destroy(&attr);
    
    for(n = 0; n < params->sthreads; n++){
      rc = pthread_join(thread[n], &status);
    }
    rc++;

    free(param);
    free(thread);
    
    clock_gettime(CLOCK, &TimeEnd);

    if (workspace->timing){
      if (complex_part == AMPLITUDE){
        fprintf(stdout,"ssc-cdi: transfer cuFFT complex result to amplitude - %lf ms\n", TIME(TimeEnd,TimeStart));
      }else if (complex_part == PHASE){
        fprintf(stdout,"ssc-cdi: transfer cuFFT complex result to amplitude - %lf ms\n", TIME(TimeEnd,TimeStart));
      }else if (complex_part == REAL_PART){
        fprintf(stdout,"ssc-cdi: transfer cuFFT complex result to real part - %lf ms\n", TIME(TimeEnd,TimeStart));
      }else if (complex_part == IMAG_PART){
        fprintf(stdout,"ssc-cdi: transfer cuFFT complex result to imaginary part  - %lf ms\n", TIME(TimeEnd,TimeStart));
      }
    }

    //
    // Saving output 
    clock_gettime(CLOCK, &TimeStart);
    FILE *fp = fopen(outpath, "w");

    fwrite( output, sizeof(float), workspace->nvoxels, fp);
    clock_gettime(CLOCK, &TimeEnd);
    fprintf(stdout,"ssc-cdi: saving output (fwrite) - %lf s\n",TIME(TimeEnd,TimeStart));

    fclose(fp);

    if (variable == SSC_VARIABLE_ITER){
      free(result_c);
    }else{
      free(result_s);
    }

    free(output);
  } 


__global__ void inplaceRedistributeKernel(cufftComplex *data, 
                                          int width, int height, int depth,
                                          int subDepth, int srcOffset, int dstOffset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Temporary staging area in shared memory (if needed)
    __shared__ cufftComplex tile[8][8][8];  // Adjust size based on block dimensions

    if (x < width && y < height && z < subDepth) {
        int srcIndex = srcOffset + z * (width * height) + y * width + x;
        int dstIndex = dstOffset + z * (width * height) + y * width + x;

        // Load data into shared memory
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = data[srcIndex];
        __syncthreads();

        // Write data from shared memory to in-place position
        data[dstIndex] = tile[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void RedistributeDataInPlace(cufftComplex *data[], int width, int height, int depth, int numGPUs) {
    int subDepth = depth / numGPUs;

    for (int gpu = 0; gpu < numGPUs; ++gpu) {
        cudaSetDevice(gpu);
        int srcOffset = gpu * (width * height * subDepth);
        int dstOffset = srcOffset;  // Adjust if needed

        // Launch kernel with 3D block and grid dimensions
        dim3 blockDim(8, 8, 8);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y,
                     (subDepth + blockDim.z - 1) / blockDim.z);

        inplaceRedistributeKernel<<<gridDim, blockDim>>>(data[gpu], width, height, depth,
                                                         subDepth, srcOffset, dstOffset);
        cudaDeviceSynchronize();  // Sync each GPU separately
    }
}


void m_projection_M(cufftHandle& plan_C2C,
                    cudaLibXtDesc* d_y,  
                    cudaLibXtDesc* d_x, 
                    float** d_signal,
                    float eps,
                    size_t totalDim, 
                    size_t perGPUDim,
                    ssc_gpus *gpus,
                    cufftComplex* host_swap,
                    bool timing){

/**
 * @brief Performs the M projection step of the Hybrid Input-Output (HIO) algorithm for phase retrieval using multiple GPUs.
 *
 * This function carries out the M projection step of the HIO algorithm, leveraging multiple GPUs to accelerate the computation. It involves the following key steps:
 * 1. Copies the input data from `d_x` to `d_y`.
 * 2. Performs forward FFT on `d_y`.
 * 3. Adjusts the phase of the FFT result.
 * 4. Performs inverse FFT on the adjusted data.
 * 5. Ensures proper data ordering throughout the process.
 *
 * The `m_projection_M` function operates on the assumption that both `d_x->subFormat` and `d_y->subFormat` are equal to `CUFFT_XT_FORMAT_INPLACE`.
 *
 * @param plan_C2C The cuFFT plan handle for complex-to-complex FFT operations.
 * @param d_y Pointer to the output array descriptor (`cudaLibXtDesc`) after the M projection.
 * @param d_x Pointer to the input array descriptor (`cudaLibXtDesc`) for the M projection.
 * @param d_signal Pointer to an array of device pointers containing the signal values for each GPU.
 * @param eps Small epsilon value for phase adjustment.
 * @param totalDim Total dimension of the input/output arrays.
 * @param perGPUDim Dimension of the array segment allocated per GPU.
 * @param gpus Structure containing information about the GPUs used in the computation.
 * @param host_swap Pointer to a host array used for temporary storage during permutation.
 * @param timing Boolean flag to enable timing measurements (currently unused in this code).
 *
 * @note This function assumes that `d_x` and `d_y` are preallocated and have dimensions aligned for multiple GPU execution.
 *       It performs synchronization after each major step to ensure data consistency. 
 */

  int n_gpus = totalDim/perGPUDim;
  const dim3 threadsPerBlock(tbx*tby*tbz);
  const dim3 gridBlock(ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
     

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time_copydx;
 
 
  // fix the format of d_z (dunno)
  // d_z->subFormat = CUFFT_XT_FORMAT_INPLACE;
  // d_x->subFormat = CUFFT_XT_FORMAT_INPLACE;

  // copy the content of d_x, which is in natural ordering format, to d_y
  // note that this copy is done directly on the content.
  for (int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaMemcpy((void*) d_y->descriptor->data[i], //  versus cudaMemcpyAsync
                               (void*) d_x->descriptor->data[i],
                                perGPUDim*sizeof(cufftComplex),
                                cudaMemcpyDeviceToDevice));
                                // gpus->streams[i]));
  }     
  for (int i = 0; i <n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    getLastCudaError("ssc-cdi: error / cudaMemcpy() failed @ m_projection_M()\n");
  }

  // this should work too, but it is not. I guess a driver update would solve  
  // checkCudaErrors(cufftXtMemcpy(plan_C2C,
  //                               d_y,  
  //                               d_x,
  //                               CUFFT_COPY_DEVICE_TO_DEVICE));


  // perform FFT on d_y
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C, 
                                           d_y,               // input
                                           d_y,               // output
                                           CUFFT_FORWARD));  

  
  printf("ssc-cdi true: d_y->subFormat = %d\n", d_y->subFormat); 
  fflush(stdout);
 

  // fix the ordering of d_y
  permuted2natural(d_y, plan_C2C, totalDim, host_swap);  



  printf("ssc-cdi true: d_y->subFormat = %d\n", d_y->subFormat); 
  fflush(stdout);



  // adjust the phase of the fft of d_y, but copy the results directly to d_y
  // to = signal * exp( i * from ) / totalDim 
  for(int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    update_with_phase_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_y->descriptor->data[i], //to 
                                                           (cufftComplex*) d_y->descriptor->data[i], //from  
                                                           d_signal[i],
                                                           eps,
                                                           totalDim,
                                                           perGPUDim);
  }
  for(int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_phase_mgpu<<<.>>>\n");
  }
 
 
  // compute the ifft of d_y. Note that we wouldn't need to convert the subformat to natural ordering
  // if we had not corrected the data subformat in the previous step. 
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C,
                                           d_y,                 //input
                                           d_y,                 //output
                                           CUFFT_INVERSE));

  // fix the ordering of d_y
  permuted2natural(d_y, plan_C2C, totalDim, host_swap); 
}
  

  void m_projection_M_shuffleddata(cufftHandle& plan_C2C,
                                   cudaLibXtDesc* d_y,  
                                   cudaLibXtDesc* d_x, 
                                   float** d_signal,
                                   float eps,
                                   size_t totalDim, 
                                   size_t perGPUDim,
                                   ssc_gpus *gpus,                  
                                   bool timing){
/**
 * @brief This function performs a projection onto the space of points that satisfy the measured data constraint, 
 * adjusting the phase of the Fourier transform of the data.
 *
 * The projection is done by copying the data from `d_x` (natural ordering) to `d_y`, performing a forward FFT on 
 * `d_y`, applying a phase adjustment based on the signal, and then performing an inverse FFT on `d_y`. The result 
 * is the projection of the data in the space of points that satisfy the measured data constraint, ensuring that 
 * the adjusted signal complies with the measured data.
 *
 * The process involves multiple GPUs, with each GPU handling a portion of the data, and it uses CUDA streams 
 * for asynchronous execution and synchronization across the devices.
 *
 * **Steps:**
 * 1. Copy `d_x` to `d_y`.
 * 2. Perform a forward FFT on `d_y`.
 * 3. Adjust the phase of the Fourier transform based on the signal.
 * 4. Perform an inverse FFT on `d_y` to return to the spatial domain.
 *
 * @param `plan_C2C`: The FFT plan for complex-to-complex transformations.
 * @param `d_y`: The descriptor holding the data that will be projected onto the measured data space.
 * @param `d_x`: The descriptor holding the input data in natural ordering format.
 * @param `d_signal`: The signal data used for phase adjustment.
 * @param `eps`: The epsilon parameter for phase adjustment.
 * @param `totalDim`: The total dimension of the data.
 * @param `perGPUDim`: The number of elements each GPU will handle.
 * @param `gpus`: A pointer to a `ssc_gpus` struct containing GPU information.
 * @param `timing`: A boolean flag to enable or disable timing.
 * 
 * @note This projection is done by adjusting the phase of the Fourier-transformed data (`d_y`), ensuring that it 
 * satisfies the measured data constraint based on the input signal (`d_signal`).
 */
 
  int n_gpus = totalDim/perGPUDim;
  const dim3 threadsPerBlock(tbx*tby*tbz);
  const dim3 gridBlock(ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
     

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time_copydx;



  // copy the content of d_x, which is in natural ordering format, to d_y
  // note that this copy is done directly on the content.
  for (int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaMemcpy((void*) d_y->descriptor->data[i],  // to   
                               (void*) d_x->descriptor->data[i],  // from
                               perGPUDim*sizeof(cufftComplex),
                               cudaMemcpyDeviceToDevice));
  }     
  for (int i = 0; i <n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    getLastCudaError("ssc-cdi: error / cudaMemcpy() failed @ m_projection_M()\n");
  }
 
  // perform FFT on d_y
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C, 
                                           d_y,               // input
                                           d_y,               // output
                                           CUFFT_FORWARD));  

  

  // adjust the phase of the fft of d_y, but copy the results directly to d_y
  // to = signal * exp( i * from ) / totalDim 
  for(int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    update_with_phase_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_y->descriptor->data[i], //to 
                                                           (cufftComplex*) d_y->descriptor->data[i], //from  
                                                           d_signal[i],
                                                           eps,
                                                           totalDim,
                                                           perGPUDim);
  }
  for(int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_phase_mgpu<<<.>>>\n");
  }
   

  // compute the ifft of d_y. Note that we wouldn't need to convert the subformat to natural ordering
  // if we had not corrected the data subformat in the previous step. 
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C,
                                           d_y,                 //input
                                           d_y,                 //output
                                           CUFFT_INVERSE));
 
}

 

 

void m_fftshift(uint8_t** data, 
                size_t dimension, 
                int dtype, 
                uint8_t* host_swap_byte,
                size_t perGPUDim,
                ssc_gpus* gpus){  
                  /**
 * @brief Experimental function to perform an FFT shift operation across multiple GPUs, currently only supports byte data type (`SSC_DTYPE_BYTE`).
 *
 * **WARNING:** This function is experimental and **should not be used in production environments**. It is designed to handle FFT shifting on a single GPU for small datasets (`dtype = SSC_DTYPE_BYTE`). The multi-GPU handling for complex data types (`SSC_DTYPE_COMPLEX`) is incomplete and likely to be unreliable.
 *
 * This kernel:
 * 1. Transfers data from the device memory to the host memory (`host_swap_byte`).
 * 2. Performs an FFT shift on the data on a single GPU.
 * 3. Copies the result back from the host to device memory.
 *
 * The number of GPUs used is determined by dividing the total data size by `perGPUDim`. The kernel for `fftshift` is launched on a single GPU (i.e., the operation does not fully utilize multiple GPUs for `SSC_DTYPE_BYTE` data).
 *
 * @param data Pointer to an array of device memory pointers, each holding data on a different GPU.
 * @param dimension The size of the data dimension.
 * @param dtype Data type for the input data (`SSC_DTYPE_BYTE` for now).
 * @param host_swap_byte Host-side memory to temporarily hold the data during the FFT shift operation.
 * @param perGPUDim The number of elements to be processed per GPU.
 * @param gpus A pointer to a `ssc_gpus` struct containing GPU device information.
 *
 * @note The current implementation does not fully support multiple GPUs for complex data types (`SSC_DTYPE_COMPLEX`). 
 * Currently this only works for dtype=SSC_DTYPE_BYTE data, since it is small enough to be performed in a single GPU. 
 * The SSC_DTYPE_COMPLEX case should be handled in multiple GPUs.
 */
 

  const int n_gpus = (dimension*dimension*dimension)/perGPUDim;
  const dim3 threadsPerBlock(tbx, tby, tbz);
  const dim3 gridBlock (ceil((dimension + threadsPerBlock.x - 1)/threadsPerBlock.x),
                        ceil((dimension + threadsPerBlock.y - 1)/threadsPerBlock.y),
                        ceil((dimension + threadsPerBlock.z - 1)/threadsPerBlock.z));
      

  
  // copy data (uint8_t**) to host_swap_byte (uint8_t*)
  for (int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaMemcpy((void*) &host_swap_byte[i*perGPUDim], // Async
                                    (void*) data[i], 
                                    perGPUDim*sizeof(uint8_t), 
                                    cudaMemcpyDeviceToHost));
                                    // gpus->streams[i])); // added this 
  }
  for (int i = 0; i<gpus->ngpus; ++i){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
  }


  // fftshift in a single GPU 
  fftshift<<<gridBlock, threadsPerBlock>>>((void*) host_swap_byte,
                                           dimension,
                                           SSC_DTYPE_BYTE);
  cudaDeviceSynchronize();
  getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");

  // copy back to data pointer
  for (int i = 0; i < n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaMemcpy((void*) data[i], //Async
                                    (void*) &host_swap_byte[i*perGPUDim],
                                    perGPUDim*sizeof(uint8_t), 
                                    cudaMemcpyHostToDevice));
                                    // gpus->streams[i])); // added this
     
  }
  for (int i=0; i<gpus->ngpus; ++i){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
  }

}




  void m_projection_S(cudaLibXtDesc* d_z,
                      cudaLibXtDesc* d_y,
                      uint8_t** d_support,
                      int extra_constraint,
                      size_t totalDim,
                      size_t perGPUDim,
                      ssc_gpus *gpus){
  /**
   * @brief Performs a multi-GPU projection operation, updating the complex array `d_z` with the values of `d_y` based on the support mask `d_support`.
   * Optionally applies an extra constraint during the projection.
   *
   * This function distributes the computation across multiple GPUs, where each GPU processes a segment of the input arrays.
   * For each GPU, it projects the array `d_y` onto `d_z` according to the support mask `d_support`. If an extra constraint is specified,
   * it applies a phase constraint in addition to the support constraint.
   *
   * The update rule is defined as:
   * 
   * \code
   * d_z[index] = d_support[index] * d_y[index] + (1 - d_support[index]) * d_z[index]
   * \endcode
   * 
   * When an extra constraint is applied, the update rule is modified to incorporate phase restrictions.
   *
   * @param d_z Pointer to a `cudaLibXtDesc` structure containing the output array segments across multiple GPUs. 
   *            Each GPU handles a portion of the total dataset (`perGPUDim` elements per GPU).
   * @param d_y Pointer to a `cudaLibXtDesc` structure containing the input array segments across multiple GPUs.
   *            The values from this array are projected onto `d_z` based on `d_support`.
   * @param d_support Array of pointers to the support masks (`uint8_t`) for each GPU, where 1 indicates support, 
   *                  and 0 indicates the region outside the support.
   * @param extra_constraint Specifies whether an additional phase constraint is applied:
   *                         - `NO_EXTRA_CONSTRAINT` applies only the support constraint.
   *                         - Other values apply the support and a phase constraint.
   * @param totalDim The total number of elements across all GPUs (combined size of `d_z` and `d_y`).
   * @param perGPUDim The number of elements assigned to each GPU for processing.
   * @param gpus Structure containing information about the available GPUs and their associated CUDA streams.
   *
   * @note The function assumes that the input arrays (`d_z`, `d_y`, `d_support`) are preallocated and distributed across the GPUs.
   *       Each GPU processes its own segment independently, with synchronization handled after the kernel execution.
   */


    int n_gpus = totalDim/perGPUDim;
    const dim3 threadsPerBlock(tbx*tby*tbz);
    const dim3 gridBlock(ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
  
    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // decide wether to use the extra constraint kernel or the vanilla one 
      if (extra_constraint==NO_EXTRA_CONSTRAINT){
        // project onto the support alone 
        update_with_support_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z->descriptor->data[i], //to
                                                                 (cufftComplex*) d_y->descriptor->data[i], //from
                                                                 (uint8_t*) d_support[i], 
                                                                 perGPUDim); 
      }else{
        // project onto the support AND the phase constraint 
        update_with_support_extra_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z->descriptor->data[i], //to
                                                                       (cufftComplex*) d_y->descriptor->data[i], //from
                                                                       (uint8_t*) d_support[i], 
                                                                       extra_constraint,
                                                                       perGPUDim); 
      }
    }

    // cudaDeviceSynchronize();
    for(int i = 0; i < n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
       checkCudaErrors(cudaStreamSynchronize(gpus->streams[i])); 
      getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_support_mgpu<<<.>>>\n");
    }
  }

  void m_projection_S_only(cudaLibXtDesc* d_z,
                           cudaLibXtDesc* d_y,
                           uint8_t** d_support,
                           int extra_constraint,
                           size_t totalDim,
                           size_t perGPUDim,
                           ssc_gpus *gpus){
    //
    // dz = support * dy
    //
    int n_gpus = totalDim/perGPUDim;
    const dim3 threadsPerBlock(tbx*tby*tbz);
    const dim3 gridBlock (ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
  
    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // decide wether to use the extra constraint kernel or the vanilla one 
      if (extra_constraint==NO_EXTRA_CONSTRAINT){
        // project onto the support alone 
        multiply_support_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z->descriptor->data[i], //to
                                                              (cufftComplex*) d_y->descriptor->data[i], //from
                                                               d_support[i], 
                                                               perGPUDim);
      }else{
        // project onto the support AND the phase constraint 
        multiply_support_extra_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z->descriptor->data[i], //to
                                                                    (cufftComplex*) d_y->descriptor->data[i], //from
                                                                    d_support[i], 
                                                                    extra_constraint,
                                                                    perGPUDim);
      }
    }

    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i])); 
      getLastCudaError("ssc-cdi: error / Kernel execution failed @ multiply_support_mgpu<<<.>>>\n");
    }
  }




  void s_projection_M(cufftHandle& plan_input,
                      cufftComplex* d_y, // to
                      cufftComplex* d_x, // from 
                      float* d_signal,
                      float eps,
                      int dimension){


    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil((dimension + threadsPerBlock.x - 1) / threadsPerBlock.x),
                          ceil((dimension + threadsPerBlock.y - 1) / threadsPerBlock.y),
                          ceil((dimension + threadsPerBlock.z - 1) / threadsPerBlock.z));


    // d_y = FFT ( d_x )
    checkCudaErrors(cufftExecC2C(plan_input,
                                 d_x, //input
                                 d_y, //output
                                 CUFFT_FORWARD));
    // cudaDeviceSynchronize();
 

    //  to = signal * exp( i * from ) / (dimension^3)
    update_with_phase<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_y, //to
                                                      (cufftComplex*) d_y, //from
                                                      (float*) d_signal,
                                                      eps,
                                                      dimension);
    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_phase<<<.>>>\n");
    



    // dy = IFFT (dy)
    checkCudaErrors(cufftExecC2C(plan_input, d_y, d_y, CUFFT_INVERSE));
    // cudaDeviceSynchronize();
 
  }


  void s_projection_S(cufftHandle& plan_input,
                      cufftComplex* d_z,
                      cufftComplex* d_y,
                      uint8_t* d_support,
                      int extra_constraint,
                      int dimension){
  /**
   * @brief Projects the complex array `d_y` onto the complex array `d_z` based on a support mask, with an optional extra constraint.
   *
   * This function applies a projection of the input array `d_y` onto the output array `d_z`, constrained by the support mask `d_support`.
   * The update is performed in-place using a CUDA kernel. If specified, an extra constraint can be applied to enforce phase conditions
   * in addition to the support constraint.
   *
   * The update rule is defined as:
   * 
   * \code
   * d_z[index] = d_support[index] * d_y[index] + (1 - d_support[index]) * d_z[index]
   * \endcode
   * 
   * When an extra constraint is applied, the projection rule is modified to account for phase restrictions.
   *
   * @param plan_input A reference to the CUFFT plan (`cufftHandle`) for executing FFT transformations, though it is not used directly in this function.
   *                   It can be useful for applying future transformations if required.
   * @param d_z Pointer to the output complex array (`cufftComplex`) that will be updated with projected values from `d_y`.
   * @param d_y Pointer to the input complex array (`cufftComplex`) to be projected and used for updating `d_z`.
   * @param d_support Pointer to the support mask (`uint8_t`) that defines which regions of `d_z` are updated based on `d_y`.
   *                  A value of `1` indicates a region inside the support, and `0` indicates outside the support.
   * @param extra_constraint Specifies whether an additional constraint is applied:
   *                         - `NO_EXTRA_CONSTRAINT` applies only the support constraint.
   *                         - Other values apply the support and a phase constraint.
   * @param dimension Specifies the total dimension size of the arrays (assumed cubic, i.e., `dimension x dimension x dimension`).
   *
   * @note This function assumes that the input arrays (`d_z`, `d_y`, and `d_support`) are preallocated and have dimensions aligned with the grid and block sizes for CUDA execution.
   *       The grid and block dimensions are calculated based on `dimension`, and kernel synchronization is handled after execution.
   */
    
    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil(dimension + threadsPerBlock.x - 1)/threadsPerBlock.x,
                          ceil(dimension + threadsPerBlock.y - 1)/threadsPerBlock.y,
                          ceil(dimension + threadsPerBlock.z - 1)/threadsPerBlock.z);
  

    // decide wether to use the extra constraint kernel or the vanilla one 
    if (extra_constraint==NO_EXTRA_CONSTRAINT){
      // project onto the support alone 
      update_with_support<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z,
                                                          (cufftComplex*) d_y,
                                                          (uint8_t*) d_support,
                                                          dimension);
    }else{
      // project onto the support alone AND the phase constraint (extra constraint)
      update_with_support_extra<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z,
                                                                (cufftComplex*) d_y,
                                                                (uint8_t*) d_support,
                                                                extra_constraint,
                                                                dimension);
    }

    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_support<<<.>>>\n");
 
  }
  
  void s_projection_S_only(cufftHandle& plan_input,
                           cufftComplex* d_z,
                           cufftComplex* d_y,
                           uint8_t* d_support,
                           int extra_constraint,
                           int dimension){
    //
    // d_z = support * dy 
   
    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil(dimension + threadsPerBlock.x - 1)/threadsPerBlock.x,
                          ceil(dimension + threadsPerBlock.y - 1)/threadsPerBlock.y,
                          ceil(dimension + threadsPerBlock.z - 1)/threadsPerBlock.z);


    // decide wether to use the extra constraint kernel or the vanilla one 
    if (extra_constraint==NO_EXTRA_CONSTRAINT){
      // project onto the support alone 
      multiply_support<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z,
                                                       (cufftComplex*) d_y,
                                                       d_support, 
                                                       dimension); 

    }else{
      // project onto the support alone AND the phase constraint (extra constraint)
      multiply_support_extra<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z,
                                                             (cufftComplex*) d_y,
                                                             d_support, 
                                                             extra_constraint,
                                                             dimension); 
    }

    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ multiply_support<<<.>>>\n");
  }

  

  void s_projection_extra_constraint(cufftComplex* d_x,
                                     cufftComplex* d_y,
                                     int extra_constraint,
                                     int dimension){
 
    
    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil(dimension + threadsPerBlock.x - 1)/threadsPerBlock.x,
                          ceil(dimension + threadsPerBlock.y - 1)/threadsPerBlock.y,
                          ceil(dimension + threadsPerBlock.z - 1)/threadsPerBlock.z);
  


    update_extra_constraint<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_x,
                                                        (cufftComplex*) d_y,
                                                        extra_constraint,
                                                        dimension);
    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_extra_constraint<<<.>>>\n");
 
  }



    void m_projection_extra_constraint(cudaLibXtDesc* d_x,
                                       cudaLibXtDesc* d_y,
                                       int extra_constraint,
                                       size_t totalDim,
                                       size_t perGPUDim,
                                       ssc_gpus *gpus){
 
    int n_gpus = totalDim/perGPUDim;
    const dim3 threadsPerBlock(tbx*tby*tbz);
    const dim3 gridBlock (ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
  
    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // to = support * from
      update_extra_constraint_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_x->descriptor->data[i], //to
                                                                   (cufftComplex*) d_y->descriptor->data[i], //from
                                                                   extra_constraint, 
                                                                   perGPUDim);
    }

    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
       checkCudaErrors(cudaStreamSynchronize(gpus->streams[i])); 
      getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_extra_constraint_mgpu<<<.>>>\n");
    }
  }


  // Function to get output from object d_x and support d_support
  void get_output(cufftComplex* obj_output, 
                  uint8_t* finsup_output, 
                  ssc_pwcdi_plan* workspace, 
                  ssc_pwcdi_params* params){
                    
    if (workspace->gpus->ngpus>1){
      // multi-GPU case
      // copy d_x to obj_output
      checkCudaErrors(cufftXtMemcpy(workspace->plan_C2C,
                                    obj_output,
                                    workspace->mgpu.d_x,
                                    CUFFT_COPY_DEVICE_TO_HOST));

      // copy d_support to finsup_output
      const size_t perGPUDim = workspace->nvoxels/workspace->gpus->ngpus;

      for (int i=0; i<workspace->gpus->ngpus; i++){
        if (params->map_d_support){
          // copy d_support_host to finsup_output
          memcpy(&(finsup_output[i*perGPUDim]), workspace->mgpu.d_support_host[i], perGPUDim*sizeof(uint8_t));
        }else{
          // copy d_support to finsup_output
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaMemcpy(&(finsup_output[i*perGPUDim]),
                                    workspace->mgpu.d_support[i], 
                                    perGPUDim*sizeof(uint8_t),
                                    cudaMemcpyDeviceToHost));
        

          // synchronize
          for (int i=0; i<workspace->gpus->ngpus; ++i){
            checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
            cudaDeviceSynchronize();
          }
        }
      }
    }else{
      // single-GPU case
      // copy d_x to obj_output
      cudaMemcpy(obj_output, workspace->sgpu.d_x, workspace->nvoxels*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
          
      if (params->map_d_support){
        // copy d_support_host to finsup_output
        memcpy(finsup_output, workspace->sgpu.d_support_host, workspace->nvoxels*sizeof(uint8_t));
      }else{
        // copy d_support to finsup_output
        cudaMemcpy(finsup_output, workspace->sgpu.d_support, workspace->nvoxels*sizeof(uint8_t), cudaMemcpyDeviceToHost);
      }
    }
  }





}   // extern "C"


 