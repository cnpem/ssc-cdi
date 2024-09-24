#include <iostream>
#include <math.h>

#include "pwcdi.h"
#include "fft.h"
#include "gpus.h"

extern "C" {
  
  void alloc_workspace(ssc_pwcdi_plan *workspace,
                       ssc_pwcdi_params *params,
                       int *gpus,
                       int ngpu){
                                 
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
        checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_support, sizeof(short)*pow(workspace->dimension, 3)));
      }else if (params->map_d_support==true){
        // allocate d_support as mapped host memory 
        checkCudaErrors(cudaMallocHost((void**) &workspace->sgpu.d_support_host, sizeof(short)*pow(workspace->dimension, 3),cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&workspace->sgpu.d_support, (void *)workspace->sgpu.d_support_host, 0));    
      }


      // malloc other variables
      checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_x, sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_y, sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      // checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_z, sizeof(cufftComplex)*pow(workspace->dimension, 3)));


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
      if (params->errType==ITER_DIFF){
        checkCudaErrors(cudaMalloc((void**) &workspace->sgpu.d_x_lasterr, sizeof(cufftComplex)*pow(workspace->dimension, 3)));
      }
      
      // profile GPU (after)
      ssc_gpus_get_info(workspace->gpus->ngpus, workspace->gpus->gpus, ASTATE);
    
    }else if (ngpu>1){
      // multi GPU case 

      // Memory allocation (Heavy):
      // -----------------
      // d_x, d_y, d_z - cufftComplex data type
      // d_signal      - float data type
      // d_support     - short data type
      // plan_C2C      - ~ 3x cufftComplex data type

  
      // allocate gpus workspace
      workspace->gpus        = (ssc_gpus*) malloc(sizeof(ssc_gpus));
      workspace->gpus->gpus  = (int*) malloc(sizeof(int)*ngpu);
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
        hostalloc_status = cudaMallocHost((void**)&workspace->mgpu.d_support, n_gpus*sizeof(short*));

        // allocate d_support_host as pinned host memory. After that, 
        // allocate a device pointer for d_signal containing a reference to d_support_host only. 
        hostalloc_status = cudaMallocHost((void **)&workspace->mgpu.d_support_host, n_gpus*sizeof(short*));
        for (int i=0; i<n_gpus; i++){
           hostalloc_status =cudaMallocHost((void **)&workspace->mgpu.d_support_host[i], (size_t) (N*N*N*sizeof(short)/n_gpus), cudaHostAllocMapped);
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
        hostalloc_status = cudaMallocHost((void**)&workspace->mgpu.d_support, (size_t) (n_gpus*sizeof(short*)));
        for (int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
          checkCudaErrors(cudaMalloc((void **)&(workspace->mgpu.d_support[i]),(size_t)N*N*N*sizeof(short)/n_gpus)); 
        }
        for (int i=0; i<n_gpus; ++i){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
        }
      }
      // workspace->mgpu.d_support  = (short**)calloc(n_gpus, sizeof(short*));
      // workspace->mgpu.d_gaussian = (cufftComplex**)calloc(n_gpus, sizeof(cufftComplex*));
      // checkCudaErrors(cudaMalloc((void**)&workspace->mgpu.d_support, (size_t) (n_gpus*sizeof(short*))));
    

      ///////////////////////////////////// 
      // dynamic allocation: allocate host swap variable 
      // workspace->host_swap = (cufftComplex*)malloc(N*N*N*sizeof(cufftComplex));
      hostalloc_status = cudaMallocHost((void **)&workspace->host_swap, N*N*N*sizeof(cufftComplex));

      // dynamic allocation: allocate host swap short variable 
      // workspace->host_swap_short = (short*)malloc(N*N*N*sizeof(short));
      hostalloc_status = cudaMallocHost((void **)&workspace->host_swap_short,N*N*N*sizeof(short));

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
      if (params->errType==ITER_DIFF){
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
      
      if (params->errType==ITER_DIFF){
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
      cudaFreeHost(workspace->host_swap_short);

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
      if (params->errType==ITER_DIFF){
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

  
  void synth_support_data(short *support, 
                          int dimension,
                          int p, 
                          float radius,  
                          float x0, 
                          float y0, 
                          float z0){

    // this will be depracted and implemented in GPU in a future version. 

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
                 float *input){
    //
    // sets initial guess and .....
    
    
    // single GPU case 
    if ( workspace->gpus->ngpus==1 ){ 
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
      short *hsupport = (short*) malloc(sizeof(short)*pow(workspace->dimension,3));

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
          *(hsupport+i) = (short) params->sup_data[i];
        }

      }

      // copy support data to GPU memory
      checkCudaErrors(cudaMemcpy(workspace->sgpu.d_support,
                                 hsupport,
                                 pow(workspace->dimension, 3)*sizeof(short),
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

      if (params->amplitude_obj_data==NULL || params->phase_obj_data==NULL){
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

        cufftComplex *d_x_inputswap = (cufftComplex*) malloc(sizeof(cufftComplex)*pow(workspace->dimension,3));

        float theta,sin,cos;
        for (int i=0; i<pow(workspace->dimension,3); i++){
          sincosf(params->phase_obj_data[i], &sin, &cos);

          d_x_inputswap[i].x = params->amplitude_obj_data[i]*cos;
          d_x_inputswap[i].y = params->amplitude_obj_data[i]*sin;
        }

        // copy support data to GPU memory
        checkCudaErrors(cudaMemcpy(workspace->sgpu.d_x,
                                   d_x_inputswap,
                                   pow(workspace->dimension, 3)*sizeof(cufftComplex),
                                   cudaMemcpyHostToDevice));
        getLastCudaError("ssc-cdi: error / cudaMemcpyHostToDevice().\n");

        // free host memory storing the amplitude and phase initial data
        free(d_x_inputswap);
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

      short* swap_support_short = (short*) malloc(sizeof(short)*workspace->nvoxels); // perGPUDim instead of workspace->nvoxels

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
          // (exclusive) from float to short
          for (int j=i*perGPUDim; j<(i+1)*perGPUDim; j++){
            swap_support_short[j-i*perGPUDim] = (short) params->sup_data[j]; 
          }
          checkCudaErrors(cudaMemcpy(workspace->mgpu.d_support[i], 
                                     swap_support_short,
                                     perGPUDim*sizeof(short),
                                     cudaMemcpyHostToDevice));
                                     // workspace->gpus->streams[i]));
        }
      }
      for (int i = 0; i <n_gpus; ++i){
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
        checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
      }
    
      // free swap variable
      free(swap_support_short);

      if(workspace->timing){
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        fprintf(stdout,"ssc-cdi: moving signal and support to device - %lf ms\n",time);
      }

      //--------------------------------------
      // initialize d_x
      //--------------------------------------
      if (params->amplitude_obj_data == NULL || params->phase_obj_data == NULL){
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
 

        // // shift back the data 
        // for (int i=0; i<n_gpus; i++){
        //   checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
        //   m_fftshift_cufftcomplex((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i],
        //                           (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 
        //                           (size_t) cbrt((double)workspace->nvoxels),
        //                           workspace->gpus,
        //                           i);
        // // }
        // m_fftshift(workspace->mgpu.d_x, 
        //                 (size_t) cbrt((double)workspace->nvoxels), 
        //                 SSC_DTYPE_SHORT, 
        //                 workspace->host_swap_short,
        //                 perGPUDim,
        //                 workspace->gpus);
      }else{
        // load initial object data from input
        printf("ssc-cdi: Using initial object data from input parameters. \n");

        float theta,sin,cos;
        cufftComplex *d_x_inputswap = (cufftComplex*) malloc(sizeof(cufftComplex)*perGPUDim); // perGPUDim only. not pow(workspace->dimension,3));

        for (int i = 0; i < n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          
          // temporarily store the real and imaginary phase for each element in the current slice of the volume
          for (int j=i*perGPUDim; j<(i+1)*perGPUDim; j++){
            sincosf(params->phase_obj_data[j], &sin, &cos);
            d_x_inputswap[j-i*perGPUDim].x = (float) params->amplitude_obj_data[j]*cos;
            d_x_inputswap[j-i*perGPUDim].y = (float) params->amplitude_obj_data[j]*sin;
          }

          // copy to device 
          checkCudaErrors(cudaMemcpy(workspace->mgpu.d_x->descriptor->data[i], 
                                     d_x_inputswap,
                                     perGPUDim*sizeof(cufftComplex),
                                     cudaMemcpyHostToDevice));
                                     // workspace->gpus->streams[i]));
          getLastCudaError("ssc-cdi: error / cudaMemcpyHostToDevice().\n");
        }

        // free host memory storing the amplitude and phase initial data
        free(d_x_inputswap);
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
      short *input = (short*) param->input;
      
      for(w=param->start; w<SSC_MIN(param->end, param->size); w++){
        param->output[w] = (float) input[w];
      }
    }
  
    pthread_exit(NULL);
  }

  
 

  void set_output(char *outpath,
                  ssc_pwcdi_params *params,
                  ssc_pwcdi_plan *workspace,
                  int variable,
                  int complex_part){
//
    // GPU operation: 
    // moving workspace->dx to host "result"
    //


    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cufftComplex *result_c;
    short *result_s;
    
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
    }else if (variable == SSC_VARIABLE_SUPP){
      result_s = (short*) malloc(workspace->nvoxels*sizeof(short));
      
      if (workspace->timing){ 
        cudaEventRecord(start);
      }
  
      if (workspace->gpus->ngpus == 1){
        // Single GPU case
        checkCudaErrors(cudaMemcpy(result_s,
                                   workspace->sgpu.d_support,
                                   workspace->nvoxels*sizeof(short),
                                   cudaMemcpyDeviceToHost));
      }else{
        // Multi GPU case
        const size_t perGPUDim = workspace->nvoxels/workspace->gpus->ngpus;

        for (int i=0; i<workspace->gpus->ngpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
    
          checkCudaErrors(cudaMemcpy(&(result_s[i*perGPUDim]),
                                     workspace->mgpu.d_support[i], 
                                     perGPUDim*sizeof(short),
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
    //

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
    checkCudaErrors(cudaMemcpy((void*) d_y->descriptor->data[i], // d_z  (d_x) // versus cudaMemcpyAsync
                                    (void*) d_x->descriptor->data[i],
                                    perGPUDim*sizeof(cufftComplex),
                                    cudaMemcpyDeviceToDevice));
                                    // gpus->streams[i]));
  }     
  for (int i = 0; i <n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    getLastCudaError("ssc-cdi: error at memcpy.\n");
  }

  // this should work too, but it is not. I guess a driver update would solve  
  // checkCudaErrors(cufftXtMemcpy(plan_C2C,
  //                               d_y, // d_z (d_x)
  //                               d_x,
  //                               CUFFT_COPY_DEVICE_TO_DEVICE));


  // perform FFT on d_z 
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C, d_y, d_y, CUFFT_FORWARD)); // d_z  (d_x)

 
  // fix the ordering of d_z
  permuted2natural(d_y, plan_C2C, totalDim, host_swap); // d_z  (d_x)
 

  // adjust the phase of the fft of d_z, but copy the results directly to d_y
  // to = signal * exp( i * from ) / totalDim 
  for(int i = 0; i < n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    update_with_phase_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_y->descriptor->data[i], //to 
                                                           (cufftComplex*) d_y->descriptor->data[i], //from d_z  (d_x)
                                                           d_signal[i],
                                                           eps,
                                                           totalDim,
                                                           perGPUDim);
  }
  for(int i = 0; i < n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_phase_mgpu<<<.>>>\n");
  }
 
  // compute the ifft of d_y. Note that we didn't need to convert the subformat to natural ordering 
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C,
                                           d_y, //input
                                           d_y, //output
                                           CUFFT_INVERSE));

  // fix the ordering of d_y
  permuted2natural(d_y, plan_C2C, totalDim, host_swap); 
}
  




void m_projection_M_nswap1(cufftHandle& plan_C2C,
                           cudaLibXtDesc* d_y, //output  
                           cudaLibXtDesc* d_x, // input
                           cufftComplex* d_x_swap,
                           float** d_signal,
                           float eps,
                           size_t totalDim, 
                           size_t perGPUDim,
                           ssc_gpus *gpus,
                           cufftComplex* host_swap,
                           bool timing){
  
  //
  //
  // This assumes that both d_x->subFormat and d_y->subformat are equal to
  // CUFFT_XT_FORMAT_INPLACE. 

  int n_gpus = totalDim/perGPUDim;

  const dim3 threadsPerBlock(tbx*tby*tbz);
  const dim3 gridBlock(ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
     


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time_copydsignal;


  // // dynamic allocation: store d_x data in host pinned memory 
  // cufftComplex* d_x_swap = 
  // copy to host
  checkCudaErrors(cufftXtMemcpy(plan_C2C,
                                d_x_swap,
                                d_x,
                                CUFFT_COPY_DEVICE_TO_HOST));
 

  // perform FFT on d_z 
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C, d_x, d_x, CUFFT_FORWARD)); // (d_z, d_z)
 
  // fix the ordering of d_z
  permuted2natural(d_x, plan_C2C, totalDim, host_swap); // d_z
 
 
 
  // adjust the phase of the fft of d_z, but copy the results directly to d_y
  // to = signal * exp( i * from ) / totalDim 
  for(int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    update_with_phase_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_y->descriptor->data[i], //to
                                                           (cufftComplex*) d_x->descriptor->data[i], //from (d_z)
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


 

  // compute the ifft of d_y. Note that we didn't need to convert the subformat to natural ordering 
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_C2C,
                                           d_y, //input
                                           d_y, //output
                                           CUFFT_INVERSE));
 

  // fix the ordering of d_y
  permuted2natural(d_y, plan_C2C, totalDim, host_swap);




  // copy d_x_swap back to the device in d_x 
  checkCudaErrors(cufftXtMemcpy(plan_C2C,
                                d_x,
                                d_x_swap,
                                CUFFT_COPY_HOST_TO_DEVICE));
  }


// __global__ void m_fftshift_cufftcomplex(cufftComplex *d_x, 
//                                        cufftComplex *d_x_in, 
//                                        int dimension,
//                                        int n_gpus,
//                                        int gpuId) {
//   // ... (rest of the code is similar to gaussian1_freq_fftshift_mgpu)

//   const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//   const size_t globalIndex = index + gpuId * perGPUDim;

//   // map 1D index to 3D
//   index3 i, ii;
//   int ix, iy, iz;
  
//   i = get3Dindex(globalIndex, ndimy, ndimz);

//   // Handle multi-GPU FFT shift based on quadrant
//   int ddimx = dimension / n_gpus; // dimension per GPU

//   if (i.ix < ddimx) {
//     ix = i.ix;
//   } else {
//     ix = dimension - 1 - (i.ix - ddimx);
//   }

//   if (i.iy < dimension) {
//     iy = i.iy;
//   } else {
//     iy = dimension - 1 - (i.iy - dimension);
//   }

//   if (i.iz < dimension) {
//     iz = i.iz;
//   } else {
//     iz = dimension - 1 - (i.iz - dimension);
//   }

//   // Calculate the final index after FFT shift
//   size_t shiftedIndex = iz * dimension * dimension + iy * dimension + ix;

//   // Copy the data from input considering multi-GPU distribution
//   d_x[index].x = d_x_in[shiftedIndex].x;
//   d_x[index].y = d_x_in[shiftedIndex].y;
// }


// void m_fftshift_cufftcomplex(short** data, 
//                               size_t dimension, 
//                               int dtype, 
//                               cufftComplex* host_swap,
//                               size_t perGPUDim,
//                               ssc_gpus* gpus){
//   //
//   // Currently this only works for dtype=SSC_DTYPE_SHORT data, since it is
//   // small enough to be performed in a single GPU. The SSC_DTYPE_COMPLEX
//   // case should be handled in multiple GPUs.

//   printf("m_fftshift: dimension = %d\n", dimension);


//   const int n_gpus = (dimension*dimension*dimension)/perGPUDim;
//   const dim3 threadsPerBlock(tbx, tby, tbz);
//   const dim3 gridBlock (ceil((dimension + threadsPerBlock.x - 1) / threadsPerBlock.x),
//                         ceil((dimension + threadsPerBlock.y - 1) / threadsPerBlock.y),
//                         ceil((dimension + threadsPerBlock.z - 1) / threadsPerBlock.z));
      

  
//   // copy data (short**) to host_swap_short (short*)
//   for (int i=0; i<n_gpus; i++){
//     checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
//     checkCudaErrors(cudaMemcpy((void*) &host_swap_short[i*perGPUDim], // Async
//                                     (void*) data[i], 
//                                     perGPUDim*sizeof(cufftComplex), 
//                                     cudaMemcpyDeviceToHost)); 
//   }
//   for (int i=0; i<gpus->ngpus; ++i){
//     checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
//     checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
//   }


//   // fftshift in a single GPU 
//   fftshift<<<gridBlock, threadsPerBlock>>>((void*) host_swap,
//                                            dimension,
//                                            SSC_DTYPE_CUFFTCOMPLEX);
//   cudaDeviceSynchronize();
//   getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");

//   // copy back to data pointer
//   for (int i=0; i<n_gpus; i++){
//     checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
//     checkCudaErrors(cudaMemcpy((void*) data[i], //Async
//                                     (void*) &host_swap_short[i*perGPUDim],
//                                     perGPUDim*sizeof(cufftComplex), 
//                                     cudaMemcpyHostToDevice));   
//   }
//   for (int i=0; i<gpus->ngpus; ++i){
//     checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
//     checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
//   }
// }



void m_fftshift(short** data, 
                size_t dimension, 
                int dtype, 
                short* host_swap_short,
                size_t perGPUDim,
                ssc_gpus* gpus){  

  // Currently this only works for dtype=SSC_DTYPE_SHORT data, since it is
  // small enough to be performed in a single GPU. The SSC_DTYPE_COMPLEX
  // case should be handled in multiple GPUs.

  printf("m_fftshift: dimension = %d\n", dimension);


  const int n_gpus = (dimension*dimension*dimension)/perGPUDim;
  const dim3 threadsPerBlock(tbx, tby, tbz);
  const dim3 gridBlock (ceil((dimension + threadsPerBlock.x - 1)/threadsPerBlock.x),
                        ceil((dimension + threadsPerBlock.y - 1)/threadsPerBlock.y),
                        ceil((dimension + threadsPerBlock.z - 1)/threadsPerBlock.z));
      

  
  // copy data (short**) to host_swap_short (short*)
  for (int i=0; i<n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaMemcpy((void*) &host_swap_short[i*perGPUDim], // Async
                                    (void*) data[i], 
                                    perGPUDim*sizeof(short), 
                                    cudaMemcpyDeviceToHost));
                                    // gpus->streams[i])); // added this 
  }
  for (int i = 0; i<gpus->ngpus; ++i){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
    checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
  }


  // fftshift in a single GPU 
  fftshift<<<gridBlock, threadsPerBlock>>>((void*) host_swap_short,
                                           dimension,
                                           SSC_DTYPE_SHORT);
  cudaDeviceSynchronize();
  getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");

  // copy back to data pointer
  for (int i = 0; i < n_gpus; i++){
    checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
    checkCudaErrors(cudaMemcpy((void*) data[i], //Async
                                    (void*) &host_swap_short[i*perGPUDim],
                                    perGPUDim*sizeof(short), 
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
                      short** d_support,
                      int extraConstraint,
                      size_t totalDim,
                      size_t perGPUDim,
                      ssc_gpus *gpus){
    //
    // dz = support * dy + (1 - support) * dz
    //
    int n_gpus = totalDim/perGPUDim;
    const dim3 threadsPerBlock(tbx*tby*tbz);
    const dim3 gridBlock(ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
  
    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // decide wether to use the extra constraint kernel or the vanilla one 
      if (extraConstraint==NO_EXTRA_CONSTRAINT){
        // project onto the support alone 
        update_with_support_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z->descriptor->data[i], //to
                                                                 (cufftComplex*) d_y->descriptor->data[i], //from
                                                                 (short*) d_support[i], 
                                                                 perGPUDim); 
      }else{
        // project onto the support AND the phase constraint 
        update_with_support_extra_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z->descriptor->data[i], //to
                                                                       (cufftComplex*) d_y->descriptor->data[i], //from
                                                                       (short*) d_support[i], 
                                                                       extraConstraint,
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
                           short** d_support,
                           int extraConstraint,
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
      if (extraConstraint==NO_EXTRA_CONSTRAINT){
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
                                                                    extraConstraint,
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
                      short* d_support,
                      int extraConstraint,
                      int dimension){
    //
    // dz = support * dy + (1 - support) * dz
    
    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil(dimension + threadsPerBlock.x - 1)/threadsPerBlock.x,
                          ceil(dimension + threadsPerBlock.y - 1)/threadsPerBlock.y,
                          ceil(dimension + threadsPerBlock.z - 1)/threadsPerBlock.z);
  

    // decide wether to use the extra constraint kernel or the vanilla one 
    if (extraConstraint==NO_EXTRA_CONSTRAINT){
      // project onto the support alone 
      update_with_support<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z,
                                                          (cufftComplex*) d_y,
                                                          (short*) d_support,
                                                          dimension);
    }else{
      // project onto the support alone AND the phase constraint (extra constraint)
      update_with_support_extra<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_z,
                                                                (cufftComplex*) d_y,
                                                                (short*) d_support,
                                                                extraConstraint,
                                                                dimension);
    }

    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_with_support<<<.>>>\n");
 
  }
  
  void s_projection_S_only(cufftHandle& plan_input,
                           cufftComplex* d_z,
                           cufftComplex* d_y,
                           short* d_support,
                           int extraConstraint,
                           int dimension){
    //
    // d_z = support * dy 
   
    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil(dimension + threadsPerBlock.x - 1)/threadsPerBlock.x,
                          ceil(dimension + threadsPerBlock.y - 1)/threadsPerBlock.y,
                          ceil(dimension + threadsPerBlock.z - 1)/threadsPerBlock.z);


    // decide wether to use the extra constraint kernel or the vanilla one 
    if (extraConstraint==NO_EXTRA_CONSTRAINT){
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
                                                             extraConstraint,
                                                             dimension); 
    }

    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ multiply_support<<<.>>>\n");
  }

  

  void s_projection_extraConstraint(cufftComplex* d_x,
                                    cufftComplex* d_y,
                                    int extraConstraint,
                                    int dimension){
 
    
    const dim3 threadsPerBlock(tbx, tby, tbz);
    const dim3 gridBlock (ceil(dimension + threadsPerBlock.x - 1)/threadsPerBlock.x,
                          ceil(dimension + threadsPerBlock.y - 1)/threadsPerBlock.y,
                          ceil(dimension + threadsPerBlock.z - 1)/threadsPerBlock.z);
  


    update_extraConstraint<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_x,
                                                        (cufftComplex*) d_y,
                                                        extraConstraint,
                                                        dimension);
    cudaDeviceSynchronize();
    getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_extraConstraint<<<.>>>\n");
 
  }



    void m_projection_extraConstraint(cudaLibXtDesc* d_x,
                                      cudaLibXtDesc* d_y,
                                      int extraConstraint,
                                      size_t totalDim,
                                      size_t perGPUDim,
                                      ssc_gpus *gpus){
 
    int n_gpus = totalDim/perGPUDim;
    const dim3 threadsPerBlock(tbx*tby*tbz);
    const dim3 gridBlock (ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
  
    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // to = support * from
      update_extraConstraint_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) d_x->descriptor->data[i], //to
                                                            (cufftComplex*) d_y->descriptor->data[i], //from
                                                            extraConstraint, 
                                                            perGPUDim);
    }

    for(int i=0; i<n_gpus; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
       checkCudaErrors(cudaStreamSynchronize(gpus->streams[i])); 
      getLastCudaError("ssc-cdi: error / Kernel execution failed @ update_extraConstraint_mgpu<<<.>>>\n");
    }
  }
}   // extern "C"