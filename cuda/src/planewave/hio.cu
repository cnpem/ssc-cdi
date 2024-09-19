#include <iostream>
#include <math.h> 

#include "pwcdi.h"
#include "fft.h"
#include "gpus.h"
#include "pwutils.h"

extern "C"{

  void largest(float *max, float *arr, int n){
    *max = arr[0];

    for (int i=1; i<n; i++){
      if (arr[i] > *max)
        *max = arr[i];
    }
  }

  void smallest(float *min, float *arr, int n){
    *min = arr[0];

    for (int i=1; i<n; i++){
      if (arr[i] <*min)
        *min = arr[i];
    }
  }
 
  void hio(ssc_pwcdi_plan *workspace,
           ssc_pwcdi_params *params,
           int globalIteration,
           int shrinkWrapSubiter,
           int initialShrinkWrapSubiter,
           int extraConstraint,
           int extraConstraintSubiter,
           int initialExtraConstraintSubiter){

    if (workspace->gpus->ngpus == 1){
      // SINGLE-GPU VERSION
    
      float total, time_projM, time_projS, time_shrinkWrap, time_projExtra, time_computeErr;
      total = 0.0f;
      time_projExtra = 0.0f;
      time_computeErr = 0.0f;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      float initialBeta = params->beta;
      float beta = params->beta;
      float sigma = params->sigma;  
      float sigma_mult  = params->sigma_mult;
      float beta_update = params->beta_update;
      float sw_threshold = params->sw_threshold;  
      int sw_iter_filter = params->sw_iter_filter;
      int sw_mask_multiply = params->sw_mask_multiply;
    
      const dim3 threadsPerBlock(tbx, tby, tbz);
      const dim3 gridBlock (ceil((workspace->dimension + threadsPerBlock.x - 1)/threadsPerBlock.x),
                            ceil((workspace->dimension + threadsPerBlock.y - 1)/threadsPerBlock.y),
                            ceil((workspace->dimension + threadsPerBlock.z - 1)/threadsPerBlock.z));
      

      for (int iter=0; iter<globalIteration; ++iter){
        // ===============================================
        // Operation: s_projection_M
        
        checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[0]));
        cudaEventRecord(start);


        s_projection_M(workspace->plan_C2C,
                       workspace->sgpu.d_y, //to
                       workspace->sgpu.d_x, //from
                       workspace->sgpu.d_signal,
                       params->eps_zeroamp,
                       workspace->dimension);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_projM, start, stop);
        if (workspace->timing){
          fprintf(stdout,"ssc-cdi: s_projection_M() time: %lf ms\n", time_projM);
        }

 
        // =================================
        // Operation: s_projection_S
        
        // set timer 
        if (workspace->timing){
          cudaEventRecord(start);
        }

        // compute difference to be used in projection S
        set_difference<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_x,   //to
                                                       (cufftComplex*) workspace->sgpu.d_x,    //from
                                                       (cufftComplex*) workspace->sgpu.d_y,
                                                       1.0,
                                                       beta,
                                                       workspace->dimension);
        cudaDeviceSynchronize();
        getLastCudaError("ssc-cdi: error / kernel execution failed @ set_difference<<<.>>>\n");

        // operate projection_S
        if (extraConstraintSubiter<=0 && iter>initialExtraConstraintSubiter){
          // extraConstraint is applied inside projection_S
          s_projection_S(workspace->plan_C2C,
                         workspace->sgpu.d_x,
                         workspace->sgpu.d_y,
                         workspace->sgpu.d_support,
                         extraConstraint,
                         workspace->dimension);
        }else{
          // extraConstraint is applied separately, after the shrinkWrap
          s_projection_S(workspace->plan_C2C,
                         workspace->sgpu.d_x,
                         workspace->sgpu.d_y,
                         workspace->sgpu.d_support,
                         NO_EXTRA_CONSTRAINT,
                         workspace->dimension);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_projS, start, stop);
        if (workspace->timing){
          fprintf(stdout,"ssc-cdi: s_projection_S() time: %lf ms\n", time_projS);
        }

 

        // =====================================================================
        // shrink wrap operation
       
        if (iter%shrinkWrapSubiter==0 && iter>initialShrinkWrapSubiter){
          // set timer
          if (workspace->timing){
            cudaEventRecord(start);
          }
          
          // save d_x to d_x_swap if necessary 
          if (params->swap_d_x==true){
            checkCudaErrors(cudaMemcpy(workspace->sgpu.d_x_swap,  //to
                                       workspace->sgpu.d_x,       //from
                                       workspace->nvoxels*sizeof(cufftComplex),
                                       cudaMemcpyDeviceToHost));
          }
         
 

          // choose how the iteration variable is handled during the filtering operation
          // FILTER_AMPLITUDE uses the absolute value of d_y, while FILTER_FULL uses the 
          // full data
          if (sw_iter_filter == FILTER_AMPLITUDE){
            // take the absolute value of the iter variable 
            absolute_value<<<gridBlock, threadsPerBlock>>>(workspace->sgpu.d_y,   // dy 
                                                            workspace->sgpu.d_x,
                                                            workspace->dimension);
            cudaDeviceSynchronize();
            getLastCudaError("ssc-cdi: error / kernel execution failed @ absolute_value<<<.>>>\n");  


            // compute fft  
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_y, // without the absolute value it is just d_x
                                         workspace->sgpu.d_y,
                                         CUFFT_FORWARD)); 

          }else if (sw_iter_filter == FILTER_FULL){
            // compute fft of the iter variable directly
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_x, // without the absolute value it is just d_x
                                         workspace->sgpu.d_y,
                                         CUFFT_FORWARD)); 

          }

          // create classical gaussian with std sigma
          if (params->swap_d_x==false){
            gaussian1<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_gaussian,
                                                      sigma,
                                                      workspace->dimension);
            cudaDeviceSynchronize();
            getLastCudaError("ssc-cdi: error / kernel execution failed @ gaussian1<<<.>>>\n");


          // fftshift the gaussian kernel 
          // fftshift<<<gridBlock, threadsPerBlock>>>((void *) workspace->sgpu.d_gaussian,
          //                                            workspace->dimension,
          //                                            SSC_DTYPE_CUFFTCOMPLEX);
          // cudaDeviceSynchronize();
          // getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");


            // // gaussian = FFT(gaussian)
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_gaussian,
                                         workspace->sgpu.d_gaussian,
                                         CUFFT_FORWARD));
   
   
            // This performs the convolution of d_y in with the blurring kernel as a multiplication
            // in the Fourier domain. The multiplication can be performed by using both real and
            // imaginary components of the blurring mask, or only its real component. 
            if (sw_mask_multiply == MULTIPLY_FULL){
              multiply<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                       (cufftComplex*) workspace->sgpu.d_gaussian,
                                                       (cufftComplex*) workspace->sgpu.d_y,
                                                       workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply<<<.>>>\n");
            }else if (sw_mask_multiply == MULTIPLY_REAL){
              multiply_rc<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_gaussian,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }else if (sw_mask_multiply == MULTIPLY_LEGACY){
              multiply_legacy<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_gaussian,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }

          // if d_x is to be swapped to the host during shrinkWrap, then create the 3D Gaussian
          // function and store it in d_x directly. 
          }else if (params->swap_d_x==true){
            gaussian1<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_x,
                                                      sigma,
                                                      workspace->dimension);
            cudaDeviceSynchronize();
            getLastCudaError("ssc-cdi: error / kernel execution failed @ gaussian1<<<.>>>\n");

            // fftshift the gaussian kernel 
            // fftshift<<<gridBlock, threadsPerBlock>>>((void *) workspace->sgpu.d_gaussian,
            //                                            workspace->dimension,
            //                                            SSC_DTYPE_CUFFTCOMPLEX);
            // cudaDeviceSynchronize();
            // getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");


            // // gaussian = FFT(gaussian)
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_x,
                                         workspace->sgpu.d_x,
                                         CUFFT_FORWARD));
   
   
            // This performs the convolution of d_y in with the blurring kernel as a multiplication
            // in the Fourier domain. The multiplication can be performed by using both real and
            // imaginary components of the blurring mask, or only its real component. 
            if (sw_mask_multiply == MULTIPLY_FULL){
              multiply<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                       (cufftComplex*) workspace->sgpu.d_x,
                                                       (cufftComplex*) workspace->sgpu.d_y,
                                                       workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply<<<.>>>\n");
            }else if (sw_mask_multiply == MULTIPLY_REAL){
              multiply_rc<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_x,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }else if (sw_mask_multiply == MULTIPLY_LEGACY){
              multiply_legacy<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_x,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }

            // restore d_x from host 
            checkCudaErrors(cudaMemcpy(workspace->sgpu.d_x,             //to
                                       workspace->sgpu.d_x_swap,       //from
                                       workspace->nvoxels*sizeof(cufftComplex),
                                       cudaMemcpyHostToDevice));
          }



          // dy = IFFT(dy)
          checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                       workspace->sgpu.d_y,
                                       workspace->sgpu.d_y,
                                       CUFFT_INVERSE));
    
          // normalize 
          normalize<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                    workspace->dimension);
          cudaDeviceSynchronize();
          getLastCudaError("ssc-cdi: error / kernel execution failed @ normalize<<<.>>>\n");
      
  
 

          // find max value
          float globalMax;
          cufftComplex cglobalMax;
          int idxMax;
          cublasHandle_t handle;
          cublasStatus_t stat;
          cublasCreate(&handle);
        
          stat = cublasIcamax(handle, workspace->dimension, workspace->sgpu.d_y, 1, &idxMax);
          
          if (stat == CUBLAS_STATUS_NOT_INITIALIZED )
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
          
          if (stat == CUBLAS_STATUS_ALLOC_FAILED )
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
          
          if (stat == CUBLAS_STATUS_EXECUTION_FAILED )
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
          
          if (stat == CUBLAS_STATUS_INVALID_VALUE )
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
          
          cudaMemcpy(&cglobalMax,
               (cufftComplex*) workspace->sgpu.d_y + idxMax - 1,
               sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);
          
          globalMax = sqrtf(powf(fabs(cglobalMax.x),2.0) + powf(fabs(cglobalMax.y),2.0));
          cublasDestroy(handle);
      
          //find min value
          float globalMin;
          cufftComplex cglobalMin;
          int idxMin;
          cublasHandle_t handleMin;
          cublasStatus_t statMin;
          cublasCreate(&handleMin);
        
          statMin = cublasIcamin(handleMin, workspace->dimension, workspace->sgpu.d_y, 1, &idxMin);
          
          if (statMin == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
          
          if (statMin == CUBLAS_STATUS_ALLOC_FAILED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
          
          if (statMin == CUBLAS_STATUS_EXECUTION_FAILED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
          
          if (statMin == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
          
          cudaMemcpy(&cglobalMin,
               (cufftComplex*) workspace->sgpu.d_y + idxMin - 1,
               sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);
          
          globalMin = sqrtf(powf(fabs(cglobalMin.x),2.0) + powf(fabs(cglobalMin.y),2.0));
          cublasDestroy(handleMin);
    
          printf("ssc-cdi: GlobalMax (single gpu) = %lf \n", globalMax);
          printf("ssc-cdi: GlobalMin (single gpu) = %lf \n", globalMin);

          // dsupp = abs( dy ) > (threshold/100) * Max ( abs(dy) )
          //
          // false: use globalMax as value for threshold
          // true: use index(globalMax) as the index for the thresholing inside kernel
          update_support_sw<<<gridBlock, threadsPerBlock>>>(workspace->sgpu.d_support,
                                                            workspace->sgpu.d_y,
                                                            sw_threshold,
                                                            globalMax,
                                                            globalMin, //0.0f, // globalMin, // 0.0f;
                                                            idxMax,
                                                            idxMin,
                                                            workspace->dimension,
                                                            false);
          cudaDeviceSynchronize();
          getLastCudaError("ssc-cdi: error / kernel execution failed @ update_support_sw<<<.>>>\n");

          // // fftshift the support computing it    
          fftshift<<<gridBlock, threadsPerBlock>>>((void *) workspace->sgpu.d_support,
                                                     workspace->dimension,
                                                     SSC_DTYPE_SHORT);
          cudaDeviceSynchronize();
          getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");


          

          //  update sigma 
          sigma = sigma_mult * sigma;
          
          // // update beta
          // if (params->betaResetSubiter>0 && iter%params->betaResetSubiter==0 && iter>0){
          //   // reset beta
          //   beta = initialBeta;
          // }else{
          //   // update beta
          //   beta = initialBeta + (1 - initialBeta)*(1 - exp( -(iter/beta_update)*(iter/beta_update)*(iter/beta_update)));
          // }

          // debug new values for beta and sigma  
          printf("ssc-cdi: new sigma = %lf \n",sigma);
          printf("ssc-cdi: new beta = %lf \n",beta);


          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_shrinkWrap, start, stop);
          if (workspace->timing){
            fprintf(stdout,"ssc-cdi: ShrinkWrap() time: %lf ms\n", time_shrinkWrap);   
          }
      
        }else{
          time_shrinkWrap = 0;
        }        


        // update beta
        if (params->betaResetSubiter>0 && iter%params->betaResetSubiter==0 && iter>0){
          // reset beta
          beta = initialBeta;
        }else{
          // update beta
          beta = initialBeta + (1 - initialBeta)*(1 - exp( -(iter/beta_update)*(iter/beta_update)*(iter/beta_update)));
        }


      // ===============================================
      // Operation s_projection_extraConstraint


        // operation projection extraconstraint 
        if (extraConstraint != NO_EXTRA_CONSTRAINT &&
            extraConstraintSubiter>0 && 
            iter>initialExtraConstraintSubiter && iter%extraConstraintSubiter==0){

          // set timer 
          if (workspace->timing){
           cudaEventRecord(start);
          }
          
          // perform the projection 
          s_projection_extraConstraint(workspace->sgpu.d_x,
                                       workspace->sgpu.d_x,  
                                       extraConstraint,
                                       workspace->dimension);

          // stop timer
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_projExtra, start, stop);
          if (workspace->timing){
            fprintf(stdout,"ssc-cdi: s_projection_extra() time: %lf ms\n", time_projExtra);   
          }
        }


        // ===============================================
        // Operation compute errors 

 
        // compute and store errors 
        if (params->errType!=NO_ERR && iter%params->errSubiter==0){
          // set timer 
          if (workspace->timing){
           cudaEventRecord(start);
          }

          // ITER_DIFF case 
          if (params->errType==ITER_DIFF){
            if (iter>params->errSubiter){
              // if this is not the first subiter, then compute the error
              set_difference<<<gridBlock, threadsPerBlock>>>(workspace->sgpu.d_x_lasterr,  
                                                                 workspace->sgpu.d_x, 
                                                                 workspace->sgpu.d_x_lasterr,
                                                                 1.0, 
                                                                 1.0,
                                                                 workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / Kernel execution failed @ set_difference<<<.>>>\n");

              // compute norm of the real and imaginary parts 
              cublasHandle_t handle;
              cublasStatus_t stat;
              cublasCreate(&handle);

              float iter_diff_norm;
              stat =  cublasScnrm2_v2(handle, 
                                      powf(workspace->dimension,3.0), 
                                      workspace->sgpu.d_x, 
                                      1, 
                                      &iter_diff_norm);
              if (stat == CUBLAS_STATUS_NOT_INITIALIZED)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2x<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              if (stat == CUBLAS_STATUS_ALLOC_FAILED)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              if (stat == CUBLAS_STATUS_EXECUTION_FAILED)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              if (stat == CUBLAS_STATUS_INVALID_VALUE)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");

              // destroy the handle 
              cublasDestroy(handle) ;

              // debug computed norm
              fprintf(stdout, "ssc-cdi: Computed norm (iter_diff) at iteration %d = %f\n", iter, iter_diff_norm);



            }else{
              // if this is the first subiter, simply store the current iteration
              set_difference<<<gridBlock, threadsPerBlock>>>(workspace->sgpu.d_x_lasterr,  
                                                                 workspace->sgpu.d_x, 
                                                                 workspace->sgpu.d_x,
                                                                 1.0, 
                                                                 0.0,
                                                                 workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / Kernel execution failed @ set_difference<<<.>>>\n");
              
            }
          }

          // stop timer 
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_computeErr, start, stop);
          if (workspace->timing){
            fprintf(stdout,"ssc-cdi: computeErr() time: %lf ms\n", time_computeErr);   
          }
        }


        // debug timers 
        if (workspace->timing){
          total = total + (time_projM + time_projS + time_shrinkWrap + time_projExtra + time_computeErr);
          fprintf(stdout,
                  "ssc-cdi: Iteration %d takes %lf ms ** \n\n", 
                  iter, 
                  time_projM + time_projS + time_shrinkWrap + time_projExtra + time_computeErr); 
        }



    
      } // end iteration 

      if (workspace->timing){
        printf("ssc-cdi: total time with HIO iterations is %lf ms\n", total); 
      }
 


 
    }else{
      // MULTI-GPU VERSION
  
      int *idxMaxvalue;
      int *idxMinvalue;
      float *maxvalue;
      float *minvalue;
      cufftComplex *cmaxvalue;
      cufftComplex *cminvalue;
      // cudaError_t status; 


  
      float total, time_projM, time_projS, time_shrinkWrap, time_projExtra, time_computeErr;
      total = 0.0f;
      time_projExtra = 0.0f;
      time_computeErr = 0.0f;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
    
      const size_t dim = (size_t) workspace->nvoxels;
      const int n_gpus =  (int) workspace->gpus->ngpus;
      const size_t perGPUDim = (size_t) (dim/n_gpus);

      const dim3 threadsPerBlock(tbx*tby*tbz);
      const dim3 gridBlock (ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));

      idxMaxvalue = (int *)malloc( sizeof(int) * n_gpus);
      maxvalue    = (float *)malloc( sizeof(float) * n_gpus);
      cmaxvalue   = (cufftComplex *)malloc( sizeof(cufftComplex) * n_gpus);
      idxMinvalue = (int *)malloc( sizeof(int) * n_gpus);
      minvalue    = (float *)malloc( sizeof(float) * n_gpus);
      cminvalue   = (cufftComplex *)malloc( sizeof(cufftComplex) * n_gpus);

      // status = cudaMallocHost((void**)&idxMaxvalue, (size_t) (n_gpus*sizeof(int)));
      // status = cudaMallocHost((void**)&maxvalue, (size_t) (n_gpus*sizeof(float)));
      // status = cudaMallocHost((void**)&cmaxvalue, (size_t) (n_gpus*sizeof(cufftComplex)));

    
      float initialBeta = params->beta;
      float beta = params->beta;
      float sw_threshold = params->sw_threshold;
      int sw_iter_filter = params->sw_iter_filter;
      int sw_mask_multiply = params->sw_mask_multiply;
      float sigma = params->sigma; //0.1;  
      float sigma_mult = params->sigma_mult;
      float beta_update = params->beta_update;


    
      //-----------------------------------------------------------
      // start iterations
      //-----------------------------------------------------------

      // reminder: workspace->dx already contains the initial point
       

      for (int iter=0; iter<globalIteration; iter++){
        // =======================================================
        // operation m_projection_M

        // start timer
        if (workspace->timing){  
          cudaEventRecord(start);
        }

        m_projection_M(workspace->plan_C2C,
                       workspace->mgpu.d_y,
                       workspace->mgpu.d_x,  
                       workspace->mgpu.d_signal,
                       params->eps_zeroamp,
                       dim,
                       perGPUDim,
                       workspace->gpus,
                       workspace->host_swap,
                       workspace->timing);


        if (workspace->timing){  
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_projM, start, stop);
          printf("ssc-cdi: m_projection_M() time: %lf ms\n", time_projM);
        }


        // ==================================
        // operation: projection_S
        
        // set timer
        if (workspace->timing){  
          cudaEventRecord(start);
        } 

        // set difference d_x = 1*d_x - beta*d_y
        for(int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));

          // compute difference kernel 
          set_difference_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], // output
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], // input1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // input2 
                                                              1.0f,
                                                              beta ,
                                                              perGPUDim);  
        }

        for(int i=0; i<n_gpus; i++){
          checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
          checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
          getLastCudaError("ssc-cdi: error / kernel execution failed @ set_difference_mgpu<<<.>>>\n");
        }
 
 
        // set timer        
        if (workspace->timing){  
          cudaEventRecord(start);
        }
         
        // compute the projection
        if (extraConstraintSubiter<=0 && iter>initialExtraConstraintSubiter){
          // extraConstraint is applied inside projection_S
          m_projection_S(workspace->mgpu.d_x, //  
                         workspace->mgpu.d_y,              
                         workspace->mgpu.d_support,
                         extraConstraint,   
                         dim,
                         perGPUDim,
                         workspace->gpus);
        }else{
          // extraConstraint is applied separately, after shrinkWrap
          m_projection_S(workspace->mgpu.d_x, //  
                         workspace->mgpu.d_y,              
                         workspace->mgpu.d_support,
                         NO_EXTRA_CONSTRAINT,   
                         dim,
                         perGPUDim,
                         workspace->gpus);
        }

        // stop timer 
        if (workspace->timing){  
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_projS, start, stop);
          printf("ssc-cdi: m_projection_S(): %lf ms\n", time_projS);
        }


        // ============================================================
        // operation: shrink wrap  

        // set timer 
        if (workspace->timing){  
          cudaEventRecord(start);
        }

        if (iter%shrinkWrapSubiter==0 && iter>initialShrinkWrapSubiter){

          // store no pointer during shrinkWrap
          if (params->swap_d_x==false){

            // copy the content of d_x, which is in natural ordering format, to d_s
            // note that this copy is done directly on the content.
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              checkCudaErrors(cudaMemcpy(workspace->mgpu.d_z->descriptor->data[i],          //Async
                                              workspace->mgpu.d_x->descriptor->data[i],
                                              perGPUDim*sizeof(cufftComplex), 
                                              cudaMemcpyDeviceToDevice));
                                              // workspace->gpus->streams[i]));
            }     
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
            }
            // or this 
            // checkCudaErrors(cufftXtMemcpy(workspace->plan_C2C,
            //   workspace->mgpu.d_z,
            //   workspace->mgpu.d_x,
            //   CUFFT_COPY_DEVICE_TO_DEVICE));

            // choose how the iteration variable is handled during the filtering operation

            // create shifted Gaussian directly 
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              
              if (params->sw_fftshift_gaussian == true){
                gaussian1_freq_fftshift_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_y->descriptor->data[i],
                                                                             sigma,
                                                                             workspace->dimension,
                                                                             n_gpus,
                                                                             i);
              }else{
                gaussian1_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*)workspace->mgpu.d_y->descriptor->data[i],
                                                               sigma,
                                                               workspace->dimension,
                                                               n_gpus,
                                                               i);
              }
            }
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              getLastCudaError("ssc-cdi: error / kernel execution failed @ gaussian1_freq_fftshift_mgpu<<<.>>>\n");
            }


            // choose how the iteration variable is handled during the filtering operation
            // FILTER_AMPLITUDE uses the absolute value of d_y, while FILTER_FULL uses the 
            // full data
            if (sw_iter_filter==FILTER_AMPLITUDE){
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                absolute_value_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*)workspace->mgpu.d_z->descriptor->data[i],
                                                                     (cufftComplex*)workspace->mgpu.d_z->descriptor->data[i],
                                                                     workspace->nvoxels, 
                                                                     perGPUDim);
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ absolute_value_mgpu<<<.>>>\n");
              }
            }else if (sw_iter_filter==FILTER_FULL){
            }
            
            // FFT of d_y, which is holding the gaussian volume
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_y, workspace->mgpu.d_y, CUFFT_FORWARD));


            // ds = FFT(ds) : after FFT forward, ds is in natural order
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C,  workspace->mgpu.d_z, workspace->mgpu.d_z, CUFFT_FORWARD));    // (d_s,d_s)

            // permuted2natural(workspace->mgpu.d_y, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);
            // permuted2natural(workspace->mgpu.d_z, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);

   
            // This performs the convolution of d_y in with the blurring kernel as a multiplication
            // in the Fourier domain. The multiplication can be performed by using both real and
            // imaginary components of the blurring mask, or only its real component. 
            if (sw_mask_multiply == MULTIPLY_FULL){
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                multiply_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], //to
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1 
                                                              (cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], //from2  (d_x)
                                                              perGPUDim);     
              }
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_mgpu<<<.>>>\n");
              }
            }else if (sw_mask_multiply == MULTIPLY_REAL){
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors( cudaSetDevice(  workspace->gpus->gpus[i] ) );
                multiply_rc_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], //to
                                                                  (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1  
                                                                  (cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], //from2  (d_x)
                                                                  perGPUDim);  
              }
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc_mgpu<<<.>>>\n");
              }
            }else if (sw_mask_multiply==MULTIPLY_LEGACY){

              for(int i=0; i<n_gpus; i++){
                checkCudaErrors( cudaSetDevice(  workspace->gpus->gpus[i] ) );
                multiply_legacy_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], //to
                                                                    (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1  
                                                                    (cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], //from2  (d_x)
                                                                    perGPUDim);  
              }
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc_mgpu<<<.>>>\n");
              }
            }
 
            // dz = IFFT(dz)
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_z, workspace->mgpu.d_z, CUFFT_INVERSE));


            // dz = normalize(dz)
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              normalize_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_z->descriptor->data[i],
                                                             workspace->nvoxels,
                                                             perGPUDim); 
            }
            for(int i = 0; i < n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
              getLastCudaError("ssc-cdi: error / kernel execution failed @ normalize_mgpu<<<.>>>\n");
            }

            // permuted2natural(workspace->mgpu.d_z, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);
            workspace->mgpu.d_y->subFormat = CUFFT_XT_FORMAT_INPLACE;


            //     // debug mgpu (cudaLibXtDesc*)
            // printf("MGPU NORMALIZER = %d\n",workspace->nvoxels);
            // workspace->mgpu.d_x->subFormat = CUFFT_XT_FORMAT_INPLACE;
            // for(int i = 0; i < n_gpus; i++){
            //   checkCudaErrors( cudaSetDevice(  workspace->gpus->gpus[i] ) );
            //   checkCudaErrors(cudaMemcpy(workspace->mgpu.d_x->descriptor->data[i],  //to
            //    workspace->mgpu.d_y->descriptor->data[i],  //from
            //    perGPUDim * sizeof(cufftComplex),
            //    cudaMemcpyDeviceToDevice)); 
            // } 
            // return;


            // int *version = (int*)malloc(sizeof(int));
            // cudaDeviceSynchronize();
            // cufftGetVersion(version);
            // printf("version ======= %d",*version);
     

            // fix the ordering of d_y, but it's not necessary to be executed since ifft already fixes the indexes
            // permuted2natural(workspace->mgpu.d_y, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);

 
            // find max value of abs(d_y)
            // TODO: unify both calls into a single one 
            //https://forums.developer.nvidia.com/t/how-to-get-the-max-value-in-a-vector-when-using-cublas/15360

            float globalMax;
            cublasHandle_t handle;
            cublasStatus_t stat;
            cublasCreate(&handle);

            // globalmin
            float globalMin;
            cublasHandle_t handleMin;
            cublasStatus_t statMin;
            cublasCreate(&handleMin);

     
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              stat = cublasIcamax(handle, perGPUDim, (const cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], 1, &idxMaxvalue[i] );
              statMin = cublasIcamin(handleMin, perGPUDim, (const cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], 1, &idxMinvalue[i] );
     

              if (stat == CUBLAS_STATUS_NOT_INITIALIZED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (stat == CUBLAS_STATUS_ALLOC_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (stat == CUBLAS_STATUS_EXECUTION_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (stat == CUBLAS_STATUS_INVALID_VALUE ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     


              if (statMin == CUBLAS_STATUS_NOT_INITIALIZED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (statMin == CUBLAS_STATUS_ALLOC_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (statMin == CUBLAS_STATUS_EXECUTION_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (statMin == CUBLAS_STATUS_INVALID_VALUE ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     

              cudaMemcpy(&cmaxvalue[i],
                 (cufftComplex *)workspace->mgpu.d_z->descriptor->data[i] + idxMaxvalue[i]-1,
                 sizeof(cufftComplex),
                 cudaMemcpyDeviceToHost);

              maxvalue[i] = sqrtf(powf(fabs(cmaxvalue[i].x),2.0) + powf(fabs(cmaxvalue[i].y), 2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",maxvalue[i] );


              cudaMemcpy(&cminvalue[i],
                 (cufftComplex *)workspace->mgpu.d_z->descriptor->data[i] + idxMinvalue[i]-1,
                 sizeof(cufftComplex),
                 cudaMemcpyDeviceToHost);

              minvalue[i] = sqrtf(powf(fabs(cminvalue[i].x),2.0) + powf(fabs(cminvalue[i].y), 2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",minvalue[i]);

            }
        
            cublasDestroy(handle);
            largest (&globalMax, maxvalue, n_gpus);
            cublasDestroy(handleMin);
            smallest (&globalMin, minvalue, n_gpus);
            
            // sync all gpus
            for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
            }

            // debug 
            fprintf(stdout,"ssc-cdi: globalMax:= %lf\n", globalMax);     
            fprintf(stdout,"ssc-cdi: globalMin:= %lf\n", globalMin);     


  

 
            // update support 
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              
              // dsupp = abs( dz ) > (threshold/100) * Max ( abs(dz) )
              update_support_sw_mgpu<<<gridBlock, threadsPerBlock>>>(workspace->mgpu.d_support[i],
                                                                    (cufftComplex *) workspace->mgpu.d_z->descriptor->data[i],
                                                                    sw_threshold,
                                                                    globalMax,
                                                                    globalMin, //0.0f, //globalMin,
                                                                    perGPUDim);
            }
        
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              getLastCudaError("ssc-cdi: error / kernel execution failed @  update_support_sw_mgpu<<<.>>>\n");
            }
     
            
            if (params->sw_fftshift_gaussian==false){
              // fftshift the support after computing it 
              m_fftshift(workspace->mgpu.d_support, 
                                    (size_t) cbrt((double)workspace->nvoxels), 
                                    SSC_DTYPE_SHORT, 
                                    workspace->host_swap_short,
                                    perGPUDim,
                                    workspace->gpus);
            }
          


            //  update sigma 
            sigma = sigma_mult * sigma;
            
            // // update beta
            // if (params->betaResetSubiter>0 && iter%params->betaResetSubiter==0 && iter>0){
            //   // reset beta
            //   beta = initialBeta;
            // }else{
            //   // update beta
            //   beta = initialBeta + (1 - initialBeta)*(1 - exp( -(iter/beta_update)*(iter/beta_update)*(iter/beta_update)));
            // }



        
          // store one pointer during shrinkwrap
          }else if (params->swap_d_x==true){
            // store d_x data in pinned host memory. d_x will be used to perform the shrinkWrap operation
            checkCudaErrors(cufftXtMemcpy(workspace->plan_C2C,
                                          workspace->mgpu.d_x_swap,
                                          workspace->mgpu.d_x,
                                          CUFFT_COPY_DEVICE_TO_HOST));
        

            // compute convolution(d_x, gaussian)
         
            // create shifted Gaussian directly 
            for(int i = 0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              if (params->sw_fftshift_gaussian==true){
                gaussian1_freq_fftshift_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*)workspace->mgpu.d_y->descriptor->data[i],
                                                                             sigma,
                                                                             workspace->dimension,
                                                                             n_gpus,
                                                                             i);
              }else{
                gaussian1_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*)workspace->mgpu.d_y->descriptor->data[i],
                                                               sigma,
                                                               workspace->dimension,
                                                               n_gpus,
                                                               i);
              }
            }
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              getLastCudaError("ssc-cdi: error / kernel execution failed @ gaussian1_freq_fftshift_mgpu<<<.>>> or gaussian1<<<<.>>>>\n");
            }

            // choose how the iteration variable is handled during the filtering operation
            // FILTER_AMPLITUDE uses the absolute value of d_y, while FILTER_FULL uses the 
            // full data
            if (sw_iter_filter == FILTER_AMPLITUDE){
              for (int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                absolute_value_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*)workspace->mgpu.d_x->descriptor->data[i],
                                                                    (cufftComplex*)workspace->mgpu.d_x->descriptor->data[i],
                                                                    workspace->nvoxels,
                                                                    perGPUDim);
              }
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ absolute_value_mgpu<<<.>>>\n");
              }
            }else if (sw_iter_filter==FILTER_FULL){
            }
   
            // FFT of d_y, which is holding the gaussian volume
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_y, workspace->mgpu.d_y, CUFFT_FORWARD));

   
            // ds = FFT(ds) : after FFT forward, ds is in natural order
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_x, workspace->mgpu.d_x, CUFFT_FORWARD));    // (d_s,d_s)

            // permuted2natural(workspace->mgpu.d_y, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);
            // permuted2natural(workspace->mgpu.d_z, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);
   
   
            // This performs the convolution of d_y in with the blurring kernel as a multiplication
            // in the Fourier domain. The multiplication can be performed by using both real and
            // imaginary components of the blurring mask, or only its real component. 
            if (sw_mask_multiply==MULTIPLY_FULL){
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          
                multiply_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                              // (cufftComplex*) workspace->mgpu.d_gaussian[i], //from1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1 
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //from2  (d_s)
                                                              perGPUDim); 
                      
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_mgpu<<<.>>>\n");
              }
            }else if(sw_mask_multiply==MULTIPLY_REAL){
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                multiply_rc_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                              // (cufftComplex*) workspace->mgpu.d_gaussian[i], //from1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1 
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //from2  (d_s)
                                                              perGPUDim); 
                      
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc_mgpu<<<.>>>\n");
              }
            }else if(sw_mask_multiply==MULTIPLY_LEGACY){
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          
                multiply_legacy_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                              // (cufftComplex*) workspace->mgpu.d_gaussian[i], //from1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1 
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //from2  (d_s)
                                                              perGPUDim); 
                      
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc_mgpu<<<.>>>\n");
              }
            }
   
            // compute IFFT
            checkCudaErrors( cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_x, workspace->mgpu.d_x, CUFFT_INVERSE) );
         

            // normalize
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));

              normalize_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i],
                                                             workspace->nvoxels,
                                                             perGPUDim); 
            }
            for(int i = 0; i < n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
              getLastCudaError("ssc-cdi: error / kernel execution failed @ normalize_mgpu<<<.>>>\n");
            }

            // permuted2natural(workspace->mgpu.d_x, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);
            workspace->mgpu.d_y->subFormat = CUFFT_XT_FORMAT_INPLACE;

            // fix the ordering of d_y, but it's not necessary to be executed since ifft already fixes the indexes
            // permuted2natural(workspace->mgpu.d_y, workspace->plan_C2C, workspace->nvoxels, workspace->host_swap);

            // find min, max value of iter 
            // todo: unify these calls into a single one 
            //https://forums.developer.nvidia.com/t/how-to-get-the-max-value-in-a-vector-when-using-cublas/15360

            float globalMax;
            cublasHandle_t handle;
            cublasStatus_t stat;
            cublasCreate(&handle);

            // globalmin
            float globalMin;
            cublasHandle_t handleMin;
            cublasStatus_t statMin;
            cublasCreate(&handleMin);

     
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              stat = cublasIcamax(handle, perGPUDim, (const cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 1, &idxMaxvalue[i]);
              statMin = cublasIcamin(handleMin, perGPUDim, (const cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 1, &idxMinvalue[i]);
     

              if (stat == CUBLAS_STATUS_NOT_INITIALIZED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (stat == CUBLAS_STATUS_ALLOC_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (stat == CUBLAS_STATUS_EXECUTION_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (stat == CUBLAS_STATUS_INVALID_VALUE ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     


              if (statMin == CUBLAS_STATUS_NOT_INITIALIZED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (statMin == CUBLAS_STATUS_ALLOC_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (statMin == CUBLAS_STATUS_EXECUTION_FAILED ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (statMin == CUBLAS_STATUS_INVALID_VALUE ){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     

              cudaMemcpy(&cmaxvalue[i],
                         (cufftComplex *)workspace->mgpu.d_x->descriptor->data[i] + idxMaxvalue[i]-1,
                         sizeof(cufftComplex),
                         cudaMemcpyDeviceToHost);

              maxvalue[i] = sqrtf(powf(fabs(cmaxvalue[i].x),2.0) + powf(fabs(cmaxvalue[i].y), 2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",maxvalue[i] );


              cudaMemcpy(&cminvalue[i],
                         (cufftComplex *)workspace->mgpu.d_x->descriptor->data[i] + idxMinvalue[i]-1,
                         sizeof(cufftComplex),
                         cudaMemcpyDeviceToHost);

              minvalue[i] = sqrtf(powf(fabs(cminvalue[i].x),2.0) + powf(fabs(cminvalue[i].y),2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",minvalue[i]);

            }
        
            cublasDestroy(handle);
            largest (&globalMax, maxvalue, n_gpus);
            cublasDestroy(handleMin);
            smallest (&globalMin, minvalue, n_gpus);


            fprintf(stdout,"ssc-cdi: computed globalMax:= %lf\n", globalMax);     
            fprintf(stdout,"ssc-cdi: computed globalMin:= %lf\n", globalMin);     

            // sync
            for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
            }



            // update support, other variables  
            // update support 
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              update_support_sw_mgpu<<<gridBlock, threadsPerBlock>>>(workspace->mgpu.d_support[i],
                                                                    (cufftComplex *) workspace->mgpu.d_x->descriptor->data[i],
                                                                    sw_threshold,
                                                                    globalMax,
                                                                    globalMin, //0.0f, //globalMin,
                                                                    perGPUDim);
            }
            for(int i = 0; i < n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              getLastCudaError("ssc-cdi: error / kernel execution failed @ update_support_sw_mgpu<<<.>>>\n");
            }
     
            
            if (params->sw_fftshift_gaussian==false){
              // fftshift the support after computing it 
              m_fftshift(workspace->mgpu.d_support, 
                         (size_t) cbrt((double)workspace->nvoxels), 
                         SSC_DTYPE_SHORT, 
                         workspace->host_swap_short,
                         perGPUDim,
                         workspace->gpus);
            }


            //  update sigma 
            sigma = sigma_mult * sigma;
            
            
            // restore d_x from host to device
            checkCudaErrors(cufftXtMemcpy(workspace->plan_C2C,
                                          workspace->mgpu.d_x,
                                          workspace->mgpu.d_x_swap,
                                          CUFFT_COPY_HOST_TO_DEVICE));
   

          } // end of the second case of nswap_vars


 
        }else{ 
          time_shrinkWrap = 0;
        } // end of shrinkwrap

        // stop timer 
        if (workspace->timing){
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_shrinkWrap, start, stop);
          fprintf(stdout, "ssc-cdi: time_shrinkWrap = %f\n", time_shrinkWrap);
        }


        // update beta
        if (params->betaResetSubiter>0 && iter%params->betaResetSubiter==0 && iter>0){
          // reset beta
          beta = initialBeta;
        }else{
          // update beta
          beta = initialBeta + (1 - initialBeta)*(1 - exp( -(iter/beta_update)*(iter/beta_update)*(iter/beta_update)));
        }


        // ==========================
        // operation: m_projection_extraConstraint

        // projection_extraConstraint is only performed in the following case
        if (extraConstraint != NO_EXTRA_CONSTRAINT &&
            extraConstraintSubiter>0 && 
            iter>initialExtraConstraintSubiter && iter%extraConstraintSubiter==0){

          // set timer 
          if (workspace->timing){  
            cudaEventRecord(start);
          }

          // perform the projection with extra constraint
          m_projection_extraConstraint(workspace->mgpu.d_x,
                                       workspace->mgpu.d_x,  
                                       extraConstraint, 
                                       dim,
                                       perGPUDim,
                                       workspace->gpus);
          // stop timer
          if (workspace->timing){  
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_projExtra, start, stop);
            fprintf(stdout, "ssc-cdi: time_projExtra = %f\n", time_projExtra);
          }
        }

        // ==========================
        // operation: compute and store errors 
        if (params->errType!=NO_ERR && iter%params->errSubiter==0){

          // set timer
          if (workspace->timing){  
            cudaEventRecord(start);
          }

          // ITER_DIFF case
          if (params->errType==ITER_DIFF){
            if (iter>params->errSubiter){
              // if this is not the first subiter, then compute the error
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                set_difference_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x_lasterr->descriptor->data[i],  
                                                                    (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 
                                                                    (cufftComplex*) workspace->mgpu.d_x_lasterr->descriptor->data[i], 
                                                                    1.0, 
                                                                    1.0,
                                                                    perGPUDim);
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
                getLastCudaError("ssc-cdi: error / kernel execution failed @ set_difference_mgpu<<<.>>>\n");
              }


              // compute norm of the real and imaginary parts 
              cublasHandle_t handle;
              cublasStatus_t stat;
              cublasCreate(&handle);

              float *iter_diff_norm_perGPU = (float*) malloc(sizeof(float)*workspace->gpus->ngpus);
              float iter_diff_norm = 0.0f;
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                stat = cublasScnrm2_v2(handle, 
                                       powf(workspace->dimension,3.0), 
                                       (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 
                                       1, 
                                       &iter_diff_norm_perGPU[i]);
              }
              for (int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              }

              if (stat == CUBLAS_STATUS_NOT_INITIALIZED)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2x<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              if (stat == CUBLAS_STATUS_ALLOC_FAILED)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              if (stat == CUBLAS_STATUS_EXECUTION_FAILED)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              if (stat == CUBLAS_STATUS_INVALID_VALUE)
                fprintf(stderr,"ssc-cdi: cublas error / cublasSnrm2_v2<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");

              // destroy the handle 
              cublasDestroy(handle);

              // compute the wanted norm by combining the norm computed for each GPU slice 
              for (int i=0; i<workspace->gpus->ngpus; i++){
                iter_diff_norm = iter_diff_norm + powf(iter_diff_norm_perGPU[i],2.0);
              }
              iter_diff_norm = sqrtf(iter_diff_norm);

              // debug computed norm
              fprintf(stdout, "ssc-cdi: Computed norm (iter_diff) at iteration %d = %f\n", iter, iter_diff_norm);

              // free iter_diff_norm_perGPU
              free(iter_diff_norm_perGPU);

            }else{
              // if this is the first subiter, simply store the current iteration
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                set_difference_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x_lasterr->descriptor->data[i],  
                                                                    (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 
                                                                    (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 
                                                                    1.0, 
                                                                    0.0,
                                                                    perGPUDim);
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
                getLastCudaError("ssc-cdi: error / kernel execution failed @ set_difference_mgpu<<<.>>>\n");
              }
              
            }
          }

          // stop timer 
          if (workspace->timing){
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_computeErr, start, stop);
            fprintf(stdout, "ssc-cdi: time computeErr() = %f\n", time_computeErr);
          } 
        }



        // debug iteration timers 
        if (workspace->timing){  
          total = total + (time_projM + time_projS + time_shrinkWrap + time_projExtra + time_computeErr);
          fprintf(stdout,
                  "ssc-cdi: Iteration %d takes %lf ms ** \n\n", 
                  iter, 
                  time_projM + 
                  time_projS + 
                  time_shrinkWrap + 
                  time_projExtra + 
                  time_computeErr);        
        }
      }

    if (workspace->timing){
        printf("ssc-cdi: total time with HIO iterations is %lf ms\n", total); 
    }


    // free global min/max variables
    free(maxvalue);
    free(cmaxvalue);
    free(idxMaxvalue);

    // cudaFree(maxvalue);
    // cudaFree(cmaxvalue);
    // cudaFree(idxMaxvalue);
    }
      
  } 
  
} // extern "C"

