#include <iostream>
#include <math.h> 

#include "pwcdi.h"
#include "compute.h"
#include "gpus.h"
#include "util.h"

  
extern "C"{


  void largest(float *max, float *arr, int n){
    *max = arr[0];

    for (int i=1; i<n; i++){
      if (arr[i] > *max){
        *max = arr[i];
      }
    }
  }

  void smallest(float *min, float *arr, int n){
    *min = arr[0];

    for (int i=1; i<n; i++){
      if (arr[i] <*min){
        *min = arr[i];
      }
    }
  }
 
  void hio(ssc_pwcdi_plan *workspace,
           ssc_pwcdi_params *params,
           int global_iteration,
           int shrinkwrap_subiter,
           int initial_shrinkwrap_subiter,
           int extra_constraint,
           int extra_constraint_subiter,
           int initial_extra_constraint_subiter,
           float shrinkwrap_threshold,                       
           int shrinkwrap_iter_filter,
           int shrinkwrap_mask_multiply,
           bool shrinkwrap_fftshift_gaussian,
           float sigma, 
           float sigma_mult, 
           float beta,
           float beta_update,
           int beta_reset_subiter){

    /**
     * @brief Executes the Hybrid Input-Output (HIO) algorithm for PWCDI with appended shrink-wrap operation.
     *
     * This function implements the HIO (Hybrid Input-Output) algorithm for Plane Wave CDI. The algorithm iterates over a series of updates 
     * to the object and support, using both the  measured diffraction data and the phase retrieval constraints. 
     * The function supports both single-GPU and multi-GPU configurations, ensuring flexibility in different computational environments.
     *
     * Additionally, the function incorporates the option to apply shrink-wrap, a technique used to refine the support based on the object 
     * updates during the phase retrieval process. The shrink-wrap operation can be enabled and controlled through several parameters, 
     * allowing for fine-grained control over the refinement process.

    * @param workspace Pointer to the `ssc_pwcdi_plan` structure containing the workspace and GPU memory allocations.
    * @param params Pointer to the `ssc_pwcdi_params` structure.

    * @note The parameter definitions can be found in the python wrapper with the same names. 
    * 
    * @pre The `workspace` and `params` structures must be properly initialized and contain valid pointers to GPU memory.
    * @post The phase retrieval process will be performed with the specified parameters, and the object and support allocated in 
    * the workspace variable will be updated accordingly.
    */


    if (workspace->gpus->ngpus == 1){
      // SINGLE-GPU VERSION
    
      float total, time_projM, time_projS, time_shrinkwrap, time_projExtra, time_computeErr;
      total = 0.0f;
      time_projExtra = 0.0f;
      time_computeErr = 0.0f;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      float initial_beta = beta;
 
      const dim3 threadsPerBlock(tbx, tby, tbz);
      const dim3 gridBlock (ceil((workspace->dimension + threadsPerBlock.x - 1)/threadsPerBlock.x),
                            ceil((workspace->dimension + threadsPerBlock.y - 1)/threadsPerBlock.y),
                            ceil((workspace->dimension + threadsPerBlock.z - 1)/threadsPerBlock.z));
      

      for (int iter=0; iter<global_iteration; ++iter){
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
        // 
        set_difference<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_x,     //to
                                                       (cufftComplex*) workspace->sgpu.d_x,     //from
                                                       (cufftComplex*) workspace->sgpu.d_y,
                                                       1.0,
                                                       beta,
                                                       workspace->dimension);
        cudaDeviceSynchronize();
        getLastCudaError("ssc-cdi: error / kernel execution failed @ set_difference<<<.>>>\n");

        // operate projection_S
        if (iter%extra_constraint_subiter==0 && iter>initial_extra_constraint_subiter){
          // extra_constraint is applied inside projection_S
          s_projection_S(workspace->plan_C2C,
                         workspace->sgpu.d_x,
                         workspace->sgpu.d_y,
                         workspace->sgpu.d_support,
                         extra_constraint,
                         workspace->dimension);
        }else{
          // extra_constraint is applied separately, after the shrinkwrap
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
       
        if (iter%shrinkwrap_subiter==0 && iter>initial_shrinkwrap_subiter){
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
          if (shrinkwrap_iter_filter == FILTER_AMPLITUDE){
            // take the absolute value of the iter variable 
            absolute_value<<<gridBlock, threadsPerBlock>>>(workspace->sgpu.d_y,     
                                                           workspace->sgpu.d_x,
                                                           workspace->dimension);
            cudaDeviceSynchronize();
            getLastCudaError("ssc-cdi: error / kernel execution failed @ absolute_value<<<.>>>\n");  


            // compute fft  
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_y, // without the absolute value it is just d_x
                                         workspace->sgpu.d_y,
                                         CUFFT_FORWARD)); 

          }else if (shrinkwrap_iter_filter == FILTER_FULL){
            // compute fft of the iter variable directly
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_x, // without the absolute value it is just d_x
                                         workspace->sgpu.d_y,
                                         CUFFT_FORWARD)); 

          }

          // create gaussian function with std sigma
          if (params->swap_d_x==false){
            gaussian1<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_gaussian,
                                                      sigma,
                                                      workspace->dimension);
            cudaDeviceSynchronize();
            getLastCudaError("ssc-cdi: error / kernel execution failed @ gaussian1<<<.>>>\n");

 


            // // gaussian = FFT(gaussian)
            checkCudaErrors(cufftExecC2C(workspace->plan_C2C,
                                         workspace->sgpu.d_gaussian,
                                         workspace->sgpu.d_gaussian,
                                         CUFFT_FORWARD));
   
   
            // This performs the convolution of d_y in with the blurring kernel as a multiplication
            // in the Fourier domain. The multiplication can be performed by using both real and
            // imaginary components of the blurring mask, or only its real component. 
            if (shrinkwrap_mask_multiply == MULTIPLY_FULL){
              multiply<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                       (cufftComplex*) workspace->sgpu.d_gaussian,
                                                       (cufftComplex*) workspace->sgpu.d_y,
                                                       workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply<<<.>>>\n");
            }else if (shrinkwrap_mask_multiply == MULTIPLY_REAL){
              multiply_rc<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_gaussian,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }else if (shrinkwrap_mask_multiply == MULTIPLY_LEGACY){
              multiply_legacy<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_gaussian,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }

          // if d_x is to be swapped to the host during shrinkwrap, then create the 3D Gaussian
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
            if (shrinkwrap_mask_multiply == MULTIPLY_FULL){
              multiply<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                       (cufftComplex*) workspace->sgpu.d_x,
                                                       (cufftComplex*) workspace->sgpu.d_y,
                                                       workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply<<<.>>>\n");
            }else if (shrinkwrap_mask_multiply == MULTIPLY_REAL){
              multiply_rc<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_x,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }else if (shrinkwrap_mask_multiply == MULTIPLY_LEGACY){
              multiply_legacy<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->sgpu.d_y,
                                                         (cufftComplex*) workspace->sgpu.d_x,
                                                         (cufftComplex*) workspace->sgpu.d_y,
                                                         workspace->dimension);
              cudaDeviceSynchronize();
              getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc<<<.>>>\n");
            }

            // restore d_x from host 
            checkCudaErrors(cudaMemcpy(workspace->sgpu.d_x,             //to
                                       workspace->sgpu.d_x_swap,         //from
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
          float global_max, global_min;
          cufftComplex cglobal_max, cglobal_min;
          int idx_max, idx_min;
          cublasHandle_t handle_max, handle_min;
          cublasStatus_t stat_max, stat_min;

          cublasCreate(&handle_max);
          cublasCreate(&handle_min);
        
          stat_max = cublasIcamax(handle_max, workspace->dimension, workspace->sgpu.d_y, 1, &idx_max);
          
          if (stat_max == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
          if (stat_max == CUBLAS_STATUS_ALLOC_FAILED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
          if (stat_max == CUBLAS_STATUS_EXECUTION_FAILED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
          if (stat_max == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
          
          cudaMemcpy(&cglobal_max,
                     (cufftComplex*) workspace->sgpu.d_y + idx_max - 1,
                     sizeof(cufftComplex),
                     cudaMemcpyDeviceToHost);
          
          global_max = sqrtf(powf(fabs(cglobal_max.x),2.0) + powf(fabs(cglobal_max.y),2.0));
          cublasDestroy(handle_max);
      
          //find min value        
          stat_min = cublasIcamin(handle_min, workspace->dimension, workspace->sgpu.d_y, 1, &idx_min);
          
          if (stat_min == CUBLAS_STATUS_NOT_INITIALIZED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
          if (stat_min == CUBLAS_STATUS_ALLOC_FAILED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
          if (stat_min == CUBLAS_STATUS_EXECUTION_FAILED)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
          if (stat_min == CUBLAS_STATUS_INVALID_VALUE)
            fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMin<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
          
          cudaMemcpy(&cglobal_min,
                    (cufftComplex*) workspace->sgpu.d_y + idx_min - 1,
                    sizeof(cufftComplex),
                    cudaMemcpyDeviceToHost);
          
          global_min = sqrtf(powf(fabs(cglobal_min.x),2.0) + powf(fabs(cglobal_min.y),2.0));
          cublasDestroy(handle_min);
          
          if (workspace->timing){
            fprintf(stdout, "ssc-cdi: global_max (single gpu, cublasIcamin) = %lf \n", global_max);
            fprintf(stdout, "ssc-cdi: global_min (single gpu, cublasIcamin) = %lf \n", global_min);
          }

          // dsupp = abs( dy ) > (threshold/100) * Max ( abs(dy) )
          //
          // false: use globalMax as value for threshold
          // true: use index(globalMax) as the index for the thresholing inside kernel
          update_support_sw<<<gridBlock, threadsPerBlock>>>(workspace->sgpu.d_support,
                                                            workspace->sgpu.d_y,
                                                            shrinkwrap_threshold,
                                                            global_max,
                                                            global_min,  
                                                            idx_max,
                                                            idx_min,
                                                            workspace->dimension,
                                                            false);
          cudaDeviceSynchronize();
          getLastCudaError("ssc-cdi: error / kernel execution failed @ update_support_sw<<<.>>>\n");

          // // fftshift the support computing it    
          fftshift<<<gridBlock, threadsPerBlock>>>((void *) workspace->sgpu.d_support,
                                                   workspace->dimension,
                                                   SSC_DTYPE_BYTE);
          cudaDeviceSynchronize();
          getLastCudaError("ssc-cdi: error / Kernel execution failed @ fftshift<<<.>>>\n");


          

          //  update sigma 
          sigma = sigma_mult*sigma;
        
          // debug new values for beta and sigma  
          if (workspace->timing){
            fprintf(stdout, "ssc-cdi: new sigma = %lf \n",sigma);
            fprintf(stdout, "ssc-cdi: new beta = %lf \n",beta);
          }


          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_shrinkwrap, start, stop);
          if (workspace->timing){
            fprintf(stdout,"ssc-cdi: Shrinkwrap() time: %lf ms\n", time_shrinkwrap);   
          }
      
        }else{
          time_shrinkwrap = 0;
        }        


        // update beta
        if (beta_reset_subiter>0 && iter%beta_reset_subiter==0 && iter>0){  
          // reset beta
          beta = initial_beta;
        }else{
          // update beta
          beta = initial_beta + (1 - initial_beta)*(1 - exp( -(iter/beta_update)*(iter/beta_update)*(iter/beta_update)));
        }


      // ===============================================
      // Operation s_projection_extra_constraint


        // operation projection extra_constraint 
        if (extra_constraint != NO_EXTRA_CONSTRAINT &&
            extra_constraint_subiter>0 && 
            iter>initial_extra_constraint_subiter && iter%extra_constraint_subiter==0){

          // set timer 
          if (workspace->timing){
           cudaEventRecord(start);
          }
          
          // perform the projection 
          s_projection_extra_constraint(workspace->sgpu.d_x,
                                        workspace->sgpu.d_x,  
                                        extra_constraint,
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
        if (params->err_type!=NO_ERR && iter%params->err_subiter==0){
          // set timer 
          if (workspace->timing){
           cudaEventRecord(start);
          }

          // ITER_DIFF case 
          if (params->err_type==ITER_DIFF){
            if (iter>params->err_subiter){
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
              if (workspace->timing){
                fprintf(stdout, "ssc-cdi: Computed norm (iter_diff) at iteration %d = %f\n", iter, iter_diff_norm);
              }


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
          total = total + (time_projM + time_projS + time_shrinkwrap + time_projExtra + time_computeErr);
          fprintf(stdout,
                  "ssc-cdi: Iteration %d takes %lf ms ** \n\n", 
                  iter, 
                  time_projM + time_projS + time_shrinkwrap + time_projExtra + time_computeErr); 
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
      cudaError_t status; 


  
      float total, time_projM, time_projS, time_shrinkwrap, time_projExtra, time_computeErr;
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
    
      float initial_beta = beta;

  
  
    
      //-----------------------------------------------------------
      // start iterations
      //-----------------------------------------------------------

      // reminder: workspace->dx already contains the initial point
       

      for (int iter=0; iter<global_iteration; iter++){
        // =======================================================
        // operation m_projection_M

        // start timer
        if (workspace->timing){  
          cudaEventRecord(start);
        }

 
        m_projection_M_shuffleddata(workspace->plan_C2C,
                                    workspace->mgpu.d_y,
                                    workspace->mgpu.d_x,  
                                    workspace->mgpu.d_signal,
                                    params->eps_zeroamp,
                                    dim,
                                    perGPUDim,
                                    workspace->gpus,
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
        if (iter%extra_constraint_subiter==0 && iter>initial_extra_constraint_subiter){ 
          // extra_constraint is applied inside projection_S
          m_projection_S(workspace->mgpu.d_x, //  
                         workspace->mgpu.d_y,              
                         workspace->mgpu.d_support,
                         extra_constraint,   
                         dim,
                         perGPUDim,
                         workspace->gpus);
        }else{
          // extra_constraint is applied separately, after shrinkwrap
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
        // operation: shrinkwrap  

        // set timer 
        if (workspace->timing){  
          cudaEventRecord(start);
        }

        if (iter%shrinkwrap_subiter==0 && iter>initial_shrinkwrap_subiter){

          // store no pointer during shrinkwrap
          if (params->swap_d_x==false){

            // copy the content of d_x, which is in natural ordering format, to d_z
            // note that this copy is done directly on the content.
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              checkCudaErrors(cudaMemcpy(workspace->mgpu.d_z->descriptor->data[i],           
                                              workspace->mgpu.d_x->descriptor->data[i],
                                              perGPUDim*sizeof(cufftComplex), 
                                              cudaMemcpyDeviceToDevice)); 
            }     
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i]));
            }

            // create shifted Gaussian directly 
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              
              if (shrinkwrap_fftshift_gaussian == true){ // params->sw_fftshift_gaussian
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
            if (shrinkwrap_iter_filter==FILTER_AMPLITUDE){
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
            }else if (shrinkwrap_iter_filter==FILTER_FULL){
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
            if (shrinkwrap_mask_multiply == MULTIPLY_FULL){
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
            }else if (shrinkwrap_mask_multiply == MULTIPLY_REAL){
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
            }else if (shrinkwrap_mask_multiply==MULTIPLY_LEGACY){

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

            // make sure the subFormat is inplace
            workspace->mgpu.d_y->subFormat = CUFFT_XT_FORMAT_INPLACE;



            // find max value of abs(d_y)
            // TODO: unify both calls into a single one 
            //https://forums.developer.nvidia.com/t/how-to-get-the-max-value-in-a-vector-when-using-cublas/15360

            float global_max, global_min;
            cublasHandle_t handle_max, handle_min;
            cublasStatus_t stat_max, stat_min;
            cublasCreate(&handle_max);
            cublasCreate(&handle_min);

     
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              stat_max = cublasIcamax(handle_max, perGPUDim, (const cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], 1, &idxMaxvalue[i]);
              stat_min = cublasIcamin(handle_min, perGPUDim, (const cufftComplex*) workspace->mgpu.d_z->descriptor->data[i], 1, &idxMinvalue[i]);
     
              // debug possible errors in stat_max
              if (stat_max == CUBLAS_STATUS_NOT_INITIALIZED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (stat_max == CUBLAS_STATUS_ALLOC_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (stat_max == CUBLAS_STATUS_EXECUTION_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (stat_max == CUBLAS_STATUS_INVALID_VALUE){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     

              // debug possible errors in stat_min  
              if (stat_min == CUBLAS_STATUS_NOT_INITIALIZED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (stat_min == CUBLAS_STATUS_ALLOC_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (stat_min == CUBLAS_STATUS_EXECUTION_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (stat_min == CUBLAS_STATUS_INVALID_VALUE){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     

              cudaMemcpy(&cmaxvalue[i],
                         (cufftComplex *)workspace->mgpu.d_z->descriptor->data[i] + idxMaxvalue[i]-1,
                         sizeof(cufftComplex),
                         cudaMemcpyDeviceToHost);

              maxvalue[i] = sqrtf(powf(fabs(cmaxvalue[i].x),2.0) + powf(fabs(cmaxvalue[i].y), 2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",maxvalue[i] ); DEBUG 


              cudaMemcpy(&cminvalue[i],
                         (cufftComplex *)workspace->mgpu.d_z->descriptor->data[i] + idxMinvalue[i]-1,
                         sizeof(cufftComplex),
                         cudaMemcpyDeviceToHost);

              minvalue[i] = sqrtf(powf(fabs(cminvalue[i].x),2.0) + powf(fabs(cminvalue[i].y), 2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",minvalue[i]); DEBUG 

            }
        
            cublasDestroy(handle_max);
            largest(&global_max, maxvalue, n_gpus);
            cublasDestroy(handle_min);
            smallest(&global_min, minvalue, n_gpus);
            
            // sync all gpus
            for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
            }

            // debug 
            if (workspace->timing){
              fprintf(stdout,"ssc-cdi: global_max:= %lf\n", global_max);     
              fprintf(stdout,"ssc-cdi: global_min:= %lf\n", global_min); 
            }    


            // update support 
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              
              // dsupp = abs( dz ) > (threshold/100) * Max ( abs(dz) )
              update_support_sw_mgpu<<<gridBlock, threadsPerBlock>>>(workspace->mgpu.d_support[i],
                                                                    (cufftComplex *) workspace->mgpu.d_z->descriptor->data[i],
                                                                    shrinkwrap_threshold,
                                                                    global_max,
                                                                    global_min,
                                                                    perGPUDim);
            }
        
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              getLastCudaError("ssc-cdi: error / kernel execution failed @  update_support_sw_mgpu<<<.>>>\n");
            }
     
            
            if (shrinkwrap_fftshift_gaussian==false){  
              // fftshift the support after computing it 
              m_fftshift(workspace->mgpu.d_support, 
                         (size_t) cbrt((double)workspace->nvoxels), 
                         SSC_DTYPE_BYTE, 
                         workspace->host_swap_byte,
                         perGPUDim,
                         workspace->gpus);
            }
          


            //  update sigma 
            sigma = sigma_mult * sigma;
 
        
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
              if (shrinkwrap_fftshift_gaussian==true){  
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
            if (shrinkwrap_iter_filter == FILTER_AMPLITUDE){
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
            }else if (shrinkwrap_iter_filter==FILTER_FULL){
            }
   
            // FFT of d_y, which is holding the gaussian volume
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_y, workspace->mgpu.d_y, CUFFT_FORWARD));

   
            // ds = FFT(ds) : after FFT forward, ds is in natural order
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_x, workspace->mgpu.d_x, CUFFT_FORWARD));    
   
            // This performs the convolution of d_y in with the blurring kernel as a multiplication
            // in the Fourier domain. The multiplication can be performed by using both real and
            // imaginary components of the blurring mask, or only its real component. 
            if (shrinkwrap_mask_multiply==MULTIPLY_FULL){
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          
                multiply_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                              // (cufftComplex*) workspace->mgpu.d_gaussian[i], //from1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1 
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //from2  
                                                              perGPUDim); 
                      
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_mgpu<<<.>>>\n");
              }
            }else if(shrinkwrap_mask_multiply==MULTIPLY_REAL){
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
                multiply_rc_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                              // (cufftComplex*) workspace->mgpu.d_gaussian[i], //from1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i], // from1 
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //from2   
                                                              perGPUDim); 
                      
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc_mgpu<<<.>>>\n");
              }
            }else if(shrinkwrap_mask_multiply==MULTIPLY_LEGACY){
              for(int i = 0; i < n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
          
                multiply_legacy_mgpu<<<gridBlock, threadsPerBlock>>>((cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], //to
                                                              // (cufftComplex*) workspace->mgpu.d_gaussian[i],                //from1
                                                              (cufftComplex*) workspace->mgpu.d_y->descriptor->data[i],        // from1 
                                                              (cufftComplex*) workspace->mgpu.d_x->descriptor->data[i],        //from2  
                                                              perGPUDim); 
                      
              }
              for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
                getLastCudaError("ssc-cdi: error / kernel execution failed @ multiply_rc_mgpu<<<.>>>\n");
              }
            }
   
            // compute IFFT
            checkCudaErrors(cufftXtExecDescriptorC2C(workspace->plan_C2C, workspace->mgpu.d_x, workspace->mgpu.d_x, CUFFT_INVERSE));
         

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

            float global_max, global_min;
            cublasHandle_t handle_max, handle_min;
            cublasStatus_t stat_max, stat_min;
            cublasCreate(&handle_max);
            cublasCreate(&handle_min);

     
            for (int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              stat_max = cublasIcamax(handle_max, perGPUDim, (const cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 1, &idxMaxvalue[i]);
              stat_min = cublasIcamin(handle_min, perGPUDim, (const cufftComplex*) workspace->mgpu.d_x->descriptor->data[i], 1, &idxMinvalue[i]);
     
              // debug possible error in stat_max  
              if (stat_max == CUBLAS_STATUS_NOT_INITIALIZED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (stat_max == CUBLAS_STATUS_ALLOC_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (stat_max == CUBLAS_STATUS_EXECUTION_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (stat_max == CUBLAS_STATUS_INVALID_VALUE){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     


              if (stat_min == CUBLAS_STATUS_NOT_INITIALIZED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_NOT_INITIALIZED\n");
              }else if (stat_min == CUBLAS_STATUS_ALLOC_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_ALLOC_FAILED\n");
              }else if (stat_min == CUBLAS_STATUS_EXECUTION_FAILED){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_EXECUTION_FAILED\n");
              }else if (stat_min == CUBLAS_STATUS_INVALID_VALUE){
                fprintf(stderr,"ssc-cdi: cublas error / cublasIsaMax<<<.>>> failed: CUBLAS_STATUS_INVALID_VALUE\n");
              }     

              cudaMemcpy(&cmaxvalue[i],
                         (cufftComplex *)workspace->mgpu.d_x->descriptor->data[i] + idxMaxvalue[i]-1,
                         sizeof(cufftComplex),
                         cudaMemcpyDeviceToHost);

              maxvalue[i] = sqrtf(powf(fabs(cmaxvalue[i].x),2.0) + powf(fabs(cmaxvalue[i].y), 2.0));
              // fprintf(stdout, "ssc-cdi: (%lf) ",maxvalue[i] ); DEBUG 


              cudaMemcpy(&cminvalue[i],
                         (cufftComplex *)workspace->mgpu.d_x->descriptor->data[i] + idxMinvalue[i]-1,
                         sizeof(cufftComplex),
                         cudaMemcpyDeviceToHost);

              minvalue[i] = sqrtf(powf(fabs(cminvalue[i].x),2.0) + powf(fabs(cminvalue[i].y),2.0));

            }
        
            cublasDestroy(handle_max);
            largest (&global_max, maxvalue, n_gpus);
            cublasDestroy(handle_min);
            smallest (&global_min, minvalue, n_gpus);

            if (workspace->timing){
              fprintf(stdout,"ssc-cdi: computed global_max:= %lf\n", global_max);     
              fprintf(stdout,"ssc-cdi: computed global_min:= %lf\n", global_min); 
            }    

            // sync all gpus
            for(int i=0; i<n_gpus; i++){
                checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
                checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
            }



            // update support
            for(int i=0; i<n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i]));
              update_support_sw_mgpu<<<gridBlock, threadsPerBlock>>>(workspace->mgpu.d_support[i],
                                                                     (cufftComplex *) workspace->mgpu.d_x->descriptor->data[i],
                                                                     shrinkwrap_threshold,
                                                                     global_max,
                                                                     global_min,  
                                                                     perGPUDim);
            }
            for(int i = 0; i < n_gpus; i++){
              checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[i])); 
              checkCudaErrors(cudaStreamSynchronize(workspace->gpus->streams[i])); 
              getLastCudaError("ssc-cdi: error / kernel execution failed @ update_support_sw_mgpu<<<.>>>\n");
            }
     
            
            if (shrinkwrap_fftshift_gaussian==false){  // params->sw_fftshift_gaussian
              // fftshift the support after computing it 
              m_fftshift(workspace->mgpu.d_support, 
                         (size_t) cbrt((double)workspace->nvoxels), 
                         SSC_DTYPE_BYTE, 
                         workspace->host_swap_byte,
                         perGPUDim,
                         workspace->gpus);
            }


            // update sigma 
            sigma = sigma_mult * sigma;
            
            
            // restore d_x from host to device
            checkCudaErrors(cufftXtMemcpy(workspace->plan_C2C,
                                          workspace->mgpu.d_x,
                                          workspace->mgpu.d_x_swap,
                                          CUFFT_COPY_HOST_TO_DEVICE));
   

          } // end of the second case of nswap_vars


 
        }else{ 
          time_shrinkwrap = 0;
        } // end of shrinkwrap

        // stop timer 
        if (workspace->timing){
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_shrinkwrap, start, stop);
          fprintf(stdout, "ssc-cdi: time_shrinkwrap = %f\n", time_shrinkwrap);
        }


        // update beta
        if (beta_reset_subiter>0 && iter%beta_reset_subiter==0 && iter>0){
          // reset beta
          beta = initial_beta;
        }else{
          // update beta
          beta = initial_beta + (1 - initial_beta)*(1 - exp( -(iter/beta_update)*(iter/beta_update)*(iter/beta_update)));
        }


        // ==========================
        // operation: m_projection_extra_constraint

        // projection_extra_constraint is only performed in the following case
        if (extra_constraint != NO_EXTRA_CONSTRAINT &&
            extra_constraint_subiter>0 && 
            iter>initial_extra_constraint_subiter && iter%extra_constraint_subiter==0){

          // set timer 
          if (workspace->timing){  
            cudaEventRecord(start);
          }

          // perform the projection with extra constraint
          m_projection_extra_constraint(workspace->mgpu.d_x,
                                        workspace->mgpu.d_x,  
                                        extra_constraint, 
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
        if (params->err_type!=NO_ERR && iter%params->err_subiter==0){

          // set timer
          if (workspace->timing){  
            cudaEventRecord(start);
          }

          // ITER_DIFF case
          if (params->err_type==ITER_DIFF){
            if (iter>params->err_subiter){
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
          total = total + (time_projM + time_projS + time_shrinkwrap + time_projExtra + time_computeErr);
          fprintf(stdout,
                  "ssc-cdi: Iteration %d takes %lf ms ** \n\n", 
                  iter, 
                  time_projM + 
                  time_projS + 
                  time_shrinkwrap + 
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

