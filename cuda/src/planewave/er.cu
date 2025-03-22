#include <iostream>
#include <math.h>

#include "pwcdi.h"
#include "compute.h"
#include "gpus.h"
#include "util.h"

extern "C"{
  void er(ssc_pwcdi_plan *workspace,
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
  * @brief Executes the Error Reduction (ER) algorithm for PWCDI.
  *
  * This function implements the ER (Error Reduction) algorithm for Plane Wave CDI. The algorithm iterates over a series of updates 
  * to the object and support, using both the  measured diffraction data and the phase retrieval constraints. The function supports 
  * both single-GPU and multi-GPU configurations, ensuring flexibility in different computational environments.
  
  * @param workspace Pointer to the `ssc_pwcdi_plan` structure containing the workspace and GPU memory allocations.
  * @param params Pointer to the `ssc_pwcdi_params` structure.

  * @note The parameter definitions can be found in the python wrapper with the same names. 
  * 
  * @pre The `workspace` and `params` structures must be properly initialized and contain valid pointers to GPU memory.
  * @post The phase retrieval process will be performed with the specified parameters, and the object and support allocated in 
  * the workspace variable will be updated accordingly.
  */

  // single gpu case
  if(workspace->gpus->ngpus==1){
		
		float total, time_projM, time_projS;
		total = 0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	
		const dim3 threadsPerBlock(tbx, tby, tbz);
		const dim3 gridBlock(ceil((workspace->dimension + threadsPerBlock.x - 1)/threadsPerBlock.x),
										  	 ceil((workspace->dimension + threadsPerBlock.y - 1)/threadsPerBlock.y),
												 ceil((workspace->dimension + threadsPerBlock.z - 1)/threadsPerBlock.z));

		// reminder: workspace->dx already contains the starting point
	
		for (int iter=0; iter<global_iteration; ++iter){
			// ===============================================
			// Operation: s_projection_M()
			
			checkCudaErrors(cudaSetDevice(workspace->gpus->gpus[0]));

			// set timer
			if (workspace->timing){
			cudaEventRecord(start);
			}
		    
        	// perform the projection onto the set M
		    s_projection_M(workspace->plan_C2C,
										   workspace->sgpu.d_y,       //to
										   workspace->sgpu.d_x,       //from
										   workspace->sgpu.d_signal,
											 params->eps_zeroamp,
										   workspace->dimension);


		    if (workspace->timing){
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&time_projM, start, stop);
				fprintf(stdout,"ssc-cdi: Projection_M() time: %lf ms\n", time_projM);
		    }
		    
		    // =====================
		    // Operation: s_projection_S_only()
        
			// set timer
			if (workspace->timing){
				cudaEventRecord(start);
			}

      s_projection_S_only(workspace->plan_C2C,
                          workspace->sgpu.d_x,
                          workspace->sgpu.d_y,
                          workspace->sgpu.d_support,
                          extra_constraint,
                          workspace->dimension);
		    
      // stop time
      if (workspace->timing){
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_projS, start, stop);
        fprintf(stdout,"ssc-cdi: s_projection_S_only(): %lf ms\n", time_projS);
      }

        // debug timers (at the end of the current ER iteration)
      if (workspace->timing){
        total = total + (time_projM + time_projS);
        fprintf(stdout,"ssc-cdi: Iteration %d takes %lf ms ** \n\n", iter, time_projM + time_projS);
      }
	

    }

      	// debug timers (at the end of ER iterations)
		if (workspace->timing){
			fprintf(stdout,"ssc-cdi: total time with ER iterations is %lf ms\n", total);	
		}





    // multi GPU case
    }else{ 
      float total, time_projM, time_projS;
      total = 0.0f;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      
      const size_t dim = workspace->nvoxels;
      const int n_gpus = workspace->gpus->ngpus;
      const size_t perGPUDim = dim/n_gpus;

      const dim3 threadsPerBlock(tbx*tby*tbz);
      const dim3 gridBlock(ceil((perGPUDim + threadsPerBlock.x - 1)/threadsPerBlock.x));
      
      // reminder: workspace->dx already contains the starting point

      for (int iter=0; iter<global_iteration; ++iter){
        // ==========================================
        // operation: s_projection_M 
        
        // set timer 
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

    
        // stop timer
        if (workspace->timing){
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_projM, start, stop);
          fprintf(stdout,"ssc-cdi: m_projection_M() time: %lf ms\n", time_projM);
        }

        // =============================== 
        // operation: m_projection_S_only

        // set timer
        if (workspace->timing){	
          cudaEventRecord(start);
        }

        // perform the m_projection_S_only
        m_projection_S_only(workspace->mgpu.d_x, //to   
                            workspace->mgpu.d_y, //from (dz) depois d_x
                            workspace->mgpu.d_support,
                            extra_constraint,
                            dim,
                            perGPUDim,
                            workspace->gpus);

        // stop timer 
        if (workspace->timing){
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time_projS, start, stop);
          fprintf(stdout,"ssc-cdi: m_projection_S_only(): %lf ms\n", time_projS);
        }
    
      
        // debug timers (at the end of the current ER iteration)
        if (workspace->timing){
          total = total + (time_projM + time_projS);
          fprintf(stdout,"ssc-cdi: Iteration %d takes %lf ms ** \n", iter, time_projM + time_projS);	
        }
      }

      // debug timers (at the end of ER iterations)
      if (workspace->timing){
        fprintf(stdout,"ssc-cdi: total time with ER iterations is %lf ms\n", total);	
      }    
    }	 
  } 
  
} // extern "C"

