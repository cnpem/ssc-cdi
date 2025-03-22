#include "pwcdi.h"
#include "hio.h"
#include "er.h"
#include "gpus.h"
#include "util.h"

#include <stdio.h>
#include <math.h>
#include <string.h>


typedef void (*FunctionCallback)(ssc_pwcdi_plan *, 
                                 ssc_pwcdi_params *,
                                 int,                   // iteration
                                 int,                   // shrinkwrap_subiter
                                 int,                   // initial_shrinkwrap_subiter
                                 int,                   // extra_constraint
                                 int,                   // extra_constraint_subiter
                                 int,                   // initial_extra_constraint_subiter
                                 float,                 // shrinkwrap_threshold
                                 int,                   // shrinkwrap_iter_filter
                                 int,                   // shrinkwrap_mask_multiply
                                 bool,                  // shrinkwrap_fftshift_gaussian
                                 float,                 // sigma
                                 float,                 // sigma_mult
                                 float,                 // beta 
                                 float,                 // beta_update
                                 int);                  // beta_reset_subiter

FunctionCallback ssc_pwcdi_run[] = {&hio,
                                    &er};

typedef void (*ssc_pwcdi_function)(ssc_pwcdi_plan *, 
                                   ssc_pwcdi_params *,
                                   int,                  // iteration             
                                   int,                  // shrinkwrap_subiter
                                   int,                  // initial_shrinkwrap_subiter
                                   int,                  // extra_constraint
                                   int,                  // extra_constraint_subiter
                                   int,                  // initial_extra_constraint_subiter
                                   float,                // shrinkwrap_threshold
                                   int,                  // shrinkwrap_iter_filter
                                   int,                  // shrinkwrap_mask_multiply
                                   bool,                 // shrinkwrap_fftshift_gaussian
                                   float,                // sigma
                                   float,                // sigma_mult
                                   float,                // beta
                                   float,                // beta_update
                                   int);                 // beta_reset_subiter

int sscAlgorithm2Index(char *name){
  int idx;
  bool defined = false;
  
  if(strcmp(name, "HIO")==0){
    idx     = 0;
    defined = true; 
  }

  if(strcmp(name, "ER")==0){
    idx     = 1;
    defined = true;
  }

  if (!defined){
      fprintf(stderr,"ssc-cdi: error / wrong algorithm name\n"); 
      exit(EXIT_FAILURE);
  }
  
  return idx;
}


void pwcdi(cufftComplex* obj_output,  
           uint8_t* finsup_output,     
           float* data_input,
           cufftComplex* obj_input,
           int* gpu,
           int ngpu,
           int nalgorithms,
           ssc_pwcdi_params params,
           ssc_pwcdi_method* algorithms){
  
  // set flag to enable zero copy access
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // create workspace
  ssc_pwcdi_plan workspace;
 
  // Alloc workspace for FFT distribution.
  alloc_workspace(&workspace, &params, gpu, ngpu);  
  
  // Set initial guess
  set_input(&workspace, &params, data_input, obj_input);

  // Run through selected algorithms: 
  for(int k=0; k<nalgorithms; k++){

    
    fprintf(stdout, "ssc-cdi: Algorithm Name = %s\n", algorithms[k].name);

    fprintf(stdout, "ssc-cdi: Iteration = %d\n", algorithms[k].iteration);
    fprintf(stdout, "ssc-cdi: Shrink-wrap sub-iteration = %d\n", algorithms[k].shrinkwrap_subiter);
    fprintf(stdout, "ssc-cdi: Initial shrink-wrap sub-iteration = %d\n", algorithms[k].initial_shrinkwrap_subiter);

    fprintf(stdout, "ssc-cdi: Extra constraint code = %d\n", algorithms[k].extra_constraint);
    fprintf(stdout, "ssc-cdi: Initial extra constraint sub-iteration = %d\n", algorithms[k].initial_extra_constraint_subiter);
    fprintf(stdout, "ssc-cdi: Extra constraint sub-iteration = %d\n", algorithms[k].extra_constraint_subiter);

    fprintf(stdout, "ssc-cdi: Shrink-wrap threshold = %.6f\n", algorithms[k].shrinkwrap_threshold);
    fprintf(stdout, "ssc-cdi: Shrink-wrap iteration filter = %d\n", algorithms[k].shrinkwrap_iter_filter);
    fprintf(stdout, "ssc-cdi: Shrink-wrap mask multiply = %d\n", algorithms[k].shrinkwrap_mask_multiply);
    fprintf(stdout, "ssc-cdi: Shrink-wrap FFT shift Gaussian = %s\n", algorithms[k].shrinkwrap_fftshift_gaussian ? "true" : "false");

    fprintf(stdout, "ssc-cdi: Sigma = %.6f\n", algorithms[k].sigma);
    fprintf(stdout, "ssc-cdi: Sigma multiplier = %.6f\n", algorithms[k].sigma_mult);

    fprintf(stdout, "ssc-cdi: Beta = %.6f\n", algorithms[k].beta);
    fprintf(stdout, "ssc-cdi: Beta update = %.6f\n", algorithms[k].beta_update);
    fprintf(stdout, "ssc-cdi: Beta reset sub-iteration = %d\n", algorithms[k].beta_reset_subiter);
    
    fflush(stdout);
        
    ssc_pwcdi_run[sscAlgorithm2Index(algorithms[k].name)] (&workspace, 
                                                            &params, 
                                                            algorithms[k].iteration, 
                                                            algorithms[k].shrinkwrap_subiter,
                                                            algorithms[k].initial_shrinkwrap_subiter,
                                                            algorithms[k].extra_constraint,
                                                            algorithms[k].extra_constraint_subiter,
                                                            algorithms[k].initial_extra_constraint_subiter,
                                                            algorithms[k].shrinkwrap_threshold,                
                                                            algorithms[k].shrinkwrap_iter_filter,
                                                            algorithms[k].shrinkwrap_mask_multiply,
                                                            algorithms[k].shrinkwrap_fftshift_gaussian,
                                                            algorithms[k].sigma,
                                                            algorithms[k].sigma_mult,
                                                            algorithms[k].beta,
                                                            algorithms[k].beta_update,
                                                            algorithms[k].beta_reset_subiter);
  }


  // copy final results to output 
  get_output(obj_output, finsup_output, &workspace, &params);

  // Free workspace 
  free_workspace(&workspace, &params);
}
 



