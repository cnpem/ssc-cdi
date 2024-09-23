#include "pwcdi.h"
#include "hio.h"
#include "er.h"
#include "gpus.h"
#include "pwutils.h"

#include <stdio.h>
#include <math.h>
#include <string.h>


typedef void (*FunctionCallback)(ssc_pwcdi_plan *, 
                                 ssc_pwcdi_params *,
                                 int, 
                                 int, int,
                                 int, int, int);

FunctionCallback ssc_pwcdi_run[] = {&hio,
                                    &er};

typedef void (*ssc_pwcdi_function)(ssc_pwcdi_plan *, 
                                   ssc_pwcdi_params *,
                                   int,                               
                                   int, int,
                                   int, int, int);


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
  alloc_workspace(&workspace, &params, gpu, ngpu); //, nswap_host);
  
  // Set initial guess
  set_input(&workspace, &params, data_input, obj_input);

  // Run through selected algorithms: 
  for(int k=0; k<nalgorithms; k++){
    fprintf(stdout,
      "ssc-cdi: Running \t %s: \t %d iterations ( shrink-wrap step %d, starting at iteration %d)\n",
      algorithms[k].name,
      algorithms[k].iteration,
      algorithms[k].shrinkWrap,
      algorithms[k].initialShrinkWrapSubiter);
    fprintf(stdout,
      "ssc-cdi: Extra constraint code = %d. initialExtraConstraintIteration = %d. extraConstraintIteration = %d. \n",
      algorithms[k].extraConstraint,
      algorithms[k].initialExtraConstraintSubiter,
      algorithms[k].extraConstraintSubiter);
      
      ssc_pwcdi_run[sscAlgorithm2Index(algorithms[k].name)] (&workspace, 
                                                             &params, 
                                                             algorithms[k].iteration, 
                                                             algorithms[k].shrinkWrap,
                                                             algorithms[k].initialShrinkWrapSubiter,
                                                             algorithms[k].extraConstraint,
                                                             algorithms[k].extraConstraintSubiter,
                                                             algorithms[k].initialExtraConstraintSubiter);
  }


  // copy final results to output 
  get_output(obj_output, finsup_output, &workspace, &params);

  // if the user wants to save the output in disk, this will be done in the python wrapper.
  // Otherwise, use the following code to save the output in disk.
  //char* outpath_real, // this will has to be passed by the python wrapper 
  //char* finsup_path,  // this will has to be passed by the python wrapper

  // char outpath_ampli[strlen(outpath)+7];
  // char outpath_phase[strlen(outpath)+7];
  // strcpy(outpath_ampli, outpath);
  // strcpy(outpath_phase, outpath);
  // strcat(outpath_ampli,".ampli");
  // strcat(outpath_phase,".phase");

  // // save amplitude and phase of the iteration variable
  // set_output(outpath_ampli, &params, &workspace, SSC_VARIABLE_ITER, AMPLITUDE);
  // set_output(outpath_phase, &params, &workspace, SSC_VARIABLE_ITER, PHASE);
  
  // // Get final support data (saves d_support)
  // set_output(finsup_path, &params, &workspace, SSC_VARIABLE_SUPP, AMPLITUDE);

  // Free workspace 
  free_workspace(&workspace, &params);
}
 



