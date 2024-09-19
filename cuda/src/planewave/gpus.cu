#include <stdlib.h>
#include <stdio.h>
#include "gpus.h"

extern "C" {

  void ssc_gpus_get_howMuchIsAlreadyThere(float *used,
                                          float *available,
                                          int type){
    //cudaError_t cuda_status;
    size_t free_byte;
    size_t total_byte;
    
    cudaMemGetInfo(&free_byte, &total_byte);
    
    float free_db = (double)free_byte;
    float total_db = (double)total_byte;
    float used_db = total_db - free_db;
    
    if(type==0){
      *used = used_db/(1024.0*1024.0);
      *available = free_db/(1024.0*1024.0);
    }else{
      *used = used_db/(1024.0*1024.0*1024.0);
      *available = free_db/(1024.0*1024.0*1024.0);
    }
  }

  void ssc_gpus_get_info(int nGPUs,
                        int *whichGPUs,
                        const char *state){
    int type = 1; //giga
    
    fprintf(stderr,"ssc-cdi: Memory available at selected GPUs [ %s ]\n", state);
    
    for (int i=0; i<nGPUs; i++){
      float available, used;
        
      cudaSetDevice(whichGPUs[i]);
        
      ssc_gpus_get_howMuchIsAlreadyThere(&used, &available, type);
        
      fprintf(stdout," |-> GPU[%d] - %f GiB  used / %f GiB available\n", whichGPUs[i], used, available);
    }
    
    fprintf(stdout,"\n");
  }

} 
