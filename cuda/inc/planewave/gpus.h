#ifndef GPUS_H
#define GPUS_H

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct{
    int ngpus;
    int *gpus;
    
    cudaStream_t* streams;
    
  }ssc_gpus;
  
  void ssc_gpus_get_howMuchIsAlreadyThere(float *used,
                                          float *available,
                                          int type);
  
  void ssc_gpus_get_info(int nGPUs, int *whichGPUs, const char *state);

#ifdef __cplusplus
}
#endif

#endif // ifndef GPUS_H
