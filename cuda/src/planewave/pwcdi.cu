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


void pwcdi(char *outpath,
            char *finsup_path, 
            float *input,
            int *gpu,
            int ngpu,
            int nalgorithms,
            ssc_pwcdi_params params,
            ssc_pwcdi_method *algorithms){
  
  // set flag to enable zero copy access
  cudaSetDeviceFlags(cudaDeviceMapHost);

  ssc_pwcdi_plan workspace;
 
  // Alloc workspace for FFT distribution.
  alloc_workspace(&workspace, &params, gpu, ngpu); //, nswap_host);
  
  // Set initial guess
  set_input(&workspace, &params, input);

 
  // Run through selected algorithms: 
  for( int k=0; k<nalgorithms; k++){
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

  
    
  char outpath_ampli[strlen(outpath)+7];
  char outpath_phase[strlen(outpath)+7];
  strcpy(outpath_ampli, outpath);
  strcpy(outpath_phase, outpath);
  strcat(outpath_ampli,".ampli");
  strcat(outpath_phase,".phase");

  
  // save amplitude and phase of the iteration variable
  set_output(outpath_ampli, &params, &workspace, SSC_VARIABLE_ITER, AMPLITUDE);
  set_output(outpath_phase, &params, &workspace, SSC_VARIABLE_ITER, PHASE);
  

  // Get final support data (saves d_support)
  set_output(finsup_path, &params, &workspace, SSC_VARIABLE_SUPP, AMPLITUDE);

  // Free workspace 
  free_workspace(&workspace, &params);
}

void methods(ssc_pwcdi_method *sequence, int nalgorithms){
  for (int i=0; i<nalgorithms; i++ ){
    fprintf(stdout,"--- Operator[%d] --- \n",i);
    fprintf(stdout," name: %s\n",sequence[i].name);
    fprintf(stdout," iteration:  %d\n",sequence[i].iteration);
    fprintf(stdout," shrinkWrapSubiter: %d\n",sequence[i].shrinkWrap);
    fprintf(stdout," initialShrinkWrapSubiter: %d\n",sequence[i].initialShrinkWrapSubiter);
    fprintf(stdout," extraConstraint: %d\n", sequence[i].extraConstraint);
    fprintf(stdout," extraConstraintSubiter: %d\n", sequence[i].extraConstraintSubiter); 
    fprintf(stdout," initialExtraConstraintSubiter: %d\n", sequence[i].initialExtraConstraintSubiter);
  }
}


 


extern "C" __global__ void linear_conv(cufftComplex *output_data,
                                       cufftComplex *input_data,
                                       cufftComplex *kernel,
                                       int data_width,
                                       int data_height,
                                       int data_depth,
                                       int kernel_width,
                                       int kernel_height,
                                       int kernel_depth){

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;

  if (x<data_width && y<data_height && z<data_depth) {
    int output_index = x + y*data_width + z*data_width*data_height;
    cufftComplex sum = {0.0f, 0.0f};

    // calculate the radius of the convolutional kernel
    int kx_radius = kernel_width/2;
    int ky_radius = kernel_height/2;
    int kz_radius = kernel_height/2;

    // Perform 3D convolution
    #pragma unroll
    for (int kz=-kz_radius; kz <= kz_radius; kz++){
      #pragma unroll 
      for (int ky=-ky_radius; ky<=ky_radius; ky++){
        #pragma unroll
        for (int kx=-kx_radius; kx<=kx_radius; kx++){
          const int ix = x+kx; // -kx for conv
          const int iy = y+ky; // -ky for conv
          const int iz = z+kz; // -kz for conv

          // check if indexes are within bounds 
          if (ix>=0 && ix<data_width &&
              iy>=0 && iy<data_height &&
              iz>=0 && iz<data_depth){

            // get global index
            const int input_index = ix + iy*data_width + iz*data_width*data_height;
            const int kernel_index = (kx + kx_radius) + (ky + ky_radius) * kernel_width + (kz + kz_radius) * kernel_width * kernel_height;
            cufftComplex data_value = input_data[input_index];
            cufftComplex kernel_value = kernel[kernel_index];
            sum.x += data_value.x * kernel_value.x - data_value.y * kernel_value.y;
            sum.y += data_value.x * kernel_value.y + data_value.y * kernel_value.x;
          }   
        }
      }
    }
    output_data[output_index] = sum;
  }
}


extern "C" void test_linear_conv(cufftComplex *output_data,
                                 cufftComplex *input_data,
                                 cufftComplex *kernel,
                                 int data_width,
                                 int data_height,
                                 int data_depth,
                                 int kernel_width,
                                 int kernel_height,
                                 int kernel_depth) {

  printf("enteredddd\n");

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cufftComplex *input_data_gpu;
    cufftComplex *kernel_gpu;
    cufftComplex *output_data_gpu;

    // Allocate memory on GPU for input_data, kernel, and output_data
    cudaMalloc((void **)&input_data_gpu, data_width * data_height * data_depth * sizeof(cufftComplex));
    cudaMalloc((void **)&kernel_gpu, kernel_width * kernel_height * kernel_depth * sizeof(cufftComplex));
    cudaMalloc((void **)&output_data_gpu, data_width * data_height * data_depth * sizeof(cufftComplex));

    // Copy input_data and kernel from host to GPU
    cudaMemcpy(input_data_gpu, input_data, data_width * data_height * data_depth * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_gpu, kernel, kernel_width * kernel_height * kernel_depth * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // // Define grid and block dimensions
    // dim3 blockDim(16, 16, 1); // Adjust as needed
    // dim3 gridDim((data_width + blockDim.x - 1) / blockDim.x,
    //              (data_height + blockDim.y - 1) / blockDim.y,
    //              (data_depth + blockDim.z - 1) / blockDim.z);

    const dim3 threadsPerBlock(tbx, tby, tbz); // blockDim
    const dim3 gridBlock (ceil((data_width + threadsPerBlock.x - 1)/threadsPerBlock.x),
                          ceil((data_height + threadsPerBlock.y - 1)/threadsPerBlock.y),
                          ceil((data_depth + threadsPerBlock.z - 1)/threadsPerBlock.z));
      



    cudaEventRecord(start);

    // Launch kernel
    linear_conv<<<gridBlock, threadsPerBlock>>>(output_data_gpu,
                                                input_data_gpu,
                                                kernel_gpu,
                                                data_width,
                                                data_height,
                                                data_depth,
                                                kernel_width,
                                                kernel_height,
                                                kernel_depth);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
 
    fprintf(stdout," |--> Op[2]: Convolution in spatial domain :  %lf ms\n", time);


    // Copy output_data from GPU to host
    cudaMemcpy(output_data, output_data_gpu, data_width * data_height * data_depth * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(input_data_gpu);
    cudaFree(kernel_gpu);
    cudaFree(output_data_gpu);
}





extern "C" __global__ void convolveX(cufftComplex *output,   cufftComplex *input,  cufftComplex *kernel, int width, int height, int depth, int kernelRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        cufftComplex sum = {0.0f, 0.0f};
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int sampleIdx = min(max(idx + k, 0), width - 1);
            cufftComplex data_value = input[(idz * height + idy) * width + sampleIdx];
            cufftComplex kernel_value = kernel[kernelRadius + k];
            sum.x += data_value.x * kernel_value.x - data_value.y * kernel_value.y;
            sum.y += data_value.x * kernel_value.y + data_value.y * kernel_value.x;
        }
        output[(idz * height + idy) * width + idx] = sum;
    }
}

extern "C" __global__ void convolveY(cufftComplex *output,  cufftComplex *input, cufftComplex *kernel, int width, int height, int depth, int kernelRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth) {
        cufftComplex sum = {0.0f, 0.0f};
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int sampleIdy = min(max(idy + k, 0), height - 1);
            cufftComplex data_value = input[(idz * height + sampleIdy) * width + idx];
            cufftComplex kernel_value = kernel[kernelRadius + k];
            sum.x += data_value.x * kernel_value.x - data_value.y * kernel_value.y;
            sum.y += data_value.x * kernel_value.y + data_value.y * kernel_value.x;
        }
        output[(idz * height + idy) * width + idx] = sum;
    }
}


extern "C" __global__ void convolveZ(cufftComplex *output,  
                                     cufftComplex *input,  
                                     cufftComplex *kernel, 
                                     int width, 
                                     int height, 
                                     int depth, 
                                     int kernelRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < width && idy < height && idz < depth){
        cufftComplex sum = {0.0f, 0.0f};
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int sampleIdz = min(max(idz + k, 0), depth - 1);
            cufftComplex data_value = input[(sampleIdz * height + idy) * width + idx]; 
            cufftComplex kernel_value = kernel[kernelRadius + k];
            sum.x += data_value.x * kernel_value.x - data_value.y * kernel_value.y;
            sum.y += data_value.x * kernel_value.y + data_value.y * kernel_value.x;
        }
        output[(idz * height + idy) * width + idx] = sum;
    }
}


extern "C" void separableConvolution3D(cufftComplex *output, 
                                       cufftComplex *input, 
                                       cufftComplex *kernel, 
                                       int width, 
                                       int height, 
                                       int depth, 
                                       int kernelLength){
    int kernelRadius = kernelLength / 2;
    cufftComplex *d_input, *d_output;
    cufftComplex *d_kernel;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_input, width * height * depth * sizeof(cufftComplex));
    cudaMalloc(&d_output, width * height * depth * sizeof(cufftComplex));
    cudaMalloc(&d_kernel, kernelLength * sizeof(cufftComplex));

    // Copy data to device
    cudaMemcpy(d_input, input, width * height * depth * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelLength * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    dim3 blockSize(tbx, tby, tbz); // Adjust based on your GPU's architecture
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (depth + blockSize.z - 1) / blockSize.z);

    // Start timer
    cudaEventRecord(start, 0);

    // Execute kernels
    convolveX<<<gridSize, blockSize>>>(d_output, d_input, d_kernel, width, height, depth, kernelRadius);
    cudaDeviceSynchronize();
    cudaMemcpy(d_input, d_output, width * height * depth * sizeof(cufftComplex), cudaMemcpyDeviceToDevice); // Use output as input for next step
    convolveY<<<gridSize, blockSize>>>(d_output, d_input, d_kernel, width, height, depth, kernelRadius);
    cudaDeviceSynchronize();
    cudaMemcpy(d_input, d_output, width * height * depth * sizeof(cufftComplex), cudaMemcpyDeviceToDevice); // Use output as input for next step
    convolveZ<<<gridSize, blockSize>>>(d_output, d_input, d_kernel, width, height, depth, kernelRadius);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // debug 
    fprintf(stdout," |--> Op[2]: Convolution in spatial domain (separable) :  %lf ms\n", time);

    // Copy result back to host
    cudaMemcpy(output, d_output, width*height*depth*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}






#include "pwcdi.h"
#include "fft.h"
#include "gpus.h"

extern "C" void m_test_linear_conv(cufftComplex *output, 
                                   cufftComplex *input, 
                                   cufftComplex *kernel, 
                                   int width, 
                                   int height, 
                                   int depth, 
                                   int kernelLength,
                                   int nGPUs,
                                   int* gpuIndexes){// }
                                   //ssc_gpus *gpus){

    
    // printf("a = %d\n b = %d\n, c = %d\n", gpuIndexes[0], gpuIndexes[1], nGPUs);
    ssc_gpus* gpus = (ssc_gpus*) malloc(sizeof(ssc_gpus));
    gpus->gpus  = (int*) malloc(sizeof(int)*nGPUs);
    gpus->ngpus = nGPUs;
    for(int k=0; k<nGPUs; k++){
      gpus->gpus[k] = gpuIndexes[k];
    }
    // create cuda streams
    gpus->streams = new cudaStream_t[nGPUs];
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamCreateWithFlags(&(gpus->streams[i]), cudaStreamNonBlocking));
    }



    int perGPUDim = width*height*depth/nGPUs;
    printf("fasd %d\n abb--> %d\n", nGPUs,perGPUDim);
    int kernelRadius = kernelLength/2;
    cufftComplex **d_input, **d_output;
    cufftComplex *d_kernel; 

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(tbx, tby, tbz); // Adjust based on your GPU's architecture
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (depth + blockSize.z - 1) / blockSize.z);


    // Allocate and copy input on device memory
    d_input = (cufftComplex**) calloc(nGPUs, sizeof(cufftComplex*));
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaMalloc((void**)&d_input[i], perGPUDim*sizeof(cufftComplex)));
    }
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaMemcpy((void*) d_input[i],(void*) &(input[i*perGPUDim]), perGPUDim*sizeof(cufftComplex), cudaMemcpyHostToDevice));
    }
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }

    // Allocate and copy output on device memory
    d_output = (cufftComplex**) calloc(nGPUs, sizeof(cufftComplex*));
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaMalloc((void**)&d_output[i], perGPUDim*sizeof(cufftComplex)));
    }
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }
    
    // Allocate and copy kernel on device memory
    checkCudaErrors(cudaMalloc(&d_kernel, kernelLength*sizeof(cufftComplex)));
    checkCudaErrors(cudaMemcpy((void*) d_kernel, (void*) kernel, kernelLength*sizeof(cufftComplex), cudaMemcpyHostToDevice));


    // Start timer
    cudaEventRecord(start, 0);

    // Run convolution in the X direction 
    for (int i=0; i<nGPUs; i++){
      // Set the active device
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // convolve in the X direction on each GPU
      convolveX<<<gridSize, blockSize>>>((cufftComplex*) d_output[i],  
                                         (cufftComplex*) d_input[i],  
                                         kernel, 
                                         (int) width/nGPUs, 
                                         height, 
                                         depth, 
                                         kernelRadius);
    }
    // synchronize all streams
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }

    // copy output to input 
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaMemcpy(d_input[i], d_output[i], perGPUDim*sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
    }
    for (int i=0; i<nGPUs; ++i){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }



    // Run convolution in the Y direction
    for (int i=0; i<nGPUs; i++){
      // Set the active device
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // convolve in the Y direction on each GPU
      convolveY<<<gridSize, blockSize>>>((cufftComplex*) d_output[i],  
                                         (cufftComplex*) d_input[i],  
                                         kernel, 
                                         (int) width/nGPUs, 
                                         height, 
                                         depth, 
                                         kernelRadius);
    }
    // synchronize all streams
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }

    // copy output to input 
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaMemcpy(d_input[i], d_output[i], perGPUDim*sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
    }
    for (int i=0; i<nGPUs; ++i){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }



    // Run convolution in the Z direction
    for (int i=0; i<nGPUs; i++){
      // Set the active device
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));

      // convolve in the Z direction on each GPU
      convolveZ<<<gridSize, blockSize>>>((cufftComplex*) d_output[i],  
                                         (cufftComplex*) d_input[i],  
                                         kernel, 
                                         (int) width/nGPUs, 
                                         height, 
                                         depth, 
                                         kernelRadius);
    }
    // synchronize all streams
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }

    // time the results
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // debug
    fprintf(stdout," |--> Op[2]: Convolution in spatial domain (separable, mgpu) :  %lf ms\n", time);


    // copy output to host
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaMemcpy((void*) &output[i*perGPUDim],(void*) d_output[i], perGPUDim*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    }
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i])); 
      checkCudaErrors(cudaStreamSynchronize(gpus->streams[i]));
    }

    // free device memory 
    for (int i=0; i<nGPUs; i++){
      checkCudaErrors(cudaSetDevice(gpus->gpus[i]));
      checkCudaErrors(cudaFree(d_input[i]));
      checkCudaErrors(cudaFree(d_output[i]));
    }
    cudaFree(d_input);
    cudaFree(d_output);

    cudaDeviceReset();

}

