#include <cmath>
#include <common/types.hpp>
#include <common/logger.hpp>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "complex.hpp"
#include "engines_common.hpp"


constexpr int max_dist = 3;
constexpr int n_pos_neighbors = max_dist * 8 + 1;

__device__ float d_pos_offx[n_pos_neighbors + 1];
__device__ float d_pos_offy[n_pos_neighbors + 1];

void init_offsets() {

    float pos_offx[n_pos_neighbors + 1];
    float pos_offy[n_pos_neighbors + 1];

    const int off_x[] = {-1, +0, +1, -1, +1, -1, +0, +1};
    const int off_y[] = {-1, -1, -1, +0, +0, +1, +1, +1};

    for (int idx = 1, dist = 1; dist<=max_dist; dist++) {
        for (int i = 0; i < 8; ++i) {
            pos_offx[idx] = off_x[i] * dist;
            pos_offy[idx] = off_y[i] * dist;
            ++idx;
        }
	}

    // Copy directly to __device__ variables
    cudaMemcpyToSymbol(d_pos_offx, pos_offx, (n_pos_neighbors + 1) * sizeof(float));
    cudaMemcpyToSymbol(d_pos_offy, pos_offy, (n_pos_neighbors + 1) * sizeof(float));
}

__global__ void KOffsetSideExitwave(GArray<complex> exitwave,
                              const GArray<complex> probe,
                              const GArray<complex> object,
                              const GArray<Position> positions,
                              float offx,
                              float offy)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= probe.shape.x)
        return;

    const int idy = blockIdx.y*blockDim.y + threadIdx.y;
    const int idz = blockIdx.z;

    const int objposx = idx + (int)positions(idz, 0, 0).x + offx;
    const int objposy = idy + (int)positions(idz, 0, 0).y + offy;

    if (objposx < 0 || objposx >= object.shape.x || objposy < 0 || objposy >= object.shape.y) {
        return;
    }

    const complex& obj = object(objposy, objposx);

    // for each incoherent mode
    for(size_t m=0; m<probe.shape.z; m++){
        exitwave(m + probe.shape.z*blockIdx.z, idy, idx) = obj * probe(m, idy, idx);
    }
}

__global__ void KComputeError(float* errors,
                              const GArray<complex> exitwave,
                              const GArray<float> diffraction_patterns,
                              const float* background,
                              size_t nummodes){

    __shared__ float shared_error_rfactor[64];
    __shared__ int shared_error_count[64];

    if(threadIdx.x < 64){
        shared_error_rfactor[threadIdx.x] = 0;
        shared_error_count[threadIdx.x] = 0;
    }
    __syncthreads();

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idy = blockIdx.y; //threadIdx.y + blockIdx.y * blockDim.y;   // <-
    size_t idz = blockIdx.z;

    if(idx >= diffraction_patterns.shape.x) {
        return;
    }

    const float diff_pattern = diffraction_patterns(idz, idy, idx);

    if(diff_pattern >= 0) {
        float wabs2 = 0.0f;

        for(int m=0; m<nummodes; m++){
            wabs2 += exitwave(idz*nummodes + m, idy, idx).abs2();
        }

        // Compute r-factor error
        const float error_rfactor = sq(sqrtf(diff_pattern) - sqrtf(wabs2));
        //const float error_rfactor = sq(diff_pattern - wabs2);

        atomicAdd(shared_error_count + threadIdx.x%64, 1);
        atomicAdd(shared_error_rfactor + threadIdx.x%64, error_rfactor);
    }
    __syncthreads();

    Reduction::KSharedReduce(shared_error_rfactor, 64);
    Reduction::KSharedReduce(shared_error_count, 64);

    if(threadIdx.x==0) {
        //const float error_count = diffraction_patterns.shape.y * diffraction_patterns.shape.x * diffraction_patterns.shape.z;
        const float error_count = shared_error_count[0];
        atomicAdd(errors + idz, shared_error_rfactor[0] / error_count);
    }
}


__global__ void KProjectPhiToProbe(const GArray<complex> probe, 
                                   complex* probe_acc, 
                                   float* probe_div,
                                   const GArray<complex> object, 
                                   const GArray<complex> exitwave, 
                                   const GArray<Position> positions,
                                   bool bFTNorm, 
                                   bool isGrad) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x){
        return;
    }

    complex pacc = complex(0);
    float pdiv = 0;

    for (size_t pos = 0; pos < positions.shape.z; pos++){
        // for each flyscan point
        for (int p = 0; p < positions.shape.x; p++){
            int objposx = idx + (int)positions(pos, 0, p).x;
            int objposy = idy + (int)positions(pos, 0, p).y;

            complex obj = object(objposy, objposx);
            complex ew = exitwave((pos * positions.shape.x + p) * probe.shape.z + blockIdx.z, idy, idx);

            pacc += ew * obj.conj();
            pdiv += obj.abs2();
        }
    }

    size_t index = blockIdx.z * probe.shape.x * probe.shape.y + idy * probe.shape.x + idx;

    if (bFTNorm) pacc /= (float)(probe.shape.x * probe.shape.y);
    if (!isGrad) pacc -= probe[index] * pdiv;

    probe_acc[index] += pacc;
    probe_div[index] += pdiv;
}

// the kernel code is replicated for complex16, for some reason cuda was not playing well with explicit instantiation on gpu kernels
__global__ void KProjectPhiToProbe(const GArray<complex> probe, 
                                   complex* probe_acc, 
                                   float* probe_div,
                                   const GArray<complex> object, 
                                   const GArray<complex16> exitwave, 
                                   const GArray<Position> positions,
                                   bool bFTNorm, bool isGrad) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x){
        return;
    }

    complex pacc = complex(0);
    float pdiv = 0;

    for (size_t pos = 0; pos < positions.shape.z; pos++){
        // for each flyscan point
        for (int p = 0; p < positions.shape.x; p++){
            int objposx = idx + (int)positions(pos, 0, p).x;
            int objposy = idy + (int)positions(pos, 0, p).y;

            complex obj = object(objposy, objposx);
            complex ew = complex(exitwave((pos * positions.shape.x + p) * probe.shape.z + blockIdx.z, idy, idx));

            pacc += ew * obj.conj();
            pdiv += obj.abs2();
        }
    }

    size_t index = blockIdx.z * probe.shape.x * probe.shape.y + idy * probe.shape.x + idx;

    if (bFTNorm) pacc /= (float)(probe.shape.x * probe.shape.y);
    if (!isGrad) pacc -= probe[index] * pdiv;

    probe_acc[index] += pacc;
    probe_div[index] += pdiv;
}

template <typename dtype>
void ProjectPhiToProbe(Ptycho& pt, int section, const MImage<dtype>& Phi, bool bNormalizeFFT, bool isGrad,
        cudaStream_t stream) {
    dim3 blk = pt.probe->ShapeBlock();
    dim3 thr = pt.probe->ShapeThread();

    for (int g = 0; g < pt.gpus.size(); g++) {
        SetDevice(pt.gpus, g);

        KProjectPhiToProbe<<<blk, thr, 0, stream>>>(
                pt.probe->arrays[g][0], pt.probe_num->Ptr(g), pt.probe_div->Ptr(g),
                pt.object->arrays[g][0], Phi.arrays[g][0], pt.positions[section]->arrays[g][0],
                bNormalizeFFT, isGrad);
    }
}

template void ProjectPhiToProbe<complex>(Ptycho& pt, int section,
        const cMImage& Phi, bool bNormalizeFFT, bool isGrad, cudaStream_t st);

template void ProjectPhiToProbe<complex16>(Ptycho& pt, int section,
        const hcMImage& Phi, bool bNormalizeFFT, bool isGrad, cudaStream_t st);

extern "C" {
    void EnablePeerToPeer(const std::vector<int>& gpus);
    void DisablePeerToPeer(const std::vector<int>& gpus);

    __global__ void KProjectReciprocalSpace(GArray<complex> exitwave,  
                                            const GArray<float> diffraction_patterns, 
                                            float* error_error_rfactor, 
                                            float* error_error_llk,
                                            float* error_error_mse,
                                            size_t upsample, 
                                            size_t nummodes,  
                                            bool isGrad) {

        __shared__ float shared_error_error_rfactor[64];
        __shared__ float shared_error_error_llk[64];
        __shared__ float shared_error_error_mse[64];

        if (threadIdx.x < 64){
            shared_error_error_rfactor[threadIdx.x] = 0;
            shared_error_error_llk[threadIdx.x] = 0;
            shared_error_error_mse[threadIdx.x] = 0;
        }

        __syncthreads();

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t idy = blockIdx.y;
        const size_t idz = blockIdx.z;

        if (idx >= diffraction_patterns.shape.x) return;

        const float diff_pattern = diffraction_patterns(idz, idy, idx);
        const float sqrt_difpad = sqrtf(diff_pattern);

        float exit_wave_factor = 1.0f;
        float exit_wave_addend = 0.0f;

        if (diff_pattern >= 0) {
            float wabs2 = 0.0f;
            float wabs = 0.0f;
            const float eps = 1e-3f;

            for (int m = 0; m < nummodes; m++)
                for (int v = 0; v < upsample; v++)
                    for (int u = 0; u < upsample; u++)
                        wabs2 += exitwave(idz * nummodes + m,  v + idy * upsample,  u + idx * upsample).abs2();

            wabs = sqrtf(wabs2) / upsample; // can we kill upsample? not sure it is necessary anymore.

            // compute rfactor error 
            atomicAdd(shared_error_error_rfactor + threadIdx.x % 64, sq(sqrt_difpad - wabs));

            // poisson negative log-likelihood 
            float llk =  + wabs2 - diff_pattern * logf(wabs2 + eps); 

            // // this constant term does not depend on wabs2, so it is not actually useful to see the
            // // general behavior of the negative log-likelihood curve. You can remove it, but then the 
            // // error curve will probably assume negative values. Also, the correct constant would be
            // // log(diff_pattern!), but the factorial is too expensive, so we approximate
            // llk = llk + diff_pattern*logf(diff_pattern + eps) 
            //           - diff_pattern 
            //           + 0.5*logf(2*3.141592*diff_pattern + eps);

            // compute negative llk error 
            atomicAdd(shared_error_error_llk + threadIdx.x % 64, llk);

            // compute mse 
            float mse = fabs(diff_pattern - wabs2)/(eps + diffraction_patterns.shape.x*diffraction_patterns.shape.y);
            atomicAdd(shared_error_error_mse + threadIdx.x % 64, mse);

            // Define ew_f and ew_a to be used in the next loop in ew = ew_f * ew + ew_a
            if (wabs > 0.0f && isGrad) {  //AP
                exit_wave_factor = (sqrt_difpad / wabs - 1); // why -1 for AP and not for RAAR?
                exit_wave_addend = 0.0f;
            }
            else if (wabs > 0.0f && !isGrad) { //RAAR
                exit_wave_factor = sqrt_difpad / wabs; // why -1 for AP and not for RAAR?
                exit_wave_addend = 0.0f;
            } else { // wabs <= 0.0f
                exit_wave_addend = sqrt_difpad;
                exit_wave_factor = 0.0f;
            }

        } else if (isGrad) { // if diff_pattern < 0 and isGrad. Make invalid points to be zero in the wavefront
            exit_wave_factor = 0.0f;
            exit_wave_addend = 0.0f;
        }

        for (int m = 0; m < nummodes; m++)
            for (int v = 0; v < upsample; v++)
                for (int u = 0; u < upsample; u++) {
                    complex ew = exitwave(idz * nummodes + m,  v + idy * upsample,   u + idx * upsample);

                    // application of the measured intensity to the exitwave (projection in reciprocal space)
                    ew = ew * exit_wave_factor + exit_wave_addend; 
                    exitwave(idz * nummodes + m,  v + idy * upsample,   u + idx * upsample) = ew;
                }

        __syncthreads();

        // reduce errors 
        Reduction::KSharedReduce(shared_error_error_rfactor, 64);
        Reduction::KSharedReduce(shared_error_error_llk, 64);
        Reduction::KSharedReduce(shared_error_error_mse, 64);

        if (threadIdx.x == 0){
            atomicAdd(error_error_rfactor + blockIdx.y, shared_error_error_rfactor[0]);
            atomicAdd(error_error_llk + blockIdx.y, shared_error_error_llk[0]);
            atomicAdd(error_error_mse + blockIdx.y, shared_error_error_mse[0]);
        }

    }
}


void ProjectReciprocalSpace(Ptycho &pt, rImage* diff_pattern, cImage* wavefront, int g, bool isGrad, cudaStream_t stream) {

    SetDevice(pt.gpus, g);

    complex* ewave = wavefront->gpuptr;

    int upsample = wavefront->sizex / diff_pattern->sizex;

    if (upsample>1){
        printf("Upsample factor >1: ", upsample);
    }

    pt.propagator[g]->Propagate(ewave, ewave, wavefront->Shape(), 1, stream);

    wavefront->FFTShift2(stream);

    KProjectReciprocalSpace<<<diff_pattern->ShapeBlock(), diff_pattern->ShapeThread(), 0, stream>>>(*wavefront, 
                                                                                                    *diff_pattern, 
                                                                                                    pt.error_rfactor->Ptr(g), 
                                                                                                    pt.error_llk->Ptr(g),
                                                                                                    pt.error_mse->Ptr(g),
                                                                                                    upsample,  
                                                                                                    pt.probe->sizez, 
                                                                                                    isGrad);

    wavefront->FFTShift2(stream);

    pt.propagator[g]->Propagate(ewave, ewave, wavefront->Shape(), -1, stream);

}


void ProjectReciprocalSpace(Ptycho &pt, rImage* diff_pattern, int g, bool isGrad, cudaStream_t stream) {

    SetDevice(pt.gpus, g);

    complex* ewave = pt.wavefront->Ptr(g);

    int upsample = pt.wavefront->sizex / diff_pattern->sizex;

    pt.propagator[g]->Propagate(ewave, ewave, pt.wavefront->Shape(), 1, stream);

    pt.wavefront->arrays[g]->FFTShift2(stream);

    KProjectReciprocalSpace<<<diff_pattern->ShapeBlock(), diff_pattern->ShapeThread(), 0, stream>>>(pt.wavefront->arrays[g][0], 
                                                                                                    *diff_pattern, 
                                                                                                    pt.error_rfactor->Ptr(g), 
                                                                                                    pt.error_llk->Ptr(g),
                                                                                                    pt.error_mse->Ptr(g),
                                                                                                    upsample, 
                                                                                                    pt.probe->sizez, 
                                                                                                    isGrad);

    pt.wavefront->arrays[g]->FFTShift2(stream); 

    pt.propagator[g]->Propagate(ewave, ewave, pt.wavefront->Shape(), -1, stream);

}

void ApplyProbeSupport(Ptycho& pt) {
    SetDevice(pt.gpus, 0);
    const dim3 shape = dim3(pt.probe->sizex, pt.probe->sizey, pt.probe->sizez);
    complex *probe_ptr = pt.probe->arrays[0]->gpuptr;
    if (pt.distance_m > 0)
        pt.probepropagator->Propagate(probe_ptr, probe_ptr, shape, +pt.distance_m);

    pt.probe->arrays[0][0] *= pt.probesupport->arrays[0][0];

    if (pt.distance_m > 0)
        pt.probepropagator->Propagate(probe_ptr, probe_ptr, shape, -pt.distance_m);

    pt.probe->BroadcastSync();
}


void ApplyProbeUpdate(Ptycho& pt, cImage& velocity, float stepsize, float momentum, float epsilon) {

    if (momentum < 0 | stepsize < 0) return;

    SetDevice(pt.gpus, 0);

    pt.probe->WeightedLerpSync(*(pt.probe_num), *(pt.probe_div), stepsize, momentum, velocity, epsilon);

    if (pt.probesupport != nullptr) {
        ApplyProbeSupport(pt);
    }
}

__global__ void KPositionCorrection(float* errorcounter, 
                                    Position* positions,
                                    const size_t batchsize,
                                    const dim3 objshape, 
                                    const dim3 probeshape) {

    const int z = blockIdx.x*blockDim.x + threadIdx.x;

    if (z >= batchsize){
        return;
    }

    float* error = errorcounter + z;
    float minerror = error[0];
    int minidx = 0;

    for(int k = 1; k <= n_pos_neighbors; k++) {
        if(error[batchsize*k]<minerror && error[batchsize*k]>0.0f) {
            minerror = error[batchsize*k];
            minidx = k;
        }
    }

    const float x = positions[z].x;
    const float y = positions[z].y;
    positions[z].x = fminf(fmaxf(x+d_pos_offx[minidx],1.1f),
            objshape.x - probeshape.x-3);
    positions[z].y = fminf(fmaxf(y+d_pos_offy[minidx],1.1f),
            objshape.y - probeshape.y-3);

    // <-
    //const float x = positions[z].x;
    //const float y = positions[z].y;
    //positions[z].x = fminf(fmaxf(x+d_pos_offx[minidx], 0.0f), (float) objshape.x-probeshape.x-1);
    //positions[z].y = fminf(fmaxf(y+d_pos_offy[minidx], 0.0f), (float) objshape.y-probeshape.y-1);

    __syncthreads();
}

void ApplyPositionCorrection(Ptycho& ptycho) {

    ptycho.errorcounter->SetGPUToZero();

    const dim3 difpadshape = ptycho.diff_pattern_shape;
    rMImage cur_difpad(difpadshape.x,
                       difpadshape.y,
                       ptycho.multibatchsize,
                       false,
                       ptycho.gpus,
                       MemoryType::EAllocGPU);

    const size_t batchsize = ptycho.positions[0]->arrays[0]->sizez;
    const size_t num_batches = PtychoNumBatches(ptycho);
    const size_t ngpus = PtychoNumGpus(ptycho);
    for(int d = 0; d < num_batches; d++) {
        const size_t difpad_batch_zsize = PtychoCurBatchZsize(ptycho, d);
        const size_t difpad_idx = d * PtychoBatchSize(ptycho);

        cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
        cur_difpad.LoadToGPU(ptycho.cpu_diff_pattern + difpad_idx * difpadshape.x * difpadshape.y);
        for(int g = 0; g < ngpus; g++) {
            for(int k = 0; k <= n_pos_neighbors; k++) {
                SetDevice(ptycho.gpus, g);
                const size_t difpadsizez = ptycho.positions[d][0][g].sizez;
                if(difpadsizez > 0) {
                    dim3 blk = ptycho.wavefront->ShapeBlock();
                    blk.z = difpadsizez;
                    dim3 thr = ptycho.wavefront->ShapeThread();

                    Image<Position>& ptr_roi = *ptycho.positions[d]->arrays[g];
                    KOffsetSideExitwave<<<blk, thr>>>(*ptycho.wavefront->arrays[g],
                                                *ptycho.probe->arrays[g],
                                                *ptycho.object->arrays[g],
                                                ptr_roi,
                                                d_pos_offx[k],
                                                d_pos_offx[k]);
                    //cudaDeviceSynchronize(); // <-

                    ptycho.propagator[g]->Propagate(ptycho.wavefront->arrays[g]->gpuptr,
                                                    ptycho.wavefront->arrays[g]->gpuptr,
                                                    ptycho.wavefront->arrays[g]->Shape(),
                                                    1);
                    // compute errors
                    KComputeError<<<blk, thr>>>(
                            ptycho.errorcounter->arrays[g]->gpuptr + batchsize*k,
                            *ptycho.wavefront->arrays[g],
                            *cur_difpad.arrays[g],
                            nullptr,
                            ptycho.probe->sizez);
                    //cudaDeviceSynchronize();    // <-
                }
            }
        }

        for(int g = 0; g < ngpus; g++){
            SetDevice(ptycho.gpus, g);
            const size_t batch_size = PtychoCurBatchGpuZsize(ptycho, d, g);
            if (batch_size > 0) {
                KPositionCorrection<<<batchsize / 256 + (batchsize % 256 > 0), 256>>>(
                    ptycho.errorcounter->arrays[g]->gpuptr,
                    ptycho.positions[d][0][g].gpuptr, batchsize,
                    ptycho.object->Shape(), ptycho.probe->Shape());
                //cudaDeviceSynchronize(); // <-
            }
        }
    }
}

void DestroyPtycho(Ptycho*& ptycho_ref) {
    Ptycho& ptycho = *ptycho_ref;
    sscDebug("Dealloc POpt.");
    if (ptycho.object_div) delete ptycho.object_div;
    ptycho.object_div = nullptr;
    if (ptycho.object_num) delete ptycho.object_num;
    ptycho.object_num = nullptr;
    if (ptycho.probe_div) delete ptycho.probe_div;
    ptycho.probe_div = nullptr;
    if (ptycho.probe_num) delete ptycho.probe_num;
    ptycho.probe_num = nullptr;

    //cudaFreeHost(ptycho.cpu_diff_pattern);
    cudaHostUnregister(ptycho.cpu_diff_pattern);

    sscDebug("Deallocating base algorithm.");
    for (int g = 0; g < ptycho.gpus.size(); g++) {
        sscDebug(format("Dealloc propagator: {}", g));
        SetDevice(ptycho.gpus, g);
        delete ptycho.propagator[g];
        ptycho.propagator[g] = nullptr;
    }

    sscDebug("Probe D2H");
    ptycho.probe->CopyTo(ptycho.cpuprobe);

    sscDebug("Object D2H");
    ptycho.object->CopyTo(ptycho.cpuobject);

    sscDebug("Positions D2H");
    int64_t offset = 0;
    for (auto& position_array : ptycho.positions){
        if (position_array){
            // copy the data to dest
            position_array->CopyTo(ptycho.cpupositions + offset);

            // calculate the total size to copy for the current MImage
            int64_t current_size = position_array->sizex*position_array->sizey*position_array->sizez;

            // update offset. multiply by 2 becase each Position is {x, y}  
            offset += current_size; 
        }
    }

    sscDebug("Dealloc probe.");
    delete ptycho.probe;
    sscDebug("Dealloc object.");
    delete ptycho.object;
    sscDebug("Dealloc exitwave.");
    delete ptycho.wavefront;

    sscDebug("Dealloc supports.");
    if (ptycho.objectsupport != nullptr) delete ptycho.objectsupport;
    if (ptycho.probesupport != nullptr) delete ptycho.probesupport;

    sscDebug("Dealloc error_errors_rfactor.");
    delete ptycho.error_rfactor;

    sscDebug("Dealloc error_errors_llk.");
    delete ptycho.error_llk;

    sscDebug("Dealloc error_errors_mse.");
    delete ptycho.error_mse;

    sscDebug("Dealloc errorcounter.");
    delete ptycho.errorcounter;

    sscDebug("Dealloc rois.");
    for (auto* pos : ptycho.positions) delete pos;

    sscDebug("Done.");

    SetDevice(ptycho.gpus, 0);
    delete ptycho.probepropagator;

    ptycho_ref = nullptr;
}

Ptycho* CreatePtycho(float* _difpads, const dim3& difshape, complex* _probe, const dim3& probeshape,
                     complex* _object, const dim3& objshape, Position* positions, int numrois, int batchsize,
                     float* _rfact, float* _llk, float* _mse, const std::vector<int>& gpus, float* _objectsupport, 
                     float* _probesupport, int numobjsupp,  float wavelength_m, float pixelsize_m, float distance_m,
                     int poscorr_iter,
                     float step_obj, float step_probe,
                     float reg_obj, float reg_probe) {

    Ptycho* ptycho = new Ptycho;
    ptycho->gpus = gpus;

    sscDebug("Initializing algorithm.");
    sscDebug("Enabling P2P");

    ptycho->pixelsize_m = pixelsize_m;
    ptycho->wavelength_m = wavelength_m;
    ptycho->distance_m = distance_m;

    EnablePeerToPeer(ptycho->gpus);

    ptycho->objreg = reg_obj;
    ptycho->probereg = reg_probe;
    ptycho->objstep = step_obj;
    ptycho->probestep = step_probe;

    ptycho->diff_pattern_shape.x = difshape.x;
    ptycho->diff_pattern_shape.y = difshape.y;
    ptycho->diff_pattern_shape.z = difshape.z;

    ptycho->poscorr_iter = poscorr_iter;

    const int ngpus = gpus.size();
    if (batchsize > 0) {
        ptycho->singlebatchsize = batchsize;
        ptycho->multibatchsize = batchsize * ngpus;

        batchsize *= ngpus;
    } else {
        ptycho->singlebatchsize = (numrois + ngpus - 1) / ngpus;
        batchsize = ptycho->multibatchsize = ptycho->singlebatchsize * ngpus;
    }
    sscDebug(format("Batches: {} {}", ptycho->singlebatchsize, ptycho->multibatchsize));

    ptycho->total_num_rois = numrois;

    ptycho->cpu_diff_pattern = _difpads;

    size_t difpad_size = ptycho->diff_pattern_shape.x * ptycho->diff_pattern_shape.y * ptycho->diff_pattern_shape.z;
    //cudaMallocHost(&(ptycho->cpu_diff_pattern), difpad_size * sizeof(float));
    //cudaMemcpy(ptycho->cpu_diff_pattern, _difpads, difpad_size * sizeof(float), cudaMemcpyHostToHost);

    ptycho->cpu_diff_pattern = _difpads;
    cudaHostRegister(ptycho->cpu_diff_pattern, difpad_size * sizeof(float), cudaHostRegisterDefault);

    ptycho->cpuprobe = _probe;
    ptycho->cpuobject = _object;
    ptycho->cpupositions = positions;
    ptycho->cpuerror_rfactor = _rfact;
    ptycho->cpuerror_llk = _llk;
    ptycho->cpuerror_mse = _mse;

    sscDebug("Alloc probe.");
    ptycho->probe = new cMImage(_probe, probeshape, true, gpus);
    sscDebug("Alloc obj");
    ptycho->object = new cMImage(_object, objshape, true, gpus);
    sscDebug("Alloc EW");
    ptycho->wavefront = new cMImage(probeshape.x, probeshape.y,
            ptycho->singlebatchsize * probeshape.z, true, gpus);

    sscDebug("Alloc Supports");

    if (numobjsupp > 0 && _objectsupport != nullptr) {
        ptycho->objectsupport = new rMImage(_objectsupport, dim3(objshape.x, objshape.y, numobjsupp), true, ptycho->gpus);
        ptycho->SupportSizes = std::vector<float>();
        for (int i = 0; i < numobjsupp; i++) {
            float s = 0;
            for (int j = 0; j < objshape.x * objshape.y; j++) s += _objectsupport[j + i * objshape.x * objshape.y];
            ptycho->SupportSizes.push_back(s);
        }
    } else
        ptycho->objectsupport = nullptr;

    if (_probesupport != nullptr)
        ptycho->probesupport = new rMImage(_probesupport, dim3(probeshape.x, probeshape.y, probeshape.z), true, ptycho->gpus);
    else
        ptycho->probesupport = nullptr;

    sscDebug("Alloc r-factor error");
    ptycho->error_rfactor = new rMImage(difshape.y, 1, 1, true, ptycho->gpus);
    ptycho->error_rfactor->SetGPUToZero();
    
    sscDebug("Alloc Poisson Log-Likelihood");
    ptycho->error_llk = new rMImage(difshape.y, 1, 1, true, ptycho->gpus);
    ptycho->error_llk->SetGPUToZero();

    sscDebug("Alloc Mean Squared Error");
    ptycho->error_mse = new rMImage(difshape.y, 1, 1, true, ptycho->gpus);
    ptycho->error_mse->SetGPUToZero();

    sscDebug("Alloc error counter for position correction");
    ptycho->errorcounter = new rMImage(n_pos_neighbors + 1, 1, batchsize, true, ptycho->gpus);
    ptycho->errorcounter->SetGPUToZero();

    printf("batchsize = %d\n", batchsize);


    SetDevice(ptycho->gpus, 0);
    ptycho->roibatch_offset = std::vector<int>();

    for (size_t n = 0; n < numrois; n += batchsize) {
        sscDebug(format("Creating DPGroup at: {} of {} at step {}" , n, numrois, batchsize));
        if (numrois - n < batchsize)  // last batch
        {
            ptycho->positions.push_back(new PositionArray(positions + n, 1, 1, numrois - n, false, gpus));
        } else {
            ptycho->positions.push_back(new PositionArray(positions + n, 1, 1, batchsize, false, gpus));
        }
        ptycho->roibatch_offset.push_back(n / ngpus);
    }

    for (int g = 0; g < gpus.size(); g++) {
        SetDevice(gpus, g);
        sscDebug(format("Creating propagator: {}", g));
        ptycho->propagator[g] = new Fraunhoffer();
    }

    sscDebug("Computing I0");
    SetDevice(gpus, 0);
    ptycho->probepropagator = new ASM(wavelength_m, pixelsize_m);

    ptycho->object_div = new rMImage(ptycho->object->Shape(), true, gpus);
    ptycho->object_num = new cMImage(ptycho->object->Shape(), true, gpus);

    ptycho->probe_div = new rMImage(ptycho->probe->Shape(), true, gpus);
    ptycho->probe_num = new cMImage(ptycho->probe->Shape(), true, gpus);


    // init shifts
    init_offsets();

    return ptycho;
}
