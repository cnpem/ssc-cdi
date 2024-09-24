#include <cmath>
#include <common/types.hpp>
#include <common/logger.hpp>
#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "complex.hpp"
#include "ptycho.hpp"

//host position correction offsets
float const pos_offx[] = { 0, 1, -1, 0, 0, -1, -1, 1, 1};
float const pos_offy[] = { 0, 0, 0, 1, -1, -1, 1, -1, 1};
//device position correction offsets, should be the same as the host
__device__ float const d_pos_offx[] = { 0, 1, -1, 0, 0, -1, -1, 1, 1};
__device__ float const d_pos_offy[] = { 0, 0, 0, 1, -1, -1, 1, -1, 1};

// 4-neighborhood, 8-neighborhood
constexpr int n_pos_neighbors = 8;


__global__ void KSideExitwave(GArray<complex> exitwave, const GArray<complex> probe, const GArray<complex> object, const GArray<Position> rois, int offx, int offy)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= probe.shape.x)
        return;

    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    if(true)
    {
        int objposx = idx + (int)rois(idz,0,0).x + offx;
        int objposy = idy + (int)rois(idz,0,0).y + offy;

        const complex& obj = object(objposy, objposx);

        for(size_t m=0; m<probe.shape.z; m++) // for each incoherent mode
            exitwave(m + probe.shape.z*blockIdx.z,idy,idx) = obj * probe(m,idy,idx);
    }
}
__global__ void KComputeError(float* rfactors, const GArray<complex> exitwave, const GArray<float> difpads,
        const float* background, size_t nummodes)
{
    __shared__ float sh_rfactor[64];

    if(threadIdx.x<64)
        sh_rfactor[threadIdx.x] = 0;

    __syncthreads();

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idy = blockIdx.y;

    if(idx >= difpads.shape.x)
        return;

    bool bApplyBkg = background != nullptr;

    float difpad = difpads(blockIdx.z, idy, idx);

    if(difpad >= 0)
    {
        float wabs2 = 0.0f;
        if( bApplyBkg ) wabs2 = sq( background[idy*difpads.shape.x+idx] );

        for(int m=0; m<nummodes; m++)
            wabs2 += exitwave(blockIdx.z*nummodes + m, idy, idx).abs2();

        const int sigmask = (difpad < 0);
        atomicAdd(sh_rfactor + threadIdx.x%64, sigmask * sq(sqrtf(difpad)-sqrtf(wabs2)));
    }

    __syncthreads();

    Reduction::KSharedReduce(sh_rfactor,64);
    if(threadIdx.x==0)
        atomicAdd(rfactors + blockIdx.z, sh_rfactor[0]);
}

__global__ void KProjectPhiToProbe(const GArray<complex> probe, complex* probe_acc, float* probe_div,
        const GArray<complex> object, const GArray<complex> exitwave, const GArray<Position> rois,
        bool bFTNorm, bool bIsGrad) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x) return;

    complex pacc = complex(0);
    float pdiv = 0;

    for (size_t roi = 0; roi < rois.shape.z; roi++)
        for (int p = 0; p < rois.shape.x; p++)  // for each flyscan point
        {
            int objposx = idx + (int)rois(roi, 0, p).x;
            int objposy = idy + (int)rois(roi, 0, p).y;

            complex obj = object(objposy, objposx);
            complex ew = exitwave((roi * rois.shape.x + p) * probe.shape.z + blockIdx.z, idy, idx);

            pacc += ew * obj.conj();
            pdiv += obj.abs2();
        }

    size_t index = blockIdx.z * probe.shape.x * probe.shape.y + idy * probe.shape.x + idx;

    if (bFTNorm) pacc /= (float)(probe.shape.x * probe.shape.y);
    if (!bIsGrad) pacc -= probe[index] * pdiv;

    probe_acc[index] += pacc;
    probe_div[index] += pdiv;
}

// the kernel code is replicated for complex16, for some reason cuda was not playing well with explicit instantiation on gpu kernels
__global__ void KProjectPhiToProbe(const GArray<complex> probe, complex* probe_acc, float* probe_div,
        const GArray<complex> object, const GArray<complex16> exitwave, const GArray<Position> rois,
        bool bFTNorm, bool bIsGrad) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x) return;

    complex pacc = complex(0);
    float pdiv = 0;

    for (size_t roi = 0; roi < rois.shape.z; roi++)
        for (int p = 0; p < rois.shape.x; p++)  // for each flyscan point
        {
            int objposx = idx + (int)rois(roi, 0, p).x;
            int objposy = idy + (int)rois(roi, 0, p).y;

            complex obj = object(objposy, objposx);
            complex ew = complex(exitwave((roi * rois.shape.x + p) * probe.shape.z + blockIdx.z, idy, idx));

            pacc += ew * obj.conj();
            pdiv += obj.abs2();
        }

    size_t index = blockIdx.z * probe.shape.x * probe.shape.y + idy * probe.shape.x + idx;

    if (bFTNorm) pacc /= (float)(probe.shape.x * probe.shape.y);
    if (!bIsGrad) pacc -= probe[index] * pdiv;

    probe_acc[index] += pacc;
    probe_div[index] += pdiv;
}

template <typename dtype>
void ProjectPhiToProbe(Ptycho& pt, int section, const MImage<dtype>& Phi, bool bNormalizeFFT, bool bIsGradPm) {
    dim3 blk = pt.probe->ShapeBlock();
    dim3 thr = pt.probe->ShapeThread();

    for (int g = 0; g < pt.gpus.size(); g++) {
        SetDevice(pt.gpus, g);

        KProjectPhiToProbe<<<blk, thr>>>(pt.probe->arrays[g][0], pt.probe_acc->Ptr(g), pt.probe_div->Ptr(g),
                pt.object->arrays[g][0], Phi.arrays[g][0], pt.positions[section]->arrays[g][0], bNormalizeFFT, bIsGradPm);
    }
}

template void ProjectPhiToProbe<complex>(Ptycho& pt, int section, const cMImage& Phi, bool bNormalizeFFT, bool bIsGradPm);

template void ProjectPhiToProbe<complex16>(Ptycho& pt, int section, const hcMImage& Phi, bool bNormalizeFFT, bool bIsGradPm);

extern "C" {
    void EnablePeerToPeer(const std::vector<int>& gpus);
    void DisablePeerToPeer(const std::vector<int>& gpus);


    __global__ void k_project_reciprocal_space(GArray<complex> exitwave,
            const GArray<float> difpads, float* rfactors,
            size_t upsample, size_t nummodes,
            bool bIsGrad) {
        __shared__ float sh_rfactor[64];

        if (threadIdx.x < 64) sh_rfactor[threadIdx.x] = 0;

        __syncthreads();


        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t idy = blockIdx.y;
        const size_t idz = blockIdx.z;

        if (idx >= difpads.shape.x) return;

        const float difpad = difpads(idz, idy, idx);
        const float sqrt_difpad = sqrtf(difpad);

        float exit_wave_factor = 1.0f;
        float exit_wave_addend = 0.0f;
        if (difpad >= 0) {
            float wabs2 = 0.0f;

            for (int m = 0; m < nummodes; m++)
                for (int v = 0; v < upsample; v++)
                    for (int u = 0; u < upsample; u++)
                        wabs2 += exitwave(idz * nummodes + m,
                                v + idy * upsample,
                                u + idx * upsample).abs2();

            wabs2 = sqrtf(wabs2) / upsample;

            const float hexptaulambda2 = 1.0;

            atomicAdd(sh_rfactor + threadIdx.x % 64, sq(sqrt_difpad - wabs2 * hexptaulambda2));

            if (wabs2 > 0.0f) {
                exit_wave_factor = (sqrt_difpad / wabs2 - 1);
                if (!bIsGrad) exit_wave_factor += 1.0f;
                exit_wave_addend = 0.0f;
            } else {
                exit_wave_addend = sqrt_difpad;
                exit_wave_factor = 0.0f;
            }
        } else if (bIsGrad) {
            exit_wave_factor = 0.0f;
            exit_wave_addend = 0.0f;
        }

        for (int m = 0; m < nummodes; m++)
            for (int v = 0; v < upsample; v++)
                for (int u = 0; u < upsample; u++) {
                    complex ew = exitwave(idz * nummodes + m,
                            v + idy * upsample,
                            u + idx * upsample);

                    ew = ew * exit_wave_factor + exit_wave_addend; //possibly has to deal with nan or inf?
                    exitwave(idz * nummodes + m,
                            v + idy * upsample,
                            u + idx * upsample) = ew;
                }

        __syncthreads();

        Reduction::KSharedReduce(sh_rfactor, 64);
        if (threadIdx.x == 0) atomicAdd(rfactors + blockIdx.y, sh_rfactor[0]);
    }
}


void project_reciprocal_space(Ptycho &pt, rImage* difpad, int g, bool bIsGradPm) {

    SetDevice(pt.gpus, g);

    complex* ewave = pt.exitwave->Ptr(g);

    int upsample = pt.exitwave->sizex / difpad->sizex;

    pt.propagator[g]->Propagate(ewave, ewave, pt.exitwave->Shape(), 1);

    pt.exitwave->arrays[g]->FFTShift2();

    k_project_reciprocal_space<<<difpad->ShapeBlock(), difpad->ShapeThread()>>>(pt.exitwave->arrays[g][0], *difpad, pt.rfactors->Ptr(g), upsample,
            pt.probe->sizez, bIsGradPm);


    pt.exitwave->arrays[g]->FFTShift2();
    pt.propagator[g]->Propagate(ewave, ewave, pt.exitwave->Shape(), -1);

}


void ApplyProbeUpdate(Ptycho& pt, cImage& velocity, float stepsize, float momentum, float epsilon) {

    if (momentum < 0 | stepsize < 0) return;



    SetDevice(pt.gpus, 0);


    pt.probe->WeightedLerpSync(*(pt.probe_acc), *(pt.probe_div), stepsize, momentum, velocity, epsilon);

    if (pt.probesupport != nullptr) {
        dim3 shape = dim3(pt.probe->sizex, pt.probe->sizey, pt.probe->sizez);
        complex* pointer = pt.probe->arrays[0]->gpuptr;
        SetDevice(pt.gpus, 0);

        if (pt.probef1 != 0) pt.probepropagator->Propagate(pointer, pointer, shape, +pt.probef1);

        pt.probe->arrays[0][0] *= pt.probesupport->arrays[0][0];

        if (pt.probef1 != 0) pt.probepropagator->Propagate(pointer, pointer, shape, -pt.probef1);

        pt.probe->BroadcastSync();
    }
}

__global__
void KPositionCorrection(float* errorcounter, Position* positions,
        const size_t batchsize,
        const dim3 objshape, const dim3 probeshape) {

    const int z = blockIdx.x*blockDim.x + threadIdx.x;

    if (z >= batchsize)
        return;

    float* error = errorcounter + z;
    float minerror = error[0];
    int minidx = 0;

    for(int k = 1; k <= n_pos_neighbors; k++) {
        if(error[batchsize*k] < minerror) {
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

}

void ApplyPositionCorrection(Ptycho& ptycho) {

    ptycho.errorcounter->SetGPUToZero();

    const dim3 difpadshape = ptycho.difpadshape;
    rMImage cur_difpad(difpadshape.x, difpadshape.y, ptycho.multibatchsize,
            false, ptycho.gpus, MemoryType::EAllocGPU);

    const size_t batchsize = ptycho.positions[0]->arrays[0]->sizez;
    const size_t num_batches = ptycho_num_batches(ptycho);
    const size_t ngpus = ptycho_num_gpus(ptycho);
    for(int d = 0; d < num_batches; d++) {
        const size_t difpad_batch_zsize = ptycho_cur_batch_zsize(ptycho, d);
        const size_t difpad_idx = d * ptycho_batch_size(ptycho);

        cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
        cur_difpad.LoadToGPU(ptycho.cpudifpads + difpad_idx * difpadshape.x * difpadshape.y);
        for(int g = 0; g < ngpus; g++) {
            for(int k = 0; k <= n_pos_neighbors; k++) {
                SetDevice(ptycho.gpus, g);
                const size_t difpadsizez = ptycho.positions[d][0][g].sizez;
                if(difpadsizez > 0) {
                    dim3 blk = ptycho.exitwave->ShapeBlock();
                    blk.z = difpadsizez;
                    dim3 thr = ptycho.exitwave->ShapeThread();

                    Image<Position>& ptr_roi = *ptycho.positions[d]->arrays[g];
                    KSideExitwave<<<blk,thr>>>(*ptycho.exitwave->arrays[g],
                            *ptycho.probe->arrays[g],
                            *ptycho.object->arrays[g],
                            ptr_roi, pos_offx[k], pos_offy[k]);
                    ptycho.propagator[g]->Propagate(ptycho.exitwave->arrays[g]->gpuptr,
                            ptycho.exitwave->arrays[g]->gpuptr,
                            ptycho.exitwave->arrays[g]->Shape(), 1);

                    KComputeError<<<blk,thr>>>(
                            ptycho.errorcounter->arrays[g]->gpuptr + batchsize*k,
                            *ptycho.exitwave->arrays[g], *cur_difpad.arrays[g],
                            nullptr,
                            ptycho.probe->sizez);
                }
            }
        }

        for(int g = 0; g < ngpus; g++) {
            SetDevice(ptycho.gpus, g);
            const size_t batch_size = ptycho_cur_batch_gpu_zsize(ptycho, d, g);
            if (batch_size > 0) {
                KPositionCorrection<<<256, batchsize / 256 + (batchsize % 256 > 0)>>>
                    (ptycho.errorcounter->arrays[g]->gpuptr,
                     ptycho.positions[d][0][g].gpuptr, batchsize,
                     ptycho.object->Shape(), ptycho.probe->Shape());
            }
        }
    }
}

void DestroyPtycho(Ptycho*& ptycho_ref) {
    Ptycho& ptycho = *ptycho_ref;
    ssc_debug("Dealloc POpt.");
    if (ptycho.object_div) delete ptycho.object_div;
    ptycho.object_div = nullptr;
    if (ptycho.object_acc) delete ptycho.object_acc;
    ptycho.object_acc = nullptr;
    if (ptycho.probe_div) delete ptycho.probe_div;
    ptycho.probe_div = nullptr;
    if (ptycho.probe_acc) delete ptycho.probe_acc;
    ptycho.probe_acc = nullptr;

    cudaFreeHost(ptycho.cpudifpads);

    ssc_debug("Deallocating base algorithm.");
    for (int g = 0; g < ptycho.gpus.size(); g++) {
        ssc_debug(format("Dealloc propagator: {}", g));
        SetDevice(ptycho.gpus, g);
        delete ptycho.propagator[g];
        ptycho.propagator[g] = nullptr;
    }

    ssc_debug("Probe D2H");
    ptycho.probe->CopyTo(ptycho.cpuprobe);
    ssc_debug("Object D2H");
    ptycho.object->CopyTo(ptycho.cpuobject);

    ssc_debug("Dealloc probe.");
    delete ptycho.probe;
    ssc_debug("Dealloc object.");
    delete ptycho.object;
    ssc_debug("Dealloc exitwave.");
    delete ptycho.exitwave;

    ssc_debug("Dealloc supports.");
    if (ptycho.objectsupport != nullptr) delete ptycho.objectsupport;
    if (ptycho.probesupport != nullptr) delete ptycho.probesupport;

    ssc_debug("Dealloc rfactors.");
    delete ptycho.rfactors;

    ssc_debug("Dealloc errorcounter.");
    delete ptycho.errorcounter;

    ssc_debug("Dealloc rois.");
    for (auto* roi : ptycho.positions) delete roi;

    ssc_debug("Done.");

    SetDevice(ptycho.gpus, 0);
    delete ptycho.probepropagator;

    ptycho_ref = nullptr;
}

Ptycho* CreatePtycho(float* _difpads, const dim3& difshape, complex* _probe, const dim3& probeshape,
        complex* _object, const dim3& objshape, Position* _rois, int numrois, int batchsize,
        float* _rfact, const std::vector<int>& gpus, float* _objectsupport, float* _probesupport,
        int numobjsupp,
        float wavelength_m, float pixelsize_m, float distance_m,
        int poscorr_iter,
        float step_obj, float step_probe,
        float reg_obj, float reg_probe) {

    Ptycho* ptycho = new Ptycho;
    ptycho->gpus = gpus;

    ssc_debug("Initializing algorithm.");
    ssc_debug("Enabling P2P");

    ptycho->pixelsize_m = pixelsize_m;
    ptycho->wavelength_m = wavelength_m;

    EnablePeerToPeer(ptycho->gpus);

    ptycho->objreg = reg_obj;
    ptycho->probereg = reg_probe;
    ptycho->objstep = step_obj;
    ptycho->probestep = step_probe;

    ptycho->difpadshape.x = difshape.x;
    ptycho->difpadshape.y = difshape.y;
    ptycho->difpadshape.z = difshape.z;

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
    ssc_debug(format("Batches: {} {}", ptycho->singlebatchsize, ptycho->multibatchsize));

    ptycho->total_num_rois = numrois;

    ptycho->cpudifpads = _difpads;

    size_t difpad_size = ptycho->difpadshape.x * ptycho->difpadshape.y * ptycho->difpadshape.z;
    cudaMallocHost(&(ptycho->cpudifpads), difpad_size * sizeof(float));
    cudaMemcpy(ptycho->cpudifpads, _difpads, difpad_size * sizeof(float), cudaMemcpyHostToHost);

    ptycho->cpuprobe = _probe;
    ptycho->cpuobject = _object;
    ptycho->cpurois = _rois;
    ptycho->cpurfact = _rfact;

    ssc_debug("Alloc probe.");
    ptycho->probe = new cMImage(_probe, probeshape, true, gpus);
    ssc_debug("Alloc obj");
    ptycho->object = new cMImage(_object, objshape, true, gpus);
    ssc_debug("Alloc EW");
    ptycho->exitwave = new cMImage(probeshape.x, probeshape.y,
            ptycho->singlebatchsize * probeshape.z, true, gpus);

    ssc_debug("Alloc Supports");

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

    ssc_debug("Alloc RF");
    ptycho->rfactors = new rMImage(difshape.y, 1, 1, true, ptycho->gpus);
    ptycho->rfactors->SetGPUToZero();

    ptycho->errorcounter = new rMImage(n_pos_neighbors + 1, 1, batchsize, true, ptycho->gpus);

    SetDevice(ptycho->gpus, 0);
    ptycho->roibatch_offset = std::vector<int>();

    for (size_t n = 0; n < numrois; n += batchsize) {
        ssc_debug(format("Creating DPGroup at: {} of {} at step {}" , n, numrois, batchsize));
        if (numrois - n < batchsize)  // last batch
        {
            ptycho->positions.push_back(new PositionArray(_rois + n, 1, 1, numrois - n, false, gpus));
        } else {
            ptycho->positions.push_back(new PositionArray(_rois + n, 1, 1, batchsize, false, gpus));
        }
        ptycho->roibatch_offset.push_back(n / ngpus);
    }

    for (int g = 0; g < gpus.size(); g++) {
        SetDevice(gpus, g);
        ssc_debug(format("Creating propagator: {}", g));
        ptycho->propagator[g] = new Fraunhoffer();
    }

    ssc_debug("Computing I0");
    SetDevice(gpus, 0);
    ptycho->I0 = ptycho->probe->arrays[0]->Norm2();
    ptycho->probepropagator = new ASM(wavelength_m, pixelsize_m);

    ptycho->object_div = new rMImage(ptycho->object->Shape(), true, gpus);
    ptycho->object_acc = new cMImage(ptycho->object->Shape(), true, gpus);

    ptycho->probe_div = new rMImage(ptycho->probe->Shape(), true, gpus);
    ptycho->probe_acc = new cMImage(ptycho->probe->Shape(), true, gpus);

    return ptycho;
}
