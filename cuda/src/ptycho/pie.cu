#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>

#include "complex.hpp"
#include "logger.hpp"
#include "engines_common.hpp"
#include "types.hpp"
#include "utils.hpp"

Pie* CreatePie(float* difpads, const dim3& difshape,
        complex* probe, const dim3& probeshape,
        complex* object, const dim3& objshape,
        Position* rois, int numrois,
        int batchsize, float* rfact,
        const std::vector<int>& gpus,
        float* objsupp, float* probesupp, int numobjsupp,
        float wavelength_m, float pixelsize_m, float distance_m,
        int poscorr_iter,
        float step_object, float step_probe, float reg_obj, float reg_probe) {
    Pie* pie = new Pie();

    pie->ptycho = CreatePtycho(difpads, difshape,
            probe, probeshape,
            object, objshape,
            rois, numrois,
            batchsize, rfact,
            gpus,
            objsupp, probesupp, numobjsupp,
            wavelength_m, pixelsize_m, distance_m,
            poscorr_iter,
            step_object, step_probe,
            reg_obj, reg_probe);

    return pie;
}

void DestroyPie(Pie*& pie) {
    DestroyPtycho(pie->ptycho);
    pie = nullptr;
}

__global__ void kPieWavefrontCalc(GArray<complex> wavefront, const GArray<complex> probe,
        const GArray<complex> object, const Position* rois) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x || idy >= probe.shape.y) return;

    const size_t objposx = (int) rois[0].x + idx;
    const size_t objposy = (int) rois[0].y + idy;

    const complex obj = object(objposy, objposx);

    const int num_modes = probe.shape.z;
    for (int m = 0; m < num_modes; ++m) {
        wavefront(m, idy, idx) = obj * probe(m, idy, idx);
    }
}

__global__ void kPieUpdateProbe(GArray<complex> object_box,
        GArray<complex> object, GArray<complex> probe,
        GArray<complex> wavefront, GArray<complex> wavefront_prev,
        float reg_probe, float step_probe, float obj_abs2_max, const Position* rois) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x || idy >= probe.shape.y) return;

    const int num_modes = probe.shape.z;

    const size_t objposx = (int) rois[0].x + idx;
    const size_t objposy = (int) rois[0].y + idy;

    const complex obj = object(objposy, objposx);
    const complex obj_conj = obj.conj();


    const double obj_abs2_sum = obj.abs2() * num_modes;

    const double denominator_p = (1.0 - reg_probe) * obj_abs2_sum + reg_probe * (obj_abs2_max);

    //update probe
    for (int m = 0; m < num_modes; ++m) {

        const complex delta_wavefront = wavefront(m, idy, idx) - wavefront_prev(m, idy, idx);
        const complex probe_delta = obj_conj * delta_wavefront;

        probe(m, idy, idx) += (step_probe * probe_delta) / denominator_p;
    }
}

__global__ void kPieUpdateObject(GArray<complex> object, GArray<complex> probe,
        GArray<complex> wavefront, GArray<complex> wavefront_prev,
        float reg_obj, float step_obj, float probe_abs2_max, const Position* rois) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x || idy >= probe.shape.y) return;

    const int num_modes = probe.shape.z;

    const size_t objposx = (int) rois[0].x + idx;
    const size_t objposy = (int) rois[0].y + idy;

    float probe_abs2_sum = 0.0f;

    for (int m = 0; m < num_modes; ++m) {
        probe_abs2_sum += probe(m, idy, idx).abs2();
    }

    //update object
    complex obj_delta = complex(0);
    for (int m = 0; m < num_modes; ++m) {
        const complex delta_wavefront = wavefront(m, idy, idx) - wavefront_prev(m, idy, idx);
        obj_delta += probe(m, idy, idx).conj() * delta_wavefront;
    }

    const double denominator_o = (1.0 - reg_obj) * probe_abs2_sum + reg_obj * (probe_abs2_max);
    object(objposy, objposx) += (step_obj * obj_delta) / denominator_o;
}

void rangeArray(int* data, size_t n) {
    for (int i = 0; i < n; ++i) {
        data[i] = i;
    }
}

void shuffleArray(int *data, size_t n) {
    std::shuffle(data, data + n, std::default_random_engine());
}

//get maxAbs2 without bringing data to cpu
__global__ void maxAbs2(complex *d_array, float *d_max, size_t size) {
    __shared__ float sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *d_max = -FLT_MAX;
    }

    if (idx < size) {
        sdata[tid] = d_array[idx].abs2();
    } else {
        sdata[tid] = -FLT_MAX;  // handle out of bounds
    }
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(d_max), __float_as_int(sdata[0]));
    }
}

void PieRun(Pie& pie, int iterations) {

    sscAssert(PtychoNumBatches(*pie.ptycho), "This algorithm does not support MultiGPU.");
    sscAssert(PtychoBatchSize(*pie.ptycho) == 1, "Batch > 1 is not supported for PIE.");

    const int gpu = 0;

    const int batch_size = 1;

    dim3 probeshape = pie.ptycho->probe->Shape();
    dim3 objectshape = pie.ptycho->object->Shape();
    dim3 difpadshape = pie.ptycho->diff_pattern_shape;
    dim3 roishape = dim3(probeshape.x, probeshape.y, 1);

    SetDevice(pie.ptycho->gpus, gpu);

    cImage obj_box(probeshape, MemoryType::EAllocGPU);

    const int num_modes = ptycho_num_modes(*pie.ptycho);

    cImage wavefront_prev(*pie.ptycho->wavefront->arrays[0]);

    rMImage cur_difpad(difpadshape.x, difpadshape.y, batch_size,
            false, pie.ptycho->gpus, MemoryType::EAllocGPU);

    auto time0 = sscTime();

    //float *obj_abs2_max;
    //float *probe_abs2_max;

    //cudaMalloc((void**)&obj_abs2_max, sizeof(float));
    //cudaMalloc((void**)&probe_abs2_max, sizeof(float));

    // when (batchsize == 1) => (num_batches == num_rois)
    const size_t num_rois = PtychoNumBatches(*pie.ptycho);
    int random_idx[num_rois];
    rangeArray(random_idx, num_rois);
    for (int iter = 0; iter < iterations; ++iter) {
        pie.ptycho->error->SetGPUToZero();

        shuffleArray(random_idx, num_rois);
        for (int pos_idx = 0; pos_idx < num_rois; ++pos_idx) {
            const size_t random_pos_idx = random_idx[pos_idx];

            float* difpad_batch_ptr = pie.ptycho->cpu_diff_pattern +
                random_pos_idx * difpadshape.x * difpadshape.y;

            cur_difpad.LoadToGPU(difpad_batch_ptr);

            dim3 blk = pie.ptycho->wavefront->ShapeBlock();
            blk.z = batch_size;
            dim3 thr = pie.ptycho->wavefront->ShapeThread();

            Position* rois = pie.ptycho->positions[random_pos_idx]->Ptr(gpu);
            cImage* probe = pie.ptycho->probe->arrays[gpu];
            cImage* obj = pie.ptycho->object->arrays[gpu];
            cImage* wavefront = pie.ptycho->wavefront->arrays[gpu];
            rImage* difpad = cur_difpad.arrays[gpu];

            kPieWavefrontCalc<<<blk, thr>>>(*wavefront, *probe, *obj, rois);

            wavefront->CopyTo(wavefront_prev);

            ProjectReciprocalSpace(*pie.ptycho, difpad, gpu, pie.isGrad);

            *wavefront /= float(probeshape.x * probeshape.y);

            const float probe_abs2_max = probe->maxAbs2();
            const dim3 pos_offset(pie.ptycho->cpupositions[random_pos_idx].x,
                    pie.ptycho->cpupositions[random_pos_idx].y, 0);
            obj->CopyRoiTo(obj_box, pos_offset, roishape);
            const float obj_abs2_max = obj_box.maxAbs2();

            //get max without going to CPU (will we need this to achieve full scalar???)
            //maxAbs2<<<(obj_box.size + 1024 - 1) / 1024, 1024>>>(obj_box.gpuptr, obj_abs2_max, obj_box.size);
            //maxAbs2<<<(probe->size + 1024 - 1) / 1024, 1024>>>(probe->gpuptr, probe_abs2_max, probe->size);

            kPieUpdateObject<<<blk, thr>>>(*obj, *probe,
                    *wavefront, wavefront_prev,
                    pie.ptycho->objreg,
                    pie.ptycho->objstep,
                    probe_abs2_max, rois);

            kPieUpdateProbe<<<blk, thr>>>(obj_box, *obj, *probe,
                    *wavefront, wavefront_prev,
                    pie.ptycho->probereg,
                    pie.ptycho->probestep,
                    obj_abs2_max, rois);

        }

        if (pie.ptycho->poscorr_iter &&
                (iter + 1) % pie.ptycho->poscorr_iter == 0)
            ApplyPositionCorrection(*pie.ptycho);

        pie.ptycho->cpuerror[iter] = sqrtf(pie.ptycho->error->SumGPU());
        if (iter % 10 == 0) {
            sscInfo(format("iter {}/{} error = {}",
                        iter, iterations, pie.ptycho->cpuerror[iter]));
        }
    }

    //cudaFree(obj_abs2_max);
    //cudaFree(probe_abs2_max);

    auto time1 = sscTime();
    sscInfo(format("End PIE iteration: {} ms", sscDiffTime(time0, time1)));
}

