/** @file */
#include <cstddef>

#include "ptycho.hpp"
#include <common/logger.hpp>
#include <common/types.hpp>
#include <common/utils.hpp>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <vector>

extern "C" {

    /**
     * CUDA Kernel. Computes the real space reflector.
     * */
    __global__ void k_RAAR_reflect_Rspace(GArray<complex> exitwave, const GArray<complex> probe, const GArray<complex> object,  const GArray<complex16> wavefront, const Position *p_rois, float objbeta) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= probe.shape.x) return;
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idz = blockIdx.z;

        const int objposx = idx + (int)(p_rois[idz].x); // getting floor the integer value from float pixel value. Should we get the nearest integer?
        const int objposy = idy + (int)(p_rois[idz].y);

        const complex obj = object(objposy, objposx);

        const int num_modes = probe.shape.z;
        for (size_t m = 0; m < num_modes; m++) {
            complex ew = probe(m, idy, idx) * obj * 2.0f -
                complex(wavefront(idz * probe.shape.z + m, idy, idx));
            exitwave(idz * probe.shape.z + m, idy, idx) = ew;
        }
    }

    /**
     * CUDA Kernel. Updates phistack based on the reciprocal space projetor.
     * Additionally, projects the new phistack to object space.
     * */
    __global__ void k_RAAR_wavefront_update(const GArray<complex> object, const GArray<complex> probe, GArray<complex> object_acc,
            GArray<float> object_div, const GArray<complex> p_pm, GArray<complex16> wavefront,
            const Position *p_rois, float objbeta) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= probe.shape.x) return;

        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idz = blockIdx.z;

        const int objposx = idx + (int)(p_rois[idz].x);
        const int objposy = idy + (int)(p_rois[idz].y);

        const complex obj = object(objposy, objposx);

        complex objacc = complex(0);
        float objdiv = 0;
        for (size_t m = 0; m < probe.shape.z; m++) {
            const complex cprobe = probe(m, idy, idx);
            //const complex pm = p_pm(blockIdx.z * probe.shape.z + m, idy, idx) / float(probe.shape.x * probe.shape.y);
            const complex pm = p_pm(blockIdx.z * probe.shape.z + m, idy, idx);
            complex wf = complex(wavefront(blockIdx.z * probe.shape.z + m, idy, idx));

            wf = (pm + wf) * objbeta + obj * cprobe * (1 - 2 * objbeta);
            wavefront(blockIdx.z * probe.shape.z + m, idy, idx) = complex16(wf);

            objacc += wf * cprobe.conj();
            objdiv += cprobe.abs2();
        }

        objacc -= obj * objdiv;

        atomicAdd(&object_acc(objposy, objposx), objacc);
        atomicAdd(&object_div(objposy, objposx), objdiv);
    }

    /**
     * CUDA Kernel. Projects phistack to the object subspace.
     * */
    __global__ void KRAAR_ObjPs(const GArray<complex> object, const GArray<complex> probe,
            const GArray<complex16> phistack, const GArray<Position> rois,
            GArray<complex> object_acc, GArray<float> object_div) {
        const int objidx = blockIdx.x * blockDim.x + threadIdx.x;
        if (objidx >= object.shape.x) return;

        const int objidy = blockIdx.y * blockDim.y + threadIdx.y;

        const complex obj = object(objidy, objidx);

        complex objacc = complex(0);
        float objdiv = 0;

        for (size_t p = 0; p < rois.shape.z; p++) {
            const int pidx = objidx - (int)rois(p, 0, 0).x;
            const int pidy = objidy - (int)rois(p, 0, 0).y;

            if (pidx >= 0 && pidy >= 0 && pidx < probe.shape.x && pidy < probe.shape.y)
                for (size_t m = 0; m < probe.shape.z; m++) {
                    complex cprobe = probe(m, pidy, pidx);
                    objacc += cprobe.conj() * complex(phistack(p * probe.shape.z + m, pidy, pidx));
                    objdiv += cprobe.abs2();
                }
        }

        object_acc(objidy, objidx) += objacc - obj * objdiv;
        object_div(objidy, objidx) += objdiv;
    }
}

RAAR *CreateRAAR(float *difpads, const dim3 &difshape, complex *probe, const dim3 &probeshape, complex *object,
        const dim3 &objshape, Position *rois, int numrois, int batchsize, float *rfact,
        const std::vector<int> &gpus, float *objsupp, float *probesupp, int numobjsupp,
        float wavelength_m, float pixelsize_m, float distance_m,
        int poscorr_iter,
        float step_obj, float step_probe,
        float reg_obj, float reg_probe) {
    RAAR *raar = new RAAR();

    raar->ptycho =
        CreatePtycho(difpads, difshape, probe, probeshape, object, objshape, rois, numrois, batchsize, rfact,
                gpus, objsupp, probesupp, numobjsupp,
                wavelength_m, pixelsize_m, distance_m,
                poscorr_iter,
                step_obj, step_probe,
                reg_obj, reg_probe);

    const size_t wavefront_size = raar->ptycho->probe->size
        * raar->ptycho->total_num_rois * raar->ptycho->gpus.size();

    const size_t num_batches = PtychoNumBatches(*raar->ptycho);

    raar->previous_wavefront.reserve(num_batches);

    for (int i = 0; i < num_batches; i++) {
        size_t batchsize = raar->ptycho->positions[i]->arrays[0]->sizez;
        auto *newphistack =
            new hcMImage(raar->ptycho->probe->sizex, raar->ptycho->probe->sizey,
                    batchsize * raar->ptycho->probe->sizez, true,
                    raar->ptycho->gpus, MemoryType::EAllocGPU);
        newphistack->SetGPUToZero();

        raar->previous_wavefront.push_back(newphistack);
    }
    return raar;
}

void DestroyRAAR(RAAR *&raar) {
    for (auto *phi : raar->previous_wavefront) {
        delete phi;
    }
    DestroyPtycho(raar->ptycho);
    raar = nullptr;
}


void RAARProjectProbe(RAAR& raar, int section, hcMImage& wavefront, cudaStream_t stream = 0) {
    ProjectPhiToProbe(*(raar.ptycho), section, wavefront, false, raar.isGradPm, stream);
}

/**
 * Applies KRAAR_ObjPs and updates the object estimate.
 * */
void RAARApplyObjectUpdate(RAAR &raar, cImage &velocity, float stepsize, float momentum, float epsilon) {
    raar.ptycho->object_num->SetGPUToZero();
    raar.ptycho->object_div->SetGPUToZero();

    dim3 blk = raar.ptycho->object->ShapeBlock();
    dim3 thr = raar.ptycho->object->ShapeThread();

    for (int section = 0; section < raar.ptycho->positions.size(); section++) {

        for (int g = 0; g < raar.ptycho->gpus.size(); g++) {

            SetDevice(raar.ptycho->gpus, g);

            const int cur_difpad_z = raar.ptycho->positions[section]->arrays[g]->sizez;

            if (cur_difpad_z > 0) {

                KRAAR_ObjPs<<<blk, thr>>>(*raar.ptycho->object->arrays[g],
                        *raar.ptycho->probe->arrays[g],
                        *raar.previous_wavefront[section]->arrays[g],
                        *raar.ptycho->positions[section]->arrays[g],
                        *raar.ptycho->object_num->arrays[g],
                        *raar.ptycho->object_div->arrays[g]);

            }
        }
    }

    raar.ptycho->object->WeightedLerpSync(*raar.ptycho->object_num, *raar.ptycho->object_div,
            raar.ptycho->objstep, momentum, velocity, epsilon);
}

/**
 * Projects phistack to the probe subspace and calls Super::ApplyProbeUpdate
 * */
void RAARApplyProbeUpdate(RAAR& raar, cImage &velocity,
        float stepsize, float momentum, float epsilon) {
    raar.ptycho->probe_num->SetGPUToZero();
    raar.ptycho->probe_div->SetGPUToZero();

    const size_t num_batches = PtychoNumBatches(*raar.ptycho);
    for (int d = 0; d < num_batches; d++) {
        RAARProjectProbe(raar, d, *raar.previous_wavefront[d]);
    }

    ApplyProbeUpdate(*raar.ptycho, velocity, stepsize, momentum, epsilon);
}

void init_wavefront(RAAR& raar) {
    for (MImage<complex16> *wavefront : raar.previous_wavefront) {
        if (wavefront != nullptr) {
            wavefront->SetGPUToZero();
        }
    }
}

/**
 * RAAR iteration loop.
 * */
void RAARRun(RAAR& raar, int iterations) {
    sscInfo("Starting RAAR iteration.");

    const dim3 probeshape = raar.ptycho->probe->Shape();

    init_wavefront(raar);

    cImage objvelocity(raar.ptycho->object->Shape()); // imagem complex com mesmo shape de object
    cImage probevelocity(raar.ptycho->probe->Shape());
    objvelocity.SetGPUToZero();
    probevelocity.SetGPUToZero();

    auto time0 = sscTime();

    const dim3 difpadshape = raar.ptycho->diff_pattern_shape;
    const dim3 ew_shape = raar.ptycho->wavefront->Shape();

    const int nstreams = 3;
    const int ngpus = raar.ptycho->gpus.size();
    cudaStream_t streams[nstreams][ngpus];

    rMImage* cur_difpad_vec[nstreams];
    cMImage* exitwave[nstreams];
    cMImage* object_num[nstreams];
    rMImage* object_div[nstreams];

    for (int st = 0; st < nstreams; ++st) {
        cur_difpad_vec[st] = new rMImage(difpadshape.x, difpadshape.y, raar.ptycho->multibatchsize,
                false, raar.ptycho->gpus, MemoryType::EAllocGPU);
        exitwave[st] = new cMImage(ew_shape, true, raar.ptycho->gpus, MemoryType::EAllocGPU);
        object_num[st] = new cMImage(raar.ptycho->object->Shape(),
                true, raar.ptycho->gpus, MemoryType::EAllocGPU);
        object_div[st] = new rMImage(raar.ptycho->object->Shape(),
                true, raar.ptycho->gpus, MemoryType::EAllocGPU);
    }

    for (int g = 0; g < ngpus; ++g) {
        SetDevice(raar.ptycho->gpus, g);
        for (int st = 0; st < nstreams; ++st) {
            cudaStreamCreate(&streams[st][g]);
        }
    }

    for (int iter = 0; iter < iterations; iter++) {

        raar.ptycho->error->SetGPUToZero();
        raar.ptycho->object_num->SetGPUToZero();
        raar.ptycho->object_div->SetGPUToZero();

        const size_t num_batches = PtychoNumBatches(*raar.ptycho);
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {

            const int stream_idx = batch_idx % nstreams;
            rMImage* cur_difpad = cur_difpad_vec[stream_idx];

            const size_t difpad_batch_zsize = raar.ptycho->positions[batch_idx]->sizez;
            const size_t global_idx = batch_idx * raar.ptycho->multibatchsize;
            float *difpad_batch_ptr = raar.ptycho->cpu_diff_pattern +
                global_idx * difpadshape.x * difpadshape.y;

            cur_difpad->Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
            cur_difpad->LoadToGPU(difpad_batch_ptr, streams[stream_idx]);

            const size_t ngpus = PtychoNumGpus(*raar.ptycho);
            for (int gpu_idx = 0; gpu_idx < ngpus; gpu_idx++) {

                SetDevice(raar.ptycho->gpus, gpu_idx);

                cudaStream_t stream = streams[stream_idx][gpu_idx];

                const size_t cur_difpad_zsize = raar.ptycho->positions[batch_idx]->arrays[gpu_idx]->sizez;

                cImage* current_exit_wave = exitwave[stream_idx]->arrays[gpu_idx];
                //cImage* current_exit_wave = raar.ptycho->wavefront->arrays[gpu_idx];
                cImage* current_object = raar.ptycho->object->arrays[gpu_idx];
                cImage* current_probe = raar.ptycho->probe->arrays[gpu_idx];
                rImage* current_obj_div = object_div[stream_idx]->arrays[gpu_idx];
                cImage* current_obj_acc = object_num[stream_idx]->arrays[gpu_idx];
                hcImage* previous_exit_wave = raar.previous_wavefront[batch_idx]->arrays[gpu_idx];

                if (cur_difpad_zsize > 0) {

                    dim3 blk = raar.ptycho->wavefront->ShapeBlock();
                    blk.z = cur_difpad_zsize;
                    dim3 thr = raar.ptycho->wavefront->ShapeThread();

                    Position* ptr_roi = raar.ptycho->positions[batch_idx]->Ptr(gpu_idx);

                    k_RAAR_reflect_Rspace<<<blk, thr, 0, stream>>>(*current_exit_wave, *current_probe, *current_object, *previous_exit_wave, ptr_roi, raar.beta);

                    ProjectReciprocalSpace(*raar.ptycho, cur_difpad->arrays[gpu_idx], current_exit_wave,
                            gpu_idx, raar.isGradPm, stream); // propagate, apply measured intensity and unpropagate

                    //normalize inverse cufft output
                    current_exit_wave->scale(1.0 / float(probeshape.x * probeshape.y), stream);

                    k_RAAR_wavefront_update<<<blk, thr, 0, stream>>>(*current_object, *current_probe,   *current_obj_acc, *current_obj_div,  *current_exit_wave, *previous_exit_wave, ptr_roi, raar.beta);

                }
                raar.previous_wavefront[batch_idx]->arrays[gpu_idx]->CopyFrom(
                        previous_exit_wave->gpuptr,
                        stream);
            }
        }


        for (int g = 0; g < ngpus; ++g) {
            SetDevice(raar.ptycho->gpus, g);
            for (int st = 0; st < nstreams; ++st) {
                cudaStreamSynchronize(streams[st][g]);
            }
        }

        //TODO: perhaps we should not use an extra num and div on ptycho, and just reuse one of the streams
        //This would involve not keeping object_num and div on the ptycho, and only obj
        //and leave num and div as temporary internal data on the Run only
        ReduceMImages(raar.ptycho->object_num, object_num, nstreams);
        ReduceMImages(raar.ptycho->object_div, object_div, nstreams);

        sscDebug("Syncing OBJ");
        raar.ptycho->object->WeightedLerpSync(*(raar.ptycho->object_num), *(raar.ptycho->object_div), raar.ptycho->objstep, raar.ptycho->objmomentum, objvelocity, raar.ptycho->objreg);

        sscDebug("Applying probe");
        probevelocity.SetGPUToZero();

        if (iter != 0) {
            RAARApplyProbeUpdate(raar, probevelocity,
                    raar.ptycho->probestep, raar.ptycho->probemomentum, raar.ptycho->probereg); // updates in real space
            RAARApplyObjectUpdate(raar, objvelocity,
                    raar.ptycho->objstep, raar.ptycho->objmomentum, raar.ptycho->objreg);
        }

        if (raar.ptycho->poscorr_iter &&
                (iter + 1) % raar.ptycho->poscorr_iter == 0)
            ApplyPositionCorrection(*raar.ptycho);

        raar.ptycho->cpuerror[iter] = sqrtf(raar.ptycho->error->SumGPU());

        if (iter % 10 == 0) {
            sscInfo(format("iter {}/{} error: {}",
                        iter, iterations, raar.ptycho->cpuerror[iter]));
        }

    }


    for (int st = 0; st < nstreams; ++st) {
        delete cur_difpad_vec[st];
        delete exitwave[st];
        delete object_num[st];
        delete object_div[st];
    }

    for (int g = 0; g < ngpus; ++g) {
        SetDevice(raar.ptycho->gpus, g);
        for (int st = 0; st < nstreams; ++st) {
            cudaStreamDestroy(streams[st][g]);
        }
    }

    auto time1 = sscTime();
    sscInfo(format("End RAAR iteration: {} ms", ssc_diff_time(time0, time1)));

}

