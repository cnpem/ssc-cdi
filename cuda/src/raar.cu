/** @file */
#include <cstddef>

#include "ptycho.hpp"
#include <common/logger.hpp>
#include <common/types.hpp>
#include <common/utils.hpp>

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
                        GArray<float> object_div, const GArray<complex> p_pm, GArray<complex16> phistack,
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
        complex phi = complex(phistack(blockIdx.z * probe.shape.z + m, idy, idx));

        phi = (pm + phi) * objbeta + obj * cprobe * (1 - 2 * objbeta);
        phistack(blockIdx.z * probe.shape.z + m, idy, idx) = complex16(phi);

        objacc += phi * cprobe.conj();
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
                 float probef1,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe) {
    RAAR *raar = new RAAR();

    raar->ptycho =
        CreatePOptAlgorithm(difpads, difshape, probe, probeshape, object, objshape, rois, numrois, batchsize, rfact,
                            gpus, objsupp, probesupp, numobjsupp, probef1,
                            step_obj, step_probe, reg_obj, reg_probe);

    const size_t wavefront_size = raar->ptycho->probe->size
        * raar->ptycho->total_num_rois * raar->ptycho->gpus.size();

    const size_t num_batches = ptycho_num_batches(*raar->ptycho);
    for (int i = 0; i < num_batches; i++) {
        size_t batchsize = raar->ptycho->positions[i]->arrays[0]->sizez;
        auto *newphistack =
            new hcMImage(raar->ptycho->probe->sizex, raar->ptycho->probe->sizey,
                        batchsize * raar->ptycho->probe->sizez, true,
                        raar->ptycho->gpus, MemoryType::EAllocGPU);
        newphistack->SetGPUToZero();
        raar->phistack.push_back(newphistack);
    }
    return raar;
}

void DestroyRAAR(RAAR *&raar) {
    for (auto *phi : raar->phistack) {
        delete phi;
    }
    DestroyPOptAlgorithm(raar->ptycho);
    raar = nullptr;
}


void RAARProjectProbe(RAAR& raar, int section, hcMImage& wavefront) {
    ProjectPhiToProbe(*(raar.ptycho), section, wavefront, false, raar.isGradPm);
}

/**
 * Applies KRAAR_ObjPs and updates the object estimate.
 * */
void RAARApplyObjectUpdate(RAAR &raar, cImage &velocity, float stepsize, float momentum, float epsilon) {
    raar.ptycho->object_acc->SetGPUToZero();
    raar.ptycho->object_div->SetGPUToZero();

    dim3 blk = raar.ptycho->object->ShapeBlock();
    dim3 thr = raar.ptycho->object->ShapeThread();

    for (int section = 0; section < raar.ptycho->positions.size(); section++) {

        for (int g = 0; g < raar.ptycho->gpus.size(); g++)
            if (raar.ptycho->positions[section]->arrays[g]->sizez > 0) {
                SetDevice(raar.ptycho->gpus, g);

                KRAAR_ObjPs<<<blk, thr>>>(raar.ptycho->object->arrays[g][0], raar.ptycho->probe->arrays[g][0],
                                          raar.phistack[section]->arrays[g][0],
                                          raar.ptycho->positions[section]->arrays[g][0],
                                          raar.ptycho->object_acc->arrays[g][0], raar.ptycho->object_div->arrays[g][0]);

            }
    }

    raar.ptycho->object->WeightedLerpSync(*raar.ptycho->object_acc, *raar.ptycho->object_div,
            raar.ptycho->objstep, momentum, velocity, epsilon);
}

/**
 * Projects phistack to the probe subspace and calls Super::ApplyProbeUpdate
 * */
void RAARApplyProbeUpdate(RAAR& raar, cImage &velocity,
        float stepsize, float momentum, float epsilon) {
    raar.ptycho->probe_acc->SetGPUToZero();
    raar.ptycho->probe_div->SetGPUToZero();

    const size_t num_batches = ptycho_num_batches(*raar.ptycho);
    for (int d = 0; d < num_batches; d++) {
          RAARProjectProbe(raar, d, *raar.phistack[d]);
    }

    ApplyProbeUpdate(*raar.ptycho, velocity, stepsize, momentum, epsilon);
}

void init_wavefront(RAAR& raar) {
    for (MImage<complex16> *wavefront : raar.phistack) {
        if (wavefront != nullptr) {
            wavefront->SetGPUToZero();
        }
    }
}

/**
 * RAAR iteration loop.
 * */
void RAARRun(RAAR& raar, int iterations) {
    ssc_info("Starting RAAR iteration.");

    const dim3 probeshape = raar.ptycho->probe->Shape();

    init_wavefront(raar);

    cImage objvelocity(raar.ptycho->object->Shape()); // imagem complex com mesmo shape de object
    cImage probevelocity(raar.ptycho->probe->Shape());
    objvelocity.SetGPUToZero();
    probevelocity.SetGPUToZero();

    auto time0 = ssc_time();

    const dim3 difpadshape = raar.ptycho->difpadshape;

    rMImage cur_difpad(difpadshape.x, difpadshape.y, raar.ptycho->multibatchsize,
                false, raar.ptycho->gpus, MemoryType::EAllocGPU);

    for (int iter = 0; iter < iterations; iter++) {

        raar.ptycho->rfactors->SetGPUToZero();
        raar.ptycho->object_acc->SetGPUToZero();
        raar.ptycho->object_div->SetGPUToZero();

        const size_t num_batches = ptycho_num_batches(*raar.ptycho);
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {

          const size_t difpad_batch_zsize = raar.ptycho->positions[batch_idx]->sizez;
          const size_t global_idx = batch_idx * raar.ptycho->multibatchsize;
          float *difpad_batch_ptr = raar.ptycho->cpudifpads +
              global_idx * difpadshape.x * difpadshape.y;

          cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
          cur_difpad.LoadToGPU(difpad_batch_ptr);

           const size_t ngpus = ptycho_num_gpus(*raar.ptycho);
           for (int gpu_idx = 0; gpu_idx < ngpus; gpu_idx++) {

                cImage* current_exit_wave = raar.ptycho->exitwave->arrays[gpu_idx];
                cImage* current_object = raar.ptycho->object->arrays[gpu_idx];
                cImage* current_probe = raar.ptycho->probe->arrays[gpu_idx];
                Image2D<complex16>* current_measurement = raar.phistack[batch_idx]->arrays[gpu_idx];
                rImage* current_obj_div = raar.ptycho->object_div->arrays[gpu_idx];
                cImage* current_obj_acc = raar.ptycho->object_acc->arrays[gpu_idx];

                const size_t cur_difpad_zsize = raar.ptycho->positions[batch_idx]->arrays[gpu_idx]->sizez;

                if (cur_difpad_zsize > 0) {
                    SetDevice(raar.ptycho->gpus, gpu_idx);

                    dim3 blk = raar.ptycho->exitwave->ShapeBlock();
                    blk.z = cur_difpad_zsize;
                    dim3 thr = raar.ptycho->exitwave->ShapeThread();

                    Position* ptr_roi = raar.ptycho->positions[batch_idx]->Ptr(gpu_idx);

                    k_RAAR_reflect_Rspace<<<blk, thr>>>(*current_exit_wave, *current_probe, *current_object, *current_measurement, ptr_roi, raar.beta);

                    project_reciprocal_space(*raar.ptycho,
                            cur_difpad.arrays[gpu_idx], gpu_idx, raar.isGradPm); // propagate, apply measured intensity and unpropagate

                    //normalize inverse cufft output
                    *current_exit_wave /= float(probeshape.x * probeshape.y);

                    k_RAAR_wavefront_update<<<blk, thr>>>(*current_object, *current_probe,   *current_obj_acc, *current_obj_div,  *current_exit_wave, *current_measurement, ptr_roi, raar.beta);

                }
            }
        }


        ssc_debug("Syncing OBJ");
        objvelocity.SetGPUToZero();
        raar.ptycho->object->WeightedLerpSync(*(raar.ptycho->object_acc), *(raar.ptycho->object_div), raar.ptycho->objstep, raar.ptycho->objmomentum, objvelocity, raar.ptycho->objreg);

        ssc_debug("Applying probe");
        probevelocity.SetGPUToZero();

        if (iter != 0) {
            RAARApplyProbeUpdate(raar, probevelocity,
                    raar.ptycho->probestep, raar.ptycho->probemomentum, raar.ptycho->probereg); // updates in real space
            RAARApplyObjectUpdate(raar, objvelocity,
                    raar.ptycho->objstep, raar.ptycho->objmomentum, raar.ptycho->objreg);
        }

        raar.ptycho->cpurfact[iter] = sqrtf(raar.ptycho->rfactors->SumGPU());

        if (iter % 10 == 0) {
            ssc_info(format("iter {}/{} error: {}",
                        iter, iterations, raar.ptycho->cpurfact[iter]));
        }

    }

    auto time1 = ssc_time();
    ssc_info(format("End RAAR iteration: {} ms", ssc_diff_time(time0, time1)));

}

