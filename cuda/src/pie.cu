#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

#include "complex.hpp"
#include "logger.hpp"
#include "ptycho.hpp"
#include "types.hpp"
#include "utils.hpp"

Pie* CreatePie(float* difpads, const dim3& difshape,
        complex* probe, const dim3& probeshape,
        complex* object, const dim3& objshape,
        ROI* rois, int numrois,
        int batchsize, float* rfact,
        const std::vector<int>& gpus,
        float* objsupp, float* probesupp, int numobjsupp,
        float* sigmask,  // TODO: can we remove sigmask, geometricsteps, background and probef1?
        int geometricsteps, float* background,
        float probef1, float step_object, float step_probe, float reg_obj,
        float reg_probe) {
    Pie* pie = new Pie();

    pie->ptycho = CreatePOptAlgorithm(difpads, difshape,
            probe, probeshape,
            object, objshape,
            rois, numrois,
            batchsize, rfact,
            gpus,
            objsupp, probesupp, numobjsupp,
            sigmask, geometricsteps,
            background, probef1,
            step_object, step_probe, reg_obj, reg_probe);

    return pie;
}

void DestroyPie(Pie*& pie) {
    DestroyPOptAlgorithm(pie->ptycho);
    pie = nullptr;
}

__global__ void k_pie_wavefront_calc(GArray<complex> wavefront, const GArray<complex> probe,
        const GArray<complex> object, const ROI* rois) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int m = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= probe.shape.x || idy >= probe.shape.y) return;

    const size_t objposx = (int) rois[0].x + idx;
    const size_t objposy = (int) rois[0].y + idy;

    const complex obj = object(objposy, objposx);

    const int num_modes = probe.shape.z;
    for (int m = 0; m < num_modes; ++m) {
        wavefront(m, idy, idx) = obj * probe(m, idy, idx);
    }
}

__global__ void k_pie_update_probe(GArray<complex> object_box,
        GArray<complex> object, GArray<complex> probe,
        GArray<complex> wavefront, GArray<complex> wavefront_prev,
        float reg_probe, float step_probe, float obj_abs2_max, const ROI* rois) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= probe.shape.x || idy >= probe.shape.y) return;

    const int num_modes = probe.shape.z;

    const size_t objposx = (int) rois[0].x + idx;
    const size_t objposy = (int) rois[0].y + idy;

    const complex obj = object(objposy, objposx);
    const complex obj_conj = obj.conj();


    const double obj_abs2_sum = obj.abs2() * num_modes;

    const double denominator_p = (1.0 - reg_probe) * obj_abs2_sum + reg_probe * obj_abs2_max;

    //update probe
    for (int m = 0; m < num_modes; ++m) {

        const complex delta_wavefront = wavefront(m, idy, idx) - wavefront_prev(m, idy, idx);
        const complex probe_delta = obj_conj * delta_wavefront;

        probe(m, idy, idx) += (step_probe * probe_delta) / denominator_p;
    }
}

__global__ void k_pie_update_object(GArray<complex> object, GArray<complex> probe,
        GArray<complex> wavefront, GArray<complex> wavefront_prev,
        float reg_obj, float step_obj, float probe_abs2_max, const ROI* rois) {

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

    const double denominator_o = (1.0 - reg_obj) * probe_abs2_sum + reg_obj * probe_abs2_max;
    object(objposy, objposx) += (step_obj * obj_delta) / denominator_o;
}

void range_array(int* data, size_t n) {
    for (int i = 0; i < n; ++i) {
        data[i] = i;
    }
}

void shuffle_array(int *data, size_t n) {
    std::shuffle(data, data + n, std::default_random_engine());
}

void PieRun(Pie& pie, int iterations) {

    ssc_assert(ptycho_num_batches(*pie.ptycho), "This algorithm does not support MultiGPU.");
    ssc_assert(ptycho_batch_size(*pie.ptycho) == 1, "Batch > 1 is not supported for PIE.");

    ssc_event_start("PieRun", {
            ssc_param_int("iter", iterations),
            ssc_param_int("difpadshape.x", (int)pie.ptycho->difpadshape.x),
            ssc_param_int("difpadshape.y", (int)pie.ptycho->difpadshape.y),
            ssc_param_int("difpadshape.z", (int)pie.ptycho->difpadshape.z)
    });

    const int gpu = 0;

    const int batch_size = 1;

    dim3 probeshape = pie.ptycho->probe->Shape();
    dim3 objectshape = pie.ptycho->object->Shape();
    dim3 difpadshape = pie.ptycho->difpadshape;

    SetDevice(pie.ptycho->gpus, gpu);

    cImage obj_box(probeshape);

    const int num_modes = ptycho_num_modes(*pie.ptycho);

    cImage wavefront_prev(*pie.ptycho->exitwave->arrays[0]);

    auto time0 = ssc_time();

    // when (batchsize == 1) => (num_batches == num_rois)
    const size_t num_rois = ptycho_num_batches(*pie.ptycho);
    int random_idx[num_rois];
    range_array(random_idx, num_rois);
    for (int iter = 0; iter < iterations; ++iter) {
        ssc_event_start("iter", { ssc_param_int("iter", iter) });
        pie.ptycho->rfactors->SetGPUToZero();

        shuffle_array(random_idx, num_rois);
        for (int pos_idx = 0; pos_idx < num_rois; ++pos_idx) {
            const size_t random_pos_idx = random_idx[pos_idx];

            float* difpad_batch_ptr = pie.ptycho->cpudifpads +
                random_pos_idx * difpadshape.x * difpadshape.y;

            // TODO: improve so we can avoid reallocating arrays every iteration,
            // if we need a speedup
            rMImage cur_difpad(difpad_batch_ptr,
                    difpadshape.x, difpadshape.y, batch_size, false,
                    pie.ptycho->gpus, MemoryType::EAllocCPUGPU);

            dim3 blk = pie.ptycho->exitwave->ShapeBlock();
            blk.z = batch_size;
            dim3 thr = pie.ptycho->exitwave->ShapeThread();

            ROI* rois = pie.ptycho->rois[random_pos_idx]->Ptr(gpu);
            cImage* probe = pie.ptycho->probe->arrays[gpu];
            cImage* obj = pie.ptycho->object->arrays[gpu];
            cImage* wavefront = pie.ptycho->exitwave->arrays[gpu];
            rImage* difpad = cur_difpad.arrays[gpu];

            k_pie_wavefront_calc<<<blk, thr>>>(*wavefront, *probe, *obj, rois);

            wavefront->CopyTo(wavefront_prev);

            project_reciprocal_space(*pie.ptycho, difpad, gpu, pie.isGradPm);

            *wavefront /= float(probeshape.x * probeshape.y);

            const float probe_abs2_max = probe->maxAbs2();
            k_pie_update_object<<<blk, thr>>>(*obj, *probe,
                    *wavefront, wavefront_prev,
                    pie.ptycho->objreg,
                    pie.ptycho->objstep,
                    probe_abs2_max, rois);

            const dim3 pos_offset(pie.ptycho->cpurois[random_pos_idx].x,
                    pie.ptycho->cpurois[random_pos_idx].y, 0);
            obj->CopyRoiTo(obj_box, pos_offset, probeshape);
            const float obj_abs2_max = obj_box.maxAbs2();

            k_pie_update_probe<<<blk, thr>>>(obj_box, *obj, *probe,
                    *wavefront, wavefront_prev,
                    pie.ptycho->probereg,
                    pie.ptycho->probestep,
                    obj_abs2_max, rois);

        }

        pie.ptycho->cpurfact[iter] = sqrtf(pie.ptycho->rfactors->SumCPU());
        if (iter % 10 == 0) {
            ssc_info(format("iter {}/{} error = {}",
                        iter, iterations, pie.ptycho->cpurfact[iter]));
        }
        ssc_event_stop(); // iter
    }

    auto time1 = ssc_time();
    ssc_info(format("End PIE iteration: {} ms", ssc_diff_time(time0, time1)));

    ssc_event_stop();  // PieRun
}

