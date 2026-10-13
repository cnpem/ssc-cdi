/** @file */
#include <cstddef>

#include "engines_common.hpp"
#include <common/logger.hpp>
#include <common/types.hpp>
#include <common/utils.hpp>
#include <cuda_runtime_api.h>

extern "C" {

    /**
     * CUDA Kernel. Computes the real space reflector.
     * */
    __global__ void k_RAAR_reflect_Rspace(GArray<complex> exitwave, const GArray<complex> probe, const GArray<complex> object,  const GArray<complex> wavefront, const Position *p_rois, float objbeta) {
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
                wavefront(idz * probe.shape.z + m, idy, idx);
            exitwave(idz * probe.shape.z + m, idy, idx) = ew;
        }
    }

    /**
     * CUDA Kernel. Updates phistack based on the reciprocal space projetor.
     * Additionally, projects the new phistack to object space.
     * */
    __global__ void k_RAAR_wavefront_update(const GArray<complex> object, const GArray<complex> probe, GArray<complex> object_acc,
            GArray<float> object_div, const GArray<complex> p_pm, GArray<complex> wavefront,
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
            const complex pm = p_pm(blockIdx.z * probe.shape.z + m, idy, idx) / float(probe.shape.x * probe.shape.y);
	    complex wf = wavefront(blockIdx.z * probe.shape.z + m, idy, idx);

            wf = (pm + wf) * objbeta + obj * cprobe * (1 - 2 * objbeta);
	    wavefront(blockIdx.z * probe.shape.z + m, idy, idx) = wf;

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
            const GArray<complex> phistack, const GArray<Position> rois,
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
		    objacc += cprobe.conj() * phistack(p * probe.shape.z + m, pidy, pidx);
                    objdiv += cprobe.abs2();
                }
        }

        object_acc(objidy, objidx) += objacc - obj * objdiv;
        object_div(objidy, objidx) += objdiv;
    }
}

RAAR *CreateRAAR(float *difpads, const dim3 &difshape, complex *probe, const dim3 &probeshape, complex *object,
        const dim3 &objshape, Position *rois, int numrois, int batchsize, float *rfact, float *llk, float* mse,
        const std::vector<int> &gpus, float *objsupp, float *probesupp, int numobjsupp,
        float wavelength_m, float pixelsize_m, float distance_m, float detector_distance_m,
        int poscorr_iter,
        int obj_propagator,
        float step_obj, float step_probe,
        float reg_obj, float reg_probe, float total_flux) {
    RAAR *raar = new RAAR();

    raar->ptycho =
        CreatePtycho(difpads, difshape, probe, probeshape, object, objshape, rois, numrois, batchsize, rfact, llk, mse,
                gpus, objsupp, probesupp, numobjsupp,
                obj_propagator,
                wavelength_m, pixelsize_m, distance_m, detector_distance_m,
                poscorr_iter,
                step_obj, step_probe,
                reg_obj, reg_probe, total_flux);

    const float bytes_per_gb = \
	1024.0f*1024.0f*1024.0f;
    const size_t num_batches =
	PtychoNumBatches(*raar->ptycho);
    const size_t num_nontrivial_dp_loader_buffers =
	(num_batches >= 2) ? 2 : 1;
    const float gpu_mem_needed_for_nontrivial_dp_loader_buffer_data_in_gb =
	(num_nontrivial_dp_loader_buffers
	 * raar->ptycho->diff_pattern_shape.x
	 * raar->ptycho->diff_pattern_shape.y
	 * raar->ptycho->multibatchsize * sizeof(float) / bytes_per_gb);
    const float max_gpu_mem_needed_for_nontrivial_wavefront_data_in_gb =
	((raar->ptycho->total_num_rois + raar->ptycho->multibatchsize)
	 * raar->ptycho->probe->size * sizeof(complex) / bytes_per_gb);
    const float min_gpu_mem_needed_for_nontrivial_wavefront_data_in_gb =
	2 * raar->ptycho->gpu_mem_needed_for_fft_in_gb;
    const float safety_multiplier =
	1.15f;
    const auto approx_gpu_mem_needed_to_avoid_wavefront_streaming_in_gb =
	(safety_multiplier
	 * (gpu_mem_needed_for_nontrivial_dp_loader_buffer_data_in_gb
	    + max_gpu_mem_needed_for_nontrivial_wavefront_data_in_gb
	    + raar->ptycho->gpu_mem_needed_for_fft_in_gb));
    const auto approx_min_gpu_mem_required_in_gb =
	(safety_multiplier
	 * (gpu_mem_needed_for_nontrivial_dp_loader_buffer_data_in_gb
	    + min_gpu_mem_needed_for_nontrivial_wavefront_data_in_gb
	    + raar->ptycho->gpu_mem_needed_for_fft_in_gb));

    SetDevice(gpus, 0);

    if (raar->ptycho->approx_initial_gpu_mem_available_in_gb
	< approx_min_gpu_mem_required_in_gb) {
	sscInfo(format("Insufficient memory available on the GPUs requested: "
		       "approximately {:.2f} GB is required in total but only "
		       "{:.2f} GB is available.",
		       approx_min_gpu_mem_required_in_gb,
		       raar->ptycho->approx_initial_gpu_mem_available_in_gb));
	exit(EXIT_FAILURE);
    }

    raar->temp_wavefront_streaming_is_enabled =
	(raar->ptycho->approx_initial_gpu_mem_available_in_gb
	 < approx_gpu_mem_needed_to_avoid_wavefront_streaming_in_gb);

    MemoryType wf_memtype = MemoryType::EAllocGPU;

    if (raar->temp_wavefront_streaming_is_enabled) {
	sscInfo(format("Not enough memory to store all temporary wavefront "
		       "data (~{:.2f} GB) on the GPUs (~{:.2f} GB available): "
		       "wavefront data data will be streamed in batches. "
		       "Expect some slowdown.",
		       approx_gpu_mem_needed_to_avoid_wavefront_streaming_in_gb,
		       raar->ptycho->approx_initial_gpu_mem_available_in_gb));
        wf_memtype = MemoryType::EAllocCPU;
    }
    
    raar->temp_wavefront.reserve(num_batches);
    
    for (size_t i = 0; i < num_batches; i++) {
        const size_t batch_size = raar->ptycho->positions[i]->arrays[0]->sizez;
        auto *newphistack =
            new cMImage(raar->ptycho->probe->sizex, raar->ptycho->probe->sizey,
                   batch_size * raar->ptycho->probe->sizez, true,
                   raar->ptycho->gpus, wf_memtype);
        if (newphistack->bHasAllocCPU()) newphistack->RegisterHost();
        newphistack->SetToZero();
        raar->temp_wavefront.push_back(newphistack);
    }
    
    return raar;
}

void DestroyRAAR(RAAR *&raar) {
    for (auto *phi : raar->temp_wavefront) {
        if (phi->bHasAllocCPU()) phi->UnregisterHost();
        delete phi;
    }
    DestroyPtycho(raar->ptycho);
    raar = nullptr;
}

void RAARProjectProbe(RAAR& raar, int section, cMImage& wavefront_buffer) {
    const size_t num_batches = PtychoNumBatches(*raar.ptycho);
    
    const auto& Phi =
	(raar.temp_wavefront_streaming_is_enabled)
	? wavefront_buffer
	: *raar.temp_wavefront[section];
    
    ProjectPhiToProbe(*(raar.ptycho), section, Phi, false, raar.isGrad);
}

/**
 * Applies KRAAR_ObjPs and updates the object estimate.
 * */
void RAARApplyObjectUpdate(RAAR &raar,
			   cImage &velocity,
			   float stepsize,
			   float momentum,
			   float epsilon,
			   cMImage& wavefront_buffer) {
    raar.ptycho->object_num->SetGPUToZero();
    raar.ptycho->object_div->SetGPUToZero();

    dim3 blk = raar.ptycho->object->ShapeBlock();
    dim3 thr = raar.ptycho->object->ShapeThread();

    const dim3 probeshape = raar.ptycho->probe->Shape();
    for (size_t section = 0; section < raar.ptycho->positions.size(); section++) {
        const size_t cur_batch_zsize = raar.ptycho->positions[section]->sizez;

        for (size_t g = 0; g < raar.ptycho->gpus.size(); g++) {
            const int gpu_batch_size = raar.ptycho->positions[section]->arrays[g]->sizez;
            if (gpu_batch_size > 0) {
                SetDevice(raar.ptycho->gpus, g);

		if (raar.temp_wavefront_streaming_is_enabled) {
		    wavefront_buffer.arrays[g]->Resize(probeshape.x,
						       probeshape.y,
						       gpu_batch_size * probeshape.z);
		    raar.temp_wavefront[section]->arrays[g]->CopyTo(*wavefront_buffer.arrays[g]);
		}
		
		const auto& phistack =
		    (raar.temp_wavefront_streaming_is_enabled)
		    ? wavefront_buffer
		    : *raar.temp_wavefront[section];

		KRAAR_ObjPs<<<blk, thr>>>(raar.ptycho->object->arrays[g][0],
					  raar.ptycho->probe->arrays[g][0],
					  phistack.arrays[g][0],
					  raar.ptycho->positions[section]->arrays[g][0],
					  raar.ptycho->object_num->arrays[g][0],
					  raar.ptycho->object_div->arrays[g][0]);

		if (raar.temp_wavefront_streaming_is_enabled)
		    raar.temp_wavefront[section]->arrays[g]->CopyFrom(*wavefront_buffer.arrays[g]);
            }
        }
    }

    raar.ptycho->object->WeightedLerpSync(*raar.ptycho->object_num, *raar.ptycho->object_div, stepsize, momentum, velocity, epsilon);
}

/**
 * Projects phistack to the probe subspace and calls Super::ApplyProbeUpdate
 * */
void RAARApplyProbeUpdate(RAAR& raar,
			  cImage &velocity,
			  float stepsize,
			  float momentum,
			  float epsilon,
			  cMImage& wavefront_buffer) {
    raar.ptycho->probe_num->SetGPUToZero();
    raar.ptycho->probe_div->SetGPUToZero();

    const int ngpus = PtychoNumGpus(*raar.ptycho);
    const dim3 probeshape = raar.ptycho->probe->Shape();
    const size_t num_batches = PtychoNumBatches(*raar.ptycho);
    for (size_t d = 0; d < num_batches; d++) {
	if (raar.temp_wavefront_streaming_is_enabled) {
	    for (int g = 0; g < ngpus; ++g) {
		SetDevice(raar.ptycho->gpus, g);
		const size_t gpu_batch_zsize = raar.ptycho->positions[d]->arrays[g]->sizez;
		wavefront_buffer.arrays[g]->Resize(probeshape.x,
						   probeshape.y,
						   gpu_batch_zsize * probeshape.z);
		raar.temp_wavefront[d]->arrays[g]->CopyTo(*wavefront_buffer.arrays[g]);
	    }
	}
	RAARProjectProbe(raar, d, wavefront_buffer);
	if (raar.temp_wavefront_streaming_is_enabled) {
	    for (int g = 0; g < ngpus; ++g) {
		SetDevice(raar.ptycho->gpus, g);
		raar.temp_wavefront[d]->CopyFrom(wavefront_buffer);
	    }
	}
    }

    ApplyProbeUpdate(*raar.ptycho, velocity, stepsize, momentum, epsilon);
}

void init_wavefront(RAAR& raar) {
    for (MImage<complex> *wavefront : raar.temp_wavefront) {
        if (wavefront != nullptr) {
            wavefront->SetToZero();
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
    const size_t num_batches = PtychoNumBatches(*raar.ptycho);

    auto wavefront_buffer =
	(raar.temp_wavefront_streaming_is_enabled)
	? cMImage(raar.ptycho->probe->sizex,
		  raar.ptycho->probe->sizey,
		  raar.ptycho->singlebatchsize * raar.ptycho->probe->sizez,
		  true,
		  raar.ptycho->gpus,
		  MemoryType::EAllocGPU)
	: cMImage(1, 1, 1, true, raar.ptycho->gpus, MemoryType::EAllocGPU);

    auto* batch_loader = CreateDifPadBatchLoader(raar.ptycho);
    FetchNextBatchAsync(batch_loader);
    auto* cur_difpad = CurrentBatch(batch_loader);

    for (size_t iter = 0; iter < iterations; iter++) {
        raar.ptycho->error_rfactor->SetGPUToZero();
        raar.ptycho->error_llk->SetGPUToZero();
        raar.ptycho->error_mse->SetGPUToZero();

        raar.ptycho->object_num->SetGPUToZero();
        raar.ptycho->object_div->SetGPUToZero();

        for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
	    if (num_batches >= 2) {
		if ((iter != 0) || (batch_idx != 0)) {
		    cur_difpad = CurrentBatch(batch_loader);
		}
                FetchNextBatchAsync(batch_loader);
	    }

            const size_t ngpus = PtychoNumGpus(*raar.ptycho);
            for (size_t gpu_idx = 0; gpu_idx < ngpus; gpu_idx++) {
                cImage* current_exit_wave = raar.ptycho->wavefront->arrays[gpu_idx];
                cImage* current_object = raar.ptycho->object->arrays[gpu_idx];
                cImage* current_probe = raar.ptycho->probe->arrays[gpu_idx];
                rImage* current_obj_div = raar.ptycho->object_div->arrays[gpu_idx];
                cImage* current_obj_acc = raar.ptycho->object_num->arrays[gpu_idx];

                const size_t cur_difpad_zsize = raar.ptycho->positions[batch_idx]->arrays[gpu_idx]->sizez;

                if (cur_difpad_zsize > 0) {
                    SetDevice(raar.ptycho->gpus, gpu_idx);

		    auto* cur_temp_wavefront =
			(raar.temp_wavefront_streaming_is_enabled)
			? wavefront_buffer.arrays[gpu_idx]
			: raar.temp_wavefront[batch_idx]->arrays[gpu_idx];

		    if (raar.temp_wavefront_streaming_is_enabled) {
			wavefront_buffer.arrays[gpu_idx]->Resize(probeshape.x, probeshape.y,
								 cur_difpad_zsize * probeshape.z);
			raar.temp_wavefront[batch_idx]->arrays[gpu_idx]->CopyTo(*wavefront_buffer.arrays[gpu_idx]);
		    }

                    dim3 blk = raar.ptycho->wavefront->ShapeBlock();
                    blk.z = cur_difpad_zsize;
                    dim3 thr = raar.ptycho->wavefront->ShapeThread();

                    Position* ptr_roi = raar.ptycho->positions[batch_idx]->Ptr(gpu_idx);

		    k_RAAR_reflect_Rspace<<<blk, thr>>>(*current_exit_wave, *current_probe, *current_object, *cur_temp_wavefront, ptr_roi, raar.beta);

                    ProjectReciprocalSpace(*raar.ptycho, cur_difpad->arrays[gpu_idx], gpu_idx, raar.isGrad); // propagate, apply measured intensity and unpropagate

		    k_RAAR_wavefront_update<<<blk, thr>>>(*current_object, *current_probe,   *current_obj_acc, *current_obj_div,  *current_exit_wave, *cur_temp_wavefront, ptr_roi, raar.beta);

                }
            }
	    if (raar.temp_wavefront_streaming_is_enabled)
		raar.temp_wavefront[batch_idx]->CopyFrom(wavefront_buffer);
        }
	sscDebug("Syncing OBJ");
        objvelocity.SetGPUToZero();
        raar.ptycho->object->WeightedLerpSync(*(raar.ptycho->object_num), *(raar.ptycho->object_div), raar.ptycho->objstep, raar.ptycho->objmomentum, objvelocity, raar.ptycho->objreg);

        sscDebug("Applying probe");
        probevelocity.SetGPUToZero();

        if (iter != 0) {
            RAARApplyProbeUpdate(raar,
				 probevelocity,
				 raar.ptycho->probestep,
				 raar.ptycho->probemomentum,
				 raar.ptycho->probereg,
				 wavefront_buffer); // updates in real space
            RAARApplyObjectUpdate(raar,
				  objvelocity,
				  raar.ptycho->objstep,
				  raar.ptycho->objmomentum,
				  raar.ptycho->objreg,
				  wavefront_buffer);
        }

        if (raar.ptycho->poscorr_iter &&  (iter + 1) % raar.ptycho->poscorr_iter == 0){
            ApplyPositionCorrection(*raar.ptycho);
        }

        raar.ptycho->cpuerror_rfactor[iter] =
	    (raar.ptycho->error_rfactor->SumGPU()
	     / raar.ptycho->total_measured_sqrt_intensity);
        raar.ptycho->cpuerror_llk[iter] =
	    (raar.ptycho->error_llk->SumGPU()
	     / raar.ptycho->total_measured_intensity);
        raar.ptycho->cpuerror_mse[iter] =
	    (raar.ptycho->error_mse->SumGPU()
	     / raar.ptycho->total_measured_sq_intensity);

        if (iter % 10 == 0) {
            sscInfo(format("iter {}/{} rfactor: {}, llk = {}, mse = {}", iter, iterations, raar.ptycho->cpuerror_rfactor[iter], raar.ptycho->cpuerror_llk[iter], raar.ptycho->cpuerror_mse[iter]));
        }

    }

    DestroyDifPadBatchLoader(batch_loader);

    auto time1 = sscTime();
    sscInfo(format("End RAAR iteration: {} ms", sscDiffTime(time0, time1)));

}

