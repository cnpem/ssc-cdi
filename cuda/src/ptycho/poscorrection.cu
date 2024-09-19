#include <cstddef>

#include "ptycho.hpp"
#include <common/logger.hpp>
#include <common/types.hpp>
#include <common/utils.hpp>

extern "C"{

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

}

PosCorrection* CreatePosCorrection(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape,
                                   complex* object, const dim3& objshape, Position* rois, int numrois, int batchsize,
                                   float* rfact, const std::vector<int>& gpus, float* objsupp, float* probesupp,
                                   int numobjsupp,
                                   float wavelength_m, float pixelsize_m, float distance_m,
                                   float step_obj, float step_probe,
                                   float reg_obj, float reg_probe) {
    PosCorrection* poscorr = new PosCorrection;
    poscorr->errorcounter = new rMImage(5, 1, batchsize, true, gpus);
    poscorr->ptycho =
        CreatePOptAlgorithm(difpads, difshape, probe, probeshape,
                object, objshape, rois, numrois, batchsize, rfact,
                gpus, objsupp, probesupp, numobjsupp,
                wavelength_m, pixelsize_m, distance_m,
                step_obj, step_probe, reg_obj, reg_probe);
    return poscorr;
}

/**
* Positions are updated along with the probe.
* */
void PosCorrectionApplyProbeUpdate(cImage& velocity, float stepsize, float momentum, float epsilon);


void PosCorrectionProjectProbe(PosCorrection& poscorr, int section) {
    ProjectPhiToProbe(*poscorr.ptycho, section, *poscorr.ptycho->exitwave, true, poscorr.isGradPm);
}

void DestroyPosCorrection(PosCorrection*& poscorr) {
    delete poscorr->errorcounter;
    const size_t num_batches = poscorr->ptycho->positions.size();
    size_t global_idx = 0;
    for(int d = 0; d < num_batches; d++) {
        for(int g = 0; g<poscorr->ptycho->gpus.size(); g++) {
            for(size_t z = 0; z < poscorr->ptycho->positions[d][0][g].sizez; z++) {
                poscorr->ptycho->cpurois[global_idx].x = poscorr->ptycho->positions[d][0][g].cpuptr[z].x;
                poscorr->ptycho->cpurois[global_idx].y = poscorr->ptycho->positions[d][0][g].cpuptr[z].y;
                global_idx++;
            }
        }
    }
    DestroyPOptAlgorithm(poscorr->ptycho);
    poscorr = nullptr;
}

void PosCorrectionApplyProbeUpdate(PosCorrection& poscorr, cImage& velocity,
        float stepsize, float momentum, float epsilon) {
    POptAlgorithm& ptycho = *poscorr.ptycho;

    ApplyProbeUpdate(ptycho, velocity, stepsize, momentum, epsilon);

    float const offx[] = {0,1,-1,0,0};
    float const offy[] = {0,0,0,1,-1};

    const size_t batchsize = ptycho.positions[0]->arrays[0]->sizez;

    const dim3 difpadshape = ptycho.difpadshape;

    rMImage cur_difpad(difpadshape.x, difpadshape.y, ptycho.multibatchsize,
            false, ptycho.gpus, MemoryType::EAllocGPU);

    const size_t num_batches = ptycho_num_batches(ptycho);
    for(int d = 0; d<num_batches; d++) {
        const size_t difpad_batch_zsize = ptycho_cur_batch_zsize(ptycho, d);
        const size_t difpad_idx = d * ptycho_batch_size(ptycho);

        cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
        cur_difpad.LoadToGPU(ptycho.cpudifpads + difpad_idx * difpadshape.x * difpadshape.y);

        poscorr.errorcounter->SetGPUToZero();
        ptycho.positions[d]->LoadFromGPU();

        const size_t ngpus = ptycho_num_gpus(ptycho);
        for(int k = 0; k<5; k++)
            for(int g = 0; g < ngpus; g++) {
                const size_t difpadsizez = ptycho.positions[d][0][g].sizez;
                if(difpadsizez > 0) {
                    SetDevice(ptycho.gpus, g);
                    dim3 blk = ptycho.exitwave->ShapeBlock(); blk.z = difpadsizez;
                    dim3 thr = ptycho.exitwave->ShapeThread();

                    Image<Position>& ptr_roi = *ptycho.positions[d]->arrays[g];
                    KSideExitwave<<<blk,thr>>>(*ptycho.exitwave->arrays[g],
                            *ptycho.probe->arrays[g],
                            *ptycho.object->arrays[g],
                            ptr_roi, offx[k], offy[k]);
                    ptycho.propagator[g]->Propagate(ptycho.exitwave->arrays[g]->gpuptr,
                            ptycho.exitwave->arrays[g]->gpuptr,
                            ptycho.exitwave->arrays[g]->Shape(), 1);

                    KComputeError<<<blk,thr>>>(
                            poscorr.errorcounter->arrays[g]->gpuptr + batchsize*k,
                            *ptycho.exitwave->arrays[g], *cur_difpad.arrays[g],
                            nullptr,
                            ptycho.probe->sizez);
            }
        }

        poscorr.errorcounter->LoadFromGPU();
        poscorr.errorcounter->SyncDevices();

        for(int g = 0; g<ptycho.gpus.size(); g++) {
            const size_t batch_size = ptycho_cur_batch_gpu_zsize(ptycho, d, g);
            for(size_t z = 0; z < batch_size; z++) {
                float* error = poscorr.errorcounter->arrays[g]->cpuptr + z;
                float minerror = 1E35f;
                int minidx = 0;

                for(int k = 0; k<5; k++) if(error[batchsize*k] < minerror) {
                    minerror = error[batchsize*k];
                    minidx = k;
                }
                ptycho.positions[d][0][g].cpuptr[z].x = fminf(fmaxf(ptycho.positions[d][0][g].cpuptr[z].x+offx[minidx],1.1f),
                        ptycho.object->sizex - ptycho.probe->sizex-3);
                ptycho.positions[d][0][g].cpuptr[z].y = fminf(fmaxf(ptycho.positions[d][0][g].cpuptr[z].y+offy[minidx],1.1f),
                        ptycho.object->sizey - ptycho.probe->sizey-3);
            }
        }

        ptycho.positions[d]->LoadToGPU();
    }
    SyncDevices(ptycho.gpus);
}


void PosCorrectionRun(PosCorrection& poscorr, int iterations) {
  ssc_debug("Starting PosCorrectionRun.");

  POptAlgorithm& ptycho = *poscorr.ptycho;

  auto time0 = ssc_time();

  ptycho.object->Set(0);
  cImage objvelocity(ptycho.object->Shape());
  cImage probevelocity(ptycho.probe->Shape());
  objvelocity.SetGPUToZero();
  probevelocity.SetGPUToZero();


  const dim3 difpadshape = ptycho.difpadshape;

  for (int iter = 0; iter < iterations; iter++) {
    ssc_debug(format("Start PosCorr iteration: {}", iter));

    // std::cout << iter << std::endl;
    const bool bIterProbe = (ptycho.probemomentum >= 0);  // & (iter > iterations/20);
    ptycho.rfactors->SetGPUToZero();
    ptycho.object_acc->SetGPUToZero();
    ptycho.object_div->SetGPUToZero();
    ptycho.probe_acc->SetGPUToZero();
    ptycho.probe_div->SetGPUToZero();

    if (iter < 2) {
      objvelocity.SetGPUToZero();
      probevelocity.SetGPUToZero();
    }

    rMImage cur_difpad(difpadshape.x, difpadshape.y, ptycho.multibatchsize,
            false, ptycho.gpus, MemoryType::EAllocGPU);

    const size_t num_batches = ptycho.positions.size();
    for (int d = 0; d < num_batches; d++) {
      const unsigned int difpad_batch_zsize = ptycho.positions[d]->sizez;
      const size_t difpad_idx = d * ptycho.multibatchsize;

      cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
      cur_difpad.LoadToGPU(ptycho.cpudifpads + difpad_idx * difpadshape.x * difpadshape.y);

      for (int g = 0; g < ptycho.gpus.size(); g++) {
          const size_t difpadsizez = ptycho_cur_batch_gpu_zsize(ptycho, d, g);
        if (difpadsizez > 0) {
          SetDevice(ptycho.gpus, g);

          dim3 blk = ptycho.exitwave->ShapeBlock();
          blk.z = difpadsizez;
          dim3 thr = ptycho.exitwave->ShapeThread();

          Image<Position>& ptr_roi = *ptycho.positions[d]->arrays[g];

          KGLExitwave<<<blk, thr>>>(*ptycho.exitwave->arrays[g],
                  *ptycho.probe->arrays[g],
                  *ptycho.object->arrays[g], ptr_roi);

          project_reciprocal_space(ptycho, cur_difpad.arrays[g],
                  g, poscorr.isGradPm);

          KGLPs<<<blk, thr>>>(*ptycho.probe->arrays[g],
                  *ptycho.object_acc->arrays[g],
                  *ptycho.object_div->arrays[g],
                  *ptycho.exitwave->arrays[g], ptr_roi);
        }
      }
      if (bIterProbe) PosCorrectionProjectProbe(poscorr, d);
    }

    ssc_debug("Syncing OBJ and setting RF");
    if (ptycho.objmomentum >= 0)
        ptycho.object->WeightedLerpSync(
                *ptycho.object_acc, *ptycho.object_div,
                ptycho.objstep, ptycho.objmomentum,
                objvelocity, ptycho.objreg);

    if (ptycho.objectsupport != nullptr)
      for (int g = 0; g < ptycho.gpus.size(); g++) {
        SetDevice(ptycho.gpus, g);
        ApplySupport(*ptycho.object->arrays[g],
                *ptycho.objectsupport->arrays[g],
                ptycho.SupportSizes);
      }

    PosCorrectionApplyProbeUpdate(poscorr, probevelocity, ptycho.probestep, ptycho.probemomentum, ptycho.probereg);

    ptycho.cpurfact[iter] = sqrtf(ptycho.rfactors->SumCPU());

    if (iter % 10 == 0) {
        ssc_info(format("iter {}/{} error: {}",
                    iter, iterations, ptycho.cpurfact[iter]));
    }

  }

  auto time1 = ssc_time();
  ssc_info(format("End GL: {} ms", ssc_diff_time(time0, time1)));
}

