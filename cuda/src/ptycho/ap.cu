#include "engines_common.hpp"
#include <common/logger.hpp>
#include <common/utils.hpp>
#include <cstddef>

extern "C" {
__global__ void KAPExitwave(GArray<complex> exitwave, const GArray<complex> probe, const GArray<complex> object,
                            const GArray<Position> rois) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= probe.shape.x) return;

  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z;

  for (int p = 0; p < rois.shape.x; p++)
  {
    int objposx = idx + (int)rois(idz, 0, p).x;
    int objposy = idy + (int)rois(idz, 0, p).y;

    const complex& obj = object(objposy, objposx);

    for (size_t m = 0; m < probe.shape.z; m++)  // for each incoherent mode
      exitwave(m + probe.shape.z * p + rois.shape.x * probe.shape.z * idz, idy, idx) = obj * probe(m, idy, idx);
  }
}

__global__ void KAPPs(const GArray<complex> probe, GArray<complex> object_acc, GArray<float> object_div,
                      const GArray<complex> p_pm, const GArray<Position> rois) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= probe.shape.x) return;

  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z;

  for (int p = 0; p < rois.shape.x; p++)  // for each flyscan point
  {
    const int objposx = idx + (int)rois(idz, 0, p).x;
    const int objposy = idy + (int)rois(idz, 0, p).y;

    complex objacc = complex(0);
    float objdiv = 0;

    for (size_t m = 0; m < probe.shape.z; m++)  // for each incoherent mode
    {
      const complex& cprobe = probe(m, idy, idx);
      objacc += p_pm(m + probe.shape.z * p + rois.shape.x * probe.shape.z * blockIdx.z, idy, idx) * cprobe.conj();
      objdiv += cprobe.abs2();
    }

    atomicAdd(&object_acc(objposy, objposx), objacc / float(probe.shape.x * probe.shape.y));
    atomicAdd(&object_div(objposy, objposx), objdiv);
  }
}

__global__ void KApplySupport(GArray<complex> img, GArray<float> support, complex constant_value) {
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t idy = blockIdx.y;
  const size_t idz = blockIdx.z;

  if (idx >= img.shape.x) return;

  img(idz, idy, idx) = img(idz, idy, idx) * (1.0f - support(idy, idx)) + constant_value * support(idy, idx);
}
}

void ApplySupport(cImage& img, rImage& support, std::vector<float>& SupportSizes) {
  for (size_t z = 0; z < support.sizez && z < SupportSizes.size(); z++) {
    complex dotproduct = img.dot(support) / SupportSizes[z];
    KApplySupport<<<img.ShapeBlock(), img.ShapeThread()>>>(img, GArray<float>(support, dim3(0, 0, z)), dotproduct);
  }
}

void APRun(AP& ap, int iterations) {
  sscInfo("Starting Alternate Projections.");

  Ptycho& ptycho = *ap.ptycho;

  auto time0 = sscTime();

  ptycho.object->Set(0);
  cImage objvelocity(ptycho.object->Shape());
  cImage probevelocity(ptycho.probe->Shape());
  objvelocity.SetGPUToZero();
  probevelocity.SetGPUToZero();

  const dim3 difpadshape = ptycho.diff_pattern_shape;
  const size_t ngpus = PtychoNumGpus(ptycho);

  for (int iter = 0; iter < iterations; iter++) {

    const bool bIterProbe = (ptycho.probemomentum >= 0);  // & (iter > iterations/20);
    ptycho.error->SetGPUToZero();
    ptycho.object_num->SetGPUToZero();
    ptycho.object_div->SetGPUToZero();
    ptycho.probe_num->SetGPUToZero();
    ptycho.probe_div->SetGPUToZero();

    if (iter < 2) {
      objvelocity.SetGPUToZero();
      probevelocity.SetGPUToZero();
    }


    // TODO: improve so we can avoid reallocating arrays every iteration,
    // if we need a speedup
    rMImage cur_difpad(difpadshape.x, difpadshape.y, ptycho.multibatchsize,
          false, ptycho.gpus, MemoryType::EAllocGPU);

    const size_t num_batches = PtychoNumBatches(ptycho);
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {

          const size_t difpad_batch_zsize = PtychoCurBatchZsize(ptycho, batch_idx);
          const size_t difpad_idx = batch_idx * PtychoBatchSize(ptycho);
          float *difpad_batch_ptr = ptycho.cpu_diff_pattern +
              difpad_idx * difpadshape.x * difpadshape.y;

          cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
          cur_difpad.LoadToGPU(difpad_batch_ptr);

          for (int g = 0; g < ngpus; g++) {
              const size_t difpadsizez = (*ptycho.positions[batch_idx])[g].sizez;
              if (difpadsizez > 0) {
                  SetDevice(ptycho.gpus, g);

                  dim3 blk = ptycho.wavefront->ShapeBlock();
                  blk.z = difpadsizez;
                  dim3 thr = ptycho.wavefront->ShapeThread();

                  Image<Position>& ptr_roi = *ptycho.positions[batch_idx]->arrays[g];
                  
                  KAPExitwave<<<blk, thr>>>(*ptycho.wavefront->arrays[g], *ptycho.probe->arrays[g], *ptycho.object->arrays[g], ptr_roi);

                  ProjectReciprocalSpace(ptycho, cur_difpad.arrays[g], g, ap.isGrad);

                  KAPPs<<<blk, thr>>>(*ptycho.probe->arrays[g],  *ptycho.object_num->arrays[g], *ptycho.object_div->arrays[g], *ptycho.wavefront->arrays[g], ptr_roi);
        }
      }
      if (bIterProbe) APProjectProbe(ap, batch_idx);
    }

    sscDebug("Syncing OBJ and setting RF");
    if (ptycho.objmomentum >= 0)
        ptycho.object->WeightedLerpSync(
                *ptycho.object_num, *ptycho.object_div,
                ptycho.objstep, ptycho.objmomentum,
                objvelocity, ptycho.objreg);

    if (ptycho.objectsupport != nullptr) {
        for (int g = 0; g < ngpus; g++) {
            SetDevice(ptycho.gpus, g);
            ApplySupport(*ptycho.object->arrays[g],
                    *ptycho.objectsupport->arrays[g],
                    ptycho.SupportSizes);
      }
    }

    ApplyProbeUpdate(ptycho, probevelocity, ptycho.probestep, ptycho.probemomentum, ptycho.probereg);

    if (ptycho.poscorr_iter &&
                (iter + 1) % ptycho.poscorr_iter == 0)
            ApplyPositionCorrection(ptycho);

    ptycho.cpuerror[iter] = sqrtf(ptycho.error->SumCPU());

    if (iter % 10 == 0) {
        sscInfo(format("iter {}/{} error = {}",
                    iter, iterations, ptycho.cpuerror[iter]));
    }
  }

  auto time1 = sscTime();
  sscInfo(format("End AP: {} ms", sscDiffTime(time0, time1)));
}


void APProjectProbe(AP& ap, int section) {
    ProjectPhiToProbe(*ap.ptycho, section, *ap.ptycho->wavefront, true, ap.isGrad);
}

AP* CreateAP(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape, complex* object,
                 const dim3& objshape, Position* rois, int numrois, int batchsize, float* rfact,
                 const std::vector<int>& gpus, float* objsupp, float* probesupp, int numobjsupp,
                 float wavelength_m, float pixelsize_m, float distance_m,
                 int poscorr_iter,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe) {
    AP* ap = new AP;
    ap->ptycho =
        CreatePtycho(difpads, difshape, probe, probeshape,
                object, objshape, rois, numrois, batchsize, rfact,
                gpus, objsupp, probesupp, numobjsupp,
                wavelength_m, pixelsize_m, distance_m,
                poscorr_iter,
                step_obj, step_probe,
                reg_obj, reg_probe);

    return ap;
}

void DestroyAP(AP*& ap) {
    DestroyPtycho(ap->ptycho);
    ap = nullptr;
}
