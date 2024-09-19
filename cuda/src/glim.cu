#include "ptycho.hpp"
#include <common/logger.hpp>
#include <common/utils.hpp>
#include <cstddef>

extern "C" {
__global__ void KGLExitwave(GArray<complex> exitwave, const GArray<complex> probe, const GArray<complex> object,
                            const GArray<Position> rois) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= probe.shape.x) return;

  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z;

  for (int p = 0; p < rois.shape.x; p++)  // for each flyscan point
  {
    int objposx = idx + (int)rois(idz, 0, p).x;
    int objposy = idy + (int)rois(idz, 0, p).y;

    const complex& obj = object(objposy, objposx);

    for (size_t m = 0; m < probe.shape.z; m++)  // for each incoherent mode
      exitwave(m + probe.shape.z * p + rois.shape.x * probe.shape.z * idz,
              idy, idx) = obj * probe(m, idy, idx);
  }
}

__global__ void KGLPs(const GArray<complex> probe, GArray<complex> object_acc, GArray<float> object_div,
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

void GLimRun(GLim& glim, int iterations) {
  ssc_info("Starting Alternate Projections.");

  POptAlgorithm& ptycho = *glim.ptycho;

  auto time0 = ssc_time();

  ptycho.object->Set(0);
  cImage objvelocity(ptycho.object->Shape());
  cImage probevelocity(ptycho.probe->Shape());
  objvelocity.SetGPUToZero();
  probevelocity.SetGPUToZero();

  const dim3 difpadshape = ptycho.difpadshape;
  const size_t ngpus = ptycho_num_gpus(ptycho);

  for (int iter = 0; iter < iterations; iter++) {

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


    // TODO: improve so we can avoid reallocating arrays every iteration,
    // if we need a speedup
    rMImage cur_difpad(difpadshape.x, difpadshape.y, ptycho.multibatchsize,
          false, ptycho.gpus, MemoryType::EAllocGPU);

    const size_t num_batches = ptycho_num_batches(ptycho);
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {

          const size_t difpad_batch_zsize = ptycho_cur_batch_zsize(ptycho, batch_idx);
          const size_t difpad_idx = batch_idx * ptycho_batch_size(ptycho);
          float *difpad_batch_ptr = ptycho.cpudifpads +
              difpad_idx * difpadshape.x * difpadshape.y;

          cur_difpad.Resize(difpadshape.x, difpadshape.y, difpad_batch_zsize);
          cur_difpad.LoadToGPU(difpad_batch_ptr);

          for (int g = 0; g < ngpus; g++) {
              const size_t difpadsizez = (*ptycho.positions[batch_idx])[g].sizez;
              if (difpadsizez > 0) {
                  SetDevice(ptycho.gpus, g);

                  dim3 blk = ptycho.exitwave->ShapeBlock();
                  blk.z = difpadsizez;
                  dim3 thr = ptycho.exitwave->ShapeThread();

                  Image2D<Position>& ptr_roi = *ptycho.positions[batch_idx]->arrays[g];
                  KGLExitwave<<<blk, thr>>>(*ptycho.exitwave->arrays[g],
                          *ptycho.probe->arrays[g],
                          *ptycho.object->arrays[g], ptr_roi);

                  project_reciprocal_space(ptycho, cur_difpad.arrays[g], g, glim.isGradPm);

                  KGLPs<<<blk, thr>>>(*ptycho.probe->arrays[g],
                          *ptycho.object_acc->arrays[g],
                          *ptycho.object_div->arrays[g],
                          *ptycho.exitwave->arrays[g], ptr_roi);
        }
      }
      if (bIterProbe) GLimProjectProbe(glim, batch_idx);
    }

    ssc_debug("Syncing OBJ and setting RF");
    if (ptycho.objmomentum >= 0)
        ptycho.object->WeightedLerpSync(
                *ptycho.object_acc, *ptycho.object_div,
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

    ptycho.cpurfact[iter] = sqrtf(ptycho.rfactors->SumCPU());

    if (iter % 10 == 0) {
        ssc_info(format("iter {}/{} error = {}",
                    iter, iterations, ptycho.cpurfact[iter]));
    }
  }

  auto time1 = ssc_time();
  ssc_info(format("End AP: {} ms", ssc_diff_time(time0, time1)));
}


void GLimProjectProbe(GLim& glim, int section) {
    ProjectPhiToProbe(*glim.ptycho, section, *glim.ptycho->exitwave, true, glim.isGradPm);
}

GLim* CreateGLim(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape, complex* object,
                 const dim3& objshape, Position* rois, int numrois, int batchsize, float* rfact,
                 const std::vector<int>& gpus, float* objsupp, float* probesupp, int numobjsupp,
                 float probef1,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe) {
    GLim* glim = new GLim;
    glim->ptycho =
        CreatePOptAlgorithm(difpads, difshape, probe, probeshape,
                object, objshape, rois, numrois, batchsize, rfact,
                gpus, objsupp, probesupp, numobjsupp, probef1,
                step_obj, step_probe, reg_obj, reg_probe);

    return glim;
}

void DestroyGLim(GLim*& glim) {
    DestroyPOptAlgorithm(glim->ptycho);
    glim = nullptr;
}
