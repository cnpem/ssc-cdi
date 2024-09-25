#include "logger.hpp"
#include <cufft.h>
#include <driver_types.h>

#include <propagator.hpp>

extern "C" {
void Fraunhoffer::Append(const dim3& dim, cudaStream_t stream) {
    std::vector<dim3>& dims = st_dims[stream];
    std::vector<cufftHandle>& plans = st_plans[stream];

    sscDebug(format("Creating new plan: {} {} {}", dim.x, dim.y, dim.z));
    if (workarea == nullptr || dim.x * dim.y * dim.z > workarea->size) {
        sscDebug(format("Reallocating plan memory to size: {} {} {}", dim.x,
                        dim.y, dim.z));
        if (workarea) delete workarea;
        workarea = new cImage(dim.x, dim.y, dim.z);

        for (auto plan : plans) cufftSetWorkArea(plan, workarea->gpuptr);
    }

    int i = dims.size();
    dims.push_back(dim);

    int n[] = {(int)dim.x, (int)dim.y};

    cufftHandle newplan;

    size_t worksize;
    sscCufftCheck(cufftCreate(&newplan));
    sscCufftCheck(cufftSetAutoAllocation(newplan, 0));
    sscCufftCheck(cufftMakePlanMany(newplan, 2, n, nullptr, 0, 0, nullptr, 0, 0,
                                    CUFFT_C2C, (int)dim.z, &worksize));
    sscCufftCheck(cufftSetWorkArea(newplan, workarea->gpuptr));
    sscCufftCheck(cufftSetStream(newplan, stream));

    plans.push_back(newplan);

    sscAssert(worksize <= 8 * workarea->size, "CuFFT being hungry!");
    sscDebug("Done.");
}

bool dim3EQ(const dim3& d1, const dim3& d2) {
    return d1.x == d2.x && d1.y == d2.y && d1.z == d2.z;
}

void Fraunhoffer::Propagate(complex* owave, complex* iwave, dim3 shape,
                            float amount, cudaStream_t stream) {
    std::vector<dim3>& dims = st_dims[stream];
    std::vector<cufftHandle>& plans = st_plans[stream];

    bool bPlanExists = false;
    auto dir = (amount > 0) ? CUFFT_FORWARD : CUFFT_INVERSE;

    cufftHandle plan;

    for (int i = 0; i < dims.size(); i++) {
        if (dim3EQ(shape, dims[i])) {
                bPlanExists = true;
                plan = plans[i];
        }
    }
    if (!bPlanExists) {
        Append(shape);
        plan = plans[plans.size() - 1];
    }
    sscCufftCheck(cufftExecC2C(plan, iwave, owave, dir));
}

Fraunhoffer::~Fraunhoffer() {
    sscDebug("Deleting propagator.");
    for (auto& [key, plans] : st_plans) {
        for (auto plan : plans) {
            if (plan) cufftDestroy(plan);
        }
    }
    if (workarea) delete workarea;
}

__global__ void KApplyASM(complex* wave, float fresnel_number, dim3 shape) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idy = blockIdx.y;
    size_t idz = blockIdx.z;

    if (idx >= shape.x) return;

    float xx =
        float(int(idx + shape.x / 2) % int(shape.x)) / float(shape.x) - 0.5f;
    float yy =
        float(int(idy + shape.y / 2) % int(shape.y)) / float(shape.y) - 0.5f;

    wave[idz * shape.x * shape.y + idy * shape.x + idx] *=
        complex::exp1j(-float(M_PI) / fresnel_number * (xx * xx + yy * yy)) /
        float(shape.x * shape.y);
}

ASM::ASM(float wavelength, float pixelsize_m)
    : wavelength_m(wavelength_m), pixelsize_m(pixelsize_m) {}

void ASM::Propagate(complex* owave, complex* iwave, dim3 shape,
                    float distance_m, cudaStream_t stream) {
    const float fresnel_number =
        (pixelsize_m * pixelsize_m) / wavelength_m / distance_m;
    Fraunhoffer::Propagate(owave, iwave, shape, 1, stream);
    const dim3 blk((shape.x + 127) / 128, shape.y, shape.z);
    const dim3 thr(128, 1, 1);
    KApplyASM<<<blk, thr, 0, stream>>>(owave, fresnel_number, shape);
    Fraunhoffer::Propagate(owave, owave, shape, -1, stream);
}
}
