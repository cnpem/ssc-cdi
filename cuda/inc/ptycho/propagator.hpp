#include <cufft.h>

#include <driver_types.h>
#include <unordered_map>
#include <vector>

#include "types.hpp"

/** @file */

using std::vector;

/**
 * Base propagator interface.
 * */
class Propagator {
    std::unordered_map<cudaStream_t, std::vector<dim3>> st_dims;
    std::unordered_map<cudaStream_t, std::vector<cufftHandle>> st_plans;
public:
    /**
     * Applies propagation to given pointers.
     * */
    virtual void Propagate(complex* owave, complex* iwave,
            dim3 shape, float distance, cudaStream_t st = 0) = 0;
    /**
     * Makes "shape" a new possible propagation plan.
     * */
    virtual void Append(const dim3& shape, cudaStream_t stream = 0) = 0;
    virtual ~Propagator() {};
};

/**
 * Implements the fraunhoffer propagator (2D-FFT). Sign of "amount" implies FORWARD and REVERSE propagations.
 * */
class Fraunhoffer : public Propagator {
public:
    std::unordered_map<cudaStream_t, std::vector<dim3>> st_dims;
    std::unordered_map<cudaStream_t, std::vector<cufftHandle>> st_plans;

    virtual void Propagate(complex* owave, complex* iwave,
            dim3 shape, float amount, cudaStream_t st = 0) override;

    Fraunhoffer() {};
    ~Fraunhoffer() override;

    virtual void Append(const dim3& shape, cudaStream_t stream = 0) override;

    cImage* workarea = nullptr; //!< Temporary storage for cufft's plans.
};

/**
 * Implements the angular spectrum method (ASM) propagator. Amount is the pixel-fresnel number f1 = pixel**2 / (distance
 * * wavelength).
 * */
class ASM : public Fraunhoffer {
    float wavelength_m, pixelsize_m;

public:
    ASM(float wavelength_m, float pixelsize_m);
    virtual void Propagate(complex* owave, complex* iwave,
            dim3 shape, float amount, cudaStream_t stream = 0) override;
};
