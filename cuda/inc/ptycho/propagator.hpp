#include <cufft.h>

#include <vector>

#include "types.hpp"

/** @file */

using std::vector;

/**
 * Base propagator interface.
 * */
class Propagator {
   public:
    /**
     * Applies propagation to given pointers.
     * */
    virtual void Propagate(complex* owave, complex* iwave, dim3 shape, float distance) = 0;
    /**
     * Makes "shape" a new possible propagation plan.
     * */
    virtual void Append(const dim3& shape) = 0;
    virtual ~Propagator(){};
};

/**
 * Implements the fraunhoffer propagator (2D-FFT). Sign of "amount" implies FORWARD and REVERSE propagations.
 * */
class Fraunhoffer : public Propagator {
   public:
    std::vector<dim3> dims;          //!< vector containing all propagation shapes.
    std::vector<cufftHandle> plans;  //!< vector containing all cufftPlans for each proapagation shape.

    virtual void Propagate(complex* owave, complex* iwave, dim3 shape, float amount) override;

    Fraunhoffer(){};
    ~Fraunhoffer() override;

    virtual void Append(const dim3& shape) override;

    cImage* workarea = nullptr;  //!< Temporary storage for cufft's plans.
};

/**
 * Implements the angular spectrum method (ASM) propagator. Amount is the pixel-fresnel number f1 = pixel**2 / (distance
 * * wavelength).
 * */
class ASM : public Fraunhoffer {
    float wavelength_m, pixelsize_m;
   public:
    ASM(float wavelength_m, float pixelsize_m);
    virtual void Propagate(complex* owave, complex* iwave, dim3 shape, float amount) override;
};
