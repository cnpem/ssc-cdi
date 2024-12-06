#include <cstddef>
#include <driver_types.h>
#include <vector>

#include <common/complex.hpp>
#include <common/types.hpp>


#include <propagator.hpp>

/** @file */

inline __device__ float sq(float x) { return x * x; }

/**
 * Struct containing all information about a scattering pattern.
 * */
struct Position {
    float x, y;     //!< x,y positions in pixels.
};

typedef MImage<Position> PositionArray;

/**
 * Base class for ptychography algorithms. Holds most of the relevant information and data.
 * */
struct Ptycho {
    std::vector<int> gpus;

    std::vector<PositionArray*> positions;    //!< List of probe positions with exact same length as their corresponding scattering patterns.
    cMImage* object = nullptr;      //!< Current object estimate.
    cMImage* probe = nullptr;       //!< Current probe estimate.
    cMImage* wavefront = nullptr;    //!< Temporary exitwave estimates to be amplitude projected.

    rMImage* error = nullptr;    //!< GPU Buffer for the error metric.
    rMImage* error_llk = nullptr; //!< GPU buffer for the llk error metric.
    rMImage* error_mse = nullptr; //!< GPU buffer for the llk error metric.

    rMImage* errorcounter = nullptr;

    rMImage* objectsupport = nullptr;  //!< List of object supports to be applied.
    rMImage* probesupport = nullptr;   //!< Support for the probe with the same shape as the probe itself.
    std::vector<float> SupportSizes;   //!< Normalization vector for the object support.

    Propagator* propagator[16];             //!< List of propagators to be used in the amplitude projection.
    Propagator* probepropagator = nullptr;  //!< Propagator to be used before applying a support to the probe.

    dim3 diff_pattern_shape;   //!< Shape of all scattering intensities.

    float pixelsize_m, wavelength_m, distance_m;

    float* cpu_diff_pattern = nullptr;   //!< Copy of difpads' memory location passed to the algorithm.
    complex* cpuobject = nullptr;  //!< Copy of the object's memory location passed to the algorithm.
    complex* cpuprobe = nullptr;   //!< Copy of the probe's memory location passed to the algorithm.
    Position* cpupositions = nullptr;        //!< Copy of the rois' memory location passed to the algorithm.

    float* cpuerror = nullptr;     //!< Copy of the output error metric's memory location passed to the algorithm.
    float* cpuerror_llk = nullptr;     //!< Copy of the outut error metric's memory location passed to the algorithm.
    float* cpuerror_mse = nullptr;     //!< Copy of the outut error metric's memory location passed to the algorithm.

    std::vector<int> roibatch_offset;
    size_t singlebatchsize;
    size_t multibatchsize;

    int total_num_rois;
    int poscorr_iter = 0;

    rMImage* object_div;  //!< Denominator in the object augmented projector / LS-gradient preconditioner.
    cMImage* object_num;  //!< Accumulator in the object augmented projector / LS-gradient preconditioner.

    rMImage* probe_div;  //!< Denominator in the probe augmented projector / LS-gradient preconditioner.
    cMImage* probe_num;  //!< Accumulator in the probe augmented projector / LS-gradient preconditioner.


    float objmomentum = 0.95f;
    float probemomentum = 0.9f;

    float objstep = 1.0f;
    float probestep = 1.0f;

    float probereg = 0.01f;
    float objreg = 0.01f;
};

inline size_t PtychoNumBatches(Ptycho& ptycho) {
    return ptycho.positions.size();
}

inline size_t PtychoNumGpus(Ptycho& ptycho) {
    return ptycho.gpus.size();
}

inline int ptycho_num_modes(Ptycho& ptycho) {
    return ptycho.probe->sizez;
}

/**
 * Number of slices for current batch.
 **/
inline size_t PtychoCurBatchZsize(Ptycho& ptycho, size_t batch_idx) {
    return ptycho.positions[batch_idx]->sizez;
}

inline size_t PtychoBatchSize(Ptycho& ptycho) {
    return ptycho.multibatchsize;
}

/**
 * Number of slices on gpu for batch.
 **/
inline size_t PtychoCurBatchGpuZsize(Ptycho& ptycho, size_t batch_idx, size_t gpu_idx) {
    return (*ptycho.positions[batch_idx])[gpu_idx].sizez;
}

/**
 * Probe real space project from a given section of the list.
 * */
void ProjectProbe(Ptycho& ptycho, int section);
/**
 * Fourier project exitwaves from a given section of the list.
 * */
void ProjectReciprocalSpace(Ptycho& ptycho, 
                            rImage* difpads, 
                            int g, 
                            bool isGrad, 
                            cudaStream_t stream = 0);


void ProjectReciprocalSpace(Ptycho &pt, 
                            rImage* difpad, 
                            cImage* wavefront, 
                            int g, 
                            bool isGrad, 
                            cudaStream_t stream = 0);

void DestroyPtycho(Ptycho*& ptycho);

Ptycho* CreatePtycho(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape,
                     complex* object, const dim3& objshape, Position* rois, int numrois, int batchsize,
                     float* rfact, float* llk, float* mse, const std::vector<int>& gpus, float* objsupp, float* probesupp,
                     int numobjsupp,
                     float wavelength_m, float pixelsize_m, float distance_m,
                     int poscorr_iter,
                     float step_obj, float step_probe,
                     float reg_obj, float reg_probe);


template <typename dtype>
void ProjectPhiToProbe(Ptycho& ptycho, int section, const MImage<dtype>& Phi, bool bNormalizeFFT, bool isGrad,
        cudaStream_t st = 0);
void ApplyProbeSupport(Ptycho& pt);
void ApplyProbeUpdate(Ptycho& ptycho, cImage& velocity, float stepsize, float momentum, float epsilon);
void ApplySupport(cImage& img, rImage& support, std::vector<float>& SupportSizes);
void ApplyPositionCorrection(Ptycho& ptycho);


/**
 * Implemention of Luke's RAAR algorithm for ptychography: x = (1-beta)Ps(x) + beta/2 (1 + RmRs)(x) with augmented
 * projectors.
 * */
struct RAAR {
    Ptycho* ptycho = nullptr;
    std::vector<hcMImage*> temp_wavefront;
    const bool isGrad = false;
    float beta = 0.9f;
};

/**
 * Constructor: Allocates phistack.
 * */
RAAR* CreateRAAR(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape, complex* object,
                 const dim3& objshape, Position* rois, int numrois, int batchsize, float* rfact, float* llk, float* mse,
                 const std::vector<int>& gpus, float* objsupp, float* probesupp, int numobjsupp,
                 float wavelength_m, float pixelsize_m, float distance_m,
                 int poscorr_iter,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe);

void DestroyRAAR(RAAR*& raar);

void RAARRun(RAAR& raar, int iter);

void RAARProjectProbe(RAAR& raar, int section, cudaStream_t st = 0);

/**
 * Projects phistack to object subspace and updates the object estimate.
 * */
void RAARApplyObjectUpdate(RAAR& raar, cImage& velocity,
        float stepsize, float momentum, float epsilon, hcMImage& cur_temp_wavefront);
/**
 * Projects phistack to the probe subspace and calls Super::ApplyProbeUpdate
 * */
void RAARApplyProbeUpdate(RAAR& raar, cImage& velocity,
        float stepsize, float momentum, float epsilon, hcMImage& cur_temp_wavefront);

/**
 * Alternated projections with momentum.
 * */
struct AP {
    Ptycho* ptycho;
    const bool isGrad = true;
};

void APRun(AP& glim, int iter);

AP* CreateAP(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape, complex* object,
                 const dim3& objshape, Position* rois, int numrois, int batchsize, float* rfact, float* llk, float* mse,
                 const std::vector<int>& gpus, float* objsupp, float* probesupp, int numobjsupp,
                 float wavelength_m, float pixelsize_m, float distance_m,
                 int poscorr_iter,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe);

void DestroyAP(AP*& glim);

void APProjectProbe(AP& glim, int section);

struct Pie {
    Ptycho* ptycho;
    const bool isGrad = false;
};

Pie* CreatePie(float* difpads, const dim3& difshape,
        complex* probe, const dim3& probeshape,
        complex* object, const dim3& objshape,
        Position* rois, int numrois,
        int batchsize,
        float* rfact, float* llk, float* mse,
        const std::vector<int>& gpus,
        float* objsupp, float* probesupp, int numobjsupp,
        float wavelength_m, float pixelsize_m, float distance_m,
        int poscorr_iter,
        float step_object, float step_probe,
        float reg_obj, float reg_probe);


void DestroyPie(Pie*& pie);

void PieRun(Pie& pie, int iterations);
