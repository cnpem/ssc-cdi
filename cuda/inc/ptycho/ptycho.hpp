#include <cstddef>
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
    cMImage* exitwave = nullptr;    //!< Temporary exitwave estimates to be amplitude projected.
    rMImage* rfactors = nullptr;    //!< GPU Buffer for the error metric.
    rMImage* errorcounter = nullptr;

    rMImage* objectsupport = nullptr;  //!< List of object supports to be applied.
    rMImage* probesupport = nullptr;   //!< Support for the probe with the same shape as the probe itself.
    std::vector<float> SupportSizes;   //!< Normalization vector for the object support.

    bool bCenterProbe = true;    //!< Currently unused.

    float I0;  //!< Initial probe norm ||P||^2

    Propagator* propagator[16];             //!< List of propagators to be used in the amplitude projection.
    Propagator* probepropagator = nullptr;  //!< Propagator to be used before applying a support to the probe.

    dim3 difpadshape;   //!< Shape of all scattering intensities.
    float probef1 = 1;  //!< Fresnel number to be applied by probepropagator.

    float pixelsize_m, wavelength_m;

    float* cpudifpads = nullptr;   //!< Copy of difpads' memory location passed to the algorithm.
    complex* cpuobject = nullptr;  //!< Copy of the object's memory location passed to the algorithm.
    complex* cpuprobe = nullptr;   //!< Copy of the probe's memory location passed to the algorithm.
    Position* cpurois = nullptr;        //!< Copy of the rois' memory location passed to the algorithm.
    float* cpurfact = nullptr;     //!< Copy of the output error metric's memory location passed to the algorithm.

    std::vector<int> roibatch_offset;
    size_t singlebatchsize;
    size_t multibatchsize;

    int total_num_rois;
    int poscorr_iter = 0;

    rMImage* object_div;  //!< Denominator in the object augmented projector / LS-gradient preconditioner.
    cMImage* object_acc;  //!< Accumulator in the object augmented projector / LS-gradient preconditioner.

    rMImage* probe_div;  //!< Denominator in the probe augmented projector / LS-gradient preconditioner.
    cMImage* probe_acc;  //!< Accumulator in the probe augmented projector / LS-gradient preconditioner.


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
void ProjectReciprocalSpace(Ptycho& ptycho, rImage* difpads, int g, bool isGradPm);

void DestroyPtycho(Ptycho*& ptycho);

Ptycho* CreatePtycho(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape,
                                   complex* object, const dim3& objshape, Position* rois, int numrois, int batchsize,
                                   float* rfact, const std::vector<int>& gpus, float* objsupp, float* probesupp,
                                   int numobjsupp,
                                   float wavelength_m, float pixelsize_m, float distance_m,
                                   int poscorr_iter,
                                   float step_obj, float step_probe,
                                   float reg_obj, float reg_probe);


template <typename dtype>
void ProjectPhiToProbe(Ptycho& ptycho, int section, const MImage<dtype>& Phi, bool bNormalizeFFT, bool isGradPm);
void ApplyProbeUpdate(Ptycho& ptycho, cImage& velocity, float stepsize, float momentum, float epsilon);
void ApplySupport(cImage& img, rImage& support, std::vector<float>& SupportSizes);
void ApplyPositionCorrection(Ptycho& ptycho);


/**
 * Implemention of Luke's RAAR algorithm for ptychography: x = (1-beta)Ps(x) + beta/2 (1 + RmRs)(x) with augmented
 * projectors.
 * */
struct RAAR {
    Ptycho* ptycho = nullptr;
    std::vector<hcMImage*> phistack;  //!< Stack of current exitwave estimates. can become very huge
    const bool isGradPm = false;
    float beta = 0.9f;
};

/**
 * Constructor: Allocates phistack.
 * */
RAAR* CreateRAAR(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape, complex* object,
                 const dim3& objshape, Position* rois, int numrois, int batchsize, float* rfact,
                 const std::vector<int>& gpus, float* objsupp, float* probesupp, int numobjsupp,
                 float wavelength_m, float pixelsize_m, float distance_m,
                 int poscorr_iter,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe);

void DestroyRAAR(RAAR*& raar);

void RAARRun(RAAR& raar, int iter);

void RAARProjectProbe(RAAR& raar, int section);

/**
 * Projects phistack to object subspace and updates the object estimate.
 * */
void RAARApplyObjectUpdate(RAAR& raar, cImage& velocity, float stepsize, float momentum, float epsilon);
/**
 * Projects phistack to the probe subspace and calls Super::ApplyProbeUpdate
 * */
void RAARApplyProbeUpdate(RAAR& raar, cImage& velocity, float stepsize, float momentum, float epsilon);

extern "C" {
__global__ void KGLExitwave(GArray<complex> exitwave, const GArray<complex> probe, const GArray<complex> object,
                            const GArray<Position> rois);

__global__ void KGLPs(const GArray<complex> probe, GArray<complex> object_acc, GArray<float> object_div,
                      const GArray<complex> p_pm, const GArray<Position> rois);

__global__ void KProjectReciprocalSpace(GArray<complex> exitwave, const GArray<float> difpads, float* rfactors, size_t upsample,
                    size_t nummodes, bool bIsGrad);
}

/**
 * Alternated projections with momentum.
 * */
struct AP {
    Ptycho* ptycho;
    const bool isGradPm = true;
};

void APRun(AP& glim, int iter);

AP* CreateAP(float* difpads, const dim3& difshape, complex* probe, const dim3& probeshape, complex* object,
                 const dim3& objshape, Position* rois, int numrois, int batchsize, float* rfact,
                 const std::vector<int>& gpus, float* objsupp, float* probesupp, int numobjsupp,
                 float wavelength_m, float pixelsize_m, float distance_m,
                 int poscorr_iter,
                 float step_obj, float step_probe,
                 float reg_obj, float reg_probe);

void DestroyAP(AP*& glim);

void APProjectProbe(AP& glim, int section);

struct Pie {
    Ptycho* ptycho;
    const bool isGradPm = false;
};

Pie* CreatePie(float* difpads, const dim3& difshape,
        complex* probe, const dim3& probeshape,
        complex* object, const dim3& objshape,
        Position* rois, int numrois,
        int batchsize,
        float* rfact,
        const std::vector<int>& gpus,
        float* objsupp, float* probesupp, int numobjsupp,
        float wavelength_m, float pixelsize_m, float distance_m,
        int poscorr_iter,
        float step_object, float step_probe,
        float reg_obj, float reg_probe);


void DestroyPie(Pie*& pie);

void PieRun(Pie& pie, int iterations);
