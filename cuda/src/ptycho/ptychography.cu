#include "complex.hpp"
#include "engines_common.hpp"

int main()
{
    return 0;
}


bool accessEnabled[16][16];

extern "C" {
    void EnablePeerToPeer(const std::vector<int>& gpus) {
        for(int g : gpus) {
            cudaSetDevice(g);
            for(int g2 : gpus) {
                if(g2 != g && !accessEnabled[g][g2]) {
                    int can_acess;
                    sscCudaCheck( cudaDeviceCanAccessPeer(&can_acess, g, g2) );
                    if(!can_acess) {
                        sscWarning(format("Warning: GPU {} cant access GPU {}", g, g2));
                    }
                    else {
                        cudaError_t cerror = cudaDeviceEnablePeerAccess(g2, 0);
                        if( cerror == cudaSuccess ) {
                            accessEnabled[g][g2] = true;
                            sscDebug(format("P2P access enabled: {} <-> {}", g, g2));
                        }
                        else if(cerror == cudaErrorPeerAccessAlreadyEnabled) {
                            sscDebug(format("GPU {} already has access to GPU {}", g, g2));
                        }
                        else {
                            sscCudaCheck( cerror );
                        }
                    }
                }

            }

        }
    }

    void DisablePeerToPeer(const std::vector<int>& gpus) {
        int numgpus;
        cudaGetDeviceCount(&numgpus);
        for(int g : gpus) {
            cudaSetDevice(g);
            for(int g2 : gpus) {
                if(g2 != g && accessEnabled[g][g2]) {
                    sscCudaCheck( cudaDeviceDisablePeerAccess(g2) );
                    sscDebug(format("P2P access disabled: {} <-> {}", g, g2));
                }
            }
        }
    }
}

extern "C"
{
    void ap_call(void* cpuobj, void* cpuprobe, void* cpudif, int psizex, int osizex, int osizey, int dsizex, void* cpurois, int numrois,
            int bsize, int numiter, int ngpus, int* cpugpus, float* error_errors_rfactor, float* error_errors_llk, float* error_errors_mse, float objbeta, float probebeta, int psizez,
            float* objsupport, float* probesupport, int numobjsupport, int poscorr_iter, float step_obj, float step_probe, float reg_obj, 
            float reg_probe, float wavelength_m, float pixelsize_m, float distance_m)
    {
        sscInfo(format("Starting AP - Probe: ({},{}), Object: ({},{}), Positions: {}, Batches: {}, Iterations: {}",  psizex, psizex, 
                       osizey,osizex, numrois, bsize, numiter));

        {
            std::vector<int> gpus;
            for(int g=0; g<ngpus; g++)
                gpus.push_back(cpugpus[g]);

            AP *ap = CreateAP((float*)cpudif, dim3(dsizex,dsizex,numrois), (complex*)cpuprobe, dim3(psizex,psizex,psizez), (complex*)cpuobj, 
                              dim3(osizex, osizey), (Position*)cpurois, numrois, bsize, error_errors_rfactor, error_errors_llk, error_errors_mse, gpus, objsupport, probesupport, 
                              numobjsupport,  wavelength_m, pixelsize_m, distance_m, poscorr_iter, step_obj, step_probe, reg_obj, reg_probe);

            ap->ptycho->objmomentum = objbeta;
            ap->ptycho->probemomentum = probebeta;

            APRun(*ap, numiter);

            DestroyAP(ap);
        }
        sscInfo("End AP.");
    }

    void piecall(void* cpuobj, int osizex, int osizey,
                 void* cpuprobe, int psizex, int psizez,
                 void* cpudif, int dsizex,
                 void* cpurois, int numrois,
                 int numiter,
                 int* cpugpus, int ngpus,
                 float* error_errors_rfactor, float* error_errors_llk, float* error_errors_mse,
                 float* probesupport,
                 int poscorr_iter,
                 float step_object, float step_probe,
                 float reg_obj, float reg_probe,
                 float wavelength_m, float pixelsize_m, float distance_m) {

        sscInfo(format("Starting PIE - Probe: ({},{}), Object: ({},{}), Positions: {}, Iterations: {}",  psizex, psizex, osizey, osizex, numrois, numiter));

        std::vector<int> gpus;
        gpus.reserve(ngpus);
        for(int g=0; g<ngpus; g++)
            gpus.push_back(cpugpus[g]);

        const int batchsize = 1;
        const int numobjsupport = 0;

        float* objsupport = nullptr;

        Pie* pie = CreatePie((float*)cpudif,
                dim3(dsizex,dsizex,numrois),
                (complex*)cpuprobe, dim3(psizex,psizex,psizez),
                (complex*)cpuobj, dim3(osizex, osizey),
                (Position*)cpurois, numrois,
                batchsize, error_errors_rfactor, error_errors_llk, error_errors_mse,
                gpus, objsupport,
                probesupport, numobjsupport,
                wavelength_m, pixelsize_m, distance_m,
                poscorr_iter,
                step_object, step_probe,
                reg_obj, reg_probe);

        PieRun(*pie, numiter);

        DestroyPie(pie);
    }

    void raarcall(void* cpuobj, void* cpuprobe, void* cpudif, int psizex, int osizex, int osizey, int dsizex, void* cpurois, int numrois,
            int bsize, int numiter, int ngpus, int* cpugpus, float* error_errors_rfactor, float* error_errors_llk, float* error_errors_mse, float objbeta, float probebeta, int psizez,
            float* objsupport, float* probesupport, int numobjsupport, int poscorr_iter, float step_obj, float step_probe, float reg_obj, 
            float reg_probe, float wavelength_m, float pixelsize_m, float distance_m, float raarbeta)
    {
        sscInfo(format("Starting RAAR - Probe: ({},{}), Object: ({},{}), Positions: {}, Batches: {}, Iterations: {}",  psizex,psizex, osizey,osizex, numrois, bsize, numiter));
        {
            std::vector<int> gpus;
            for(int g=0; g<ngpus; g++)
                gpus.push_back(cpugpus[g]);

            RAAR* raar = CreateRAAR((float*)cpudif, dim3(dsizex,dsizex,numrois), (complex*)cpuprobe, dim3(psizex,psizex,psizez), 
                                    (complex*)cpuobj, dim3(osizex, osizey), (Position*)cpurois, numrois, bsize, error_errors_rfactor, error_errors_llk, error_errors_mse, gpus, 
                                    objsupport, probesupport, numobjsupport, wavelength_m, pixelsize_m, distance_m, poscorr_iter, step_obj, 
                                    step_probe, reg_obj, reg_probe);

            raar->ptycho->objmomentum = objbeta; // why is this not already inside CreateRAAR?
            raar->ptycho->probemomentum = probebeta;

            RAARRun(*raar, numiter); // perhaps objbeta should also be a  parameter here, since it works like tvmu and epsilon

            DestroyRAAR(raar);

            sscInfo("End RAAR.");
        }
    }
}


#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>


extern "C"{
    void _fhandler(int signo, siginfo_t *info, void *extra)
    {
        std::cerr << "The process received signal " << signo << " with code " << info->si_code << std::endl;
        abort();
    }

    void _setHandler(void (*handler)(int,siginfo_t *,void *))
    {
        struct sigaction action;
        action.sa_flags = SA_SIGINFO;
        action.sa_sigaction = handler;

        if (sigaction(SIGFPE, &action, NULL) == -1) {
            perror("sigfpe: sigaction");
            _exit(1);
        }
        if (sigaction(SIGSEGV, &action, NULL) == -1) {
            perror("sigsegv: sigaction");
            _exit(1);
        }
        if (sigaction(SIGILL, &action, NULL) == -1) {
            perror("sigill: sigaction");
            _exit(1);
        }
        if (sigaction(SIGBUS, &action, NULL) == -1) {
            perror("sigbus: sigaction");
            _exit(1);
        }
    }
}

#include <numeric>
struct Initializer
{
    Initializer()
    {
        memset(accessEnabled,0,256);
    };
    ~Initializer()
    {

        std::vector<int> gpus(16);
        std::iota(gpus.begin(), gpus.end(), 0);

        //DisablePeerToPeer(gpus);
        sscDebug("Disabled P2P.");
    };
};

Initializer init;
