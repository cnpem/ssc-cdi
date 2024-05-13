#include "complex.hpp"
#include "ptycho.hpp"

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
                    ssc_cuda_check( cudaDeviceCanAccessPeer(&can_acess, g, g2) );
                    if(!can_acess) {
                        ssc_warning(format("Warning: GPU {} cant access GPU {}", g, g2));
                    }
                    else {
                        cudaError_t cerror = cudaDeviceEnablePeerAccess(g2, 0);
                        if( cerror == cudaSuccess ) {
                            accessEnabled[g][g2] = true;
                            ssc_debug(format("P2P access enabled: {} <-> {}", g, g2));
                        }
                        else if(cerror == cudaErrorPeerAccessAlreadyEnabled) {
                            ssc_debug(format("GPU {} already has access to GPU {}", g, g2));
                        }
                        else {
                            ssc_cuda_check( cerror );
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
                    ssc_cuda_check( cudaDeviceDisablePeerAccess(g2) );
                    ssc_debug(format("P2P access disabled: {} <-> {}", g, g2));
                }
            }
        }
    }
}

extern "C"
{
    void glcall(void* cpuobj, void* cpuprobe, void* cpudif, int psizex, int osizex, int osizey, int dsizex, void* cpurois, int numrois,
            int bsize, int numiter, int ngpus, int* cpugpus, float* rfactors, float objbeta, float probebeta, int psizez,
            float* objsupport, float* probesupport, int numobjsupport, float* sigmask, int geometricsteps, float epsilon, float* background, float probef1)
    {
        ssc_info(format("Starting GL - p: {} o: {} r: {} b: {} n: {}",
                    psizex, osizex, numrois, bsize, numiter));
        {
            std::vector<int> gpus;
            for(int g=0; g<ngpus; g++)
                gpus.push_back(cpugpus[g]);

            GLim *gl = CreateGLim((float*)cpudif, dim3(dsizex,dsizex,numrois), (complex*)cpuprobe, dim3(psizex,psizex,psizez), (complex*)cpuobj, dim3(osizex, osizey),
                    (ROI*)cpurois, numrois, bsize, rfactors, gpus, objsupport, probesupport, numobjsupport, sigmask, geometricsteps, background, probef1, epsilon);

            gl->ptycho->objmomentum = objbeta;
            gl->ptycho->probemomentum = probebeta;

            GLimRun(*gl, numiter);

            DestroyGLim(gl);
        }
        ssc_info("End GL.");
    }

    void piecall(void* cpuobj, int osizex, int osizey,
            void* cpuprobe, int psizex, int psizez,
            void* cpudif, int dsizex,
            void* cpurois, int numrois,
            void* sigmask,
            int numiter,
            int* cpugpus, int ngpus,
            float* rfactors,
            float step_object, float step_probe,
            float reg_obj, float reg_probe) {

        ssc_info(format("Starting PIE - p: {} o: {} r: {} n: {}",
                    psizex, osizex, numrois, numiter));

        std::vector<int> gpus;
        gpus.reserve(ngpus);
        for(int g=0; g<ngpus; g++)
            gpus.push_back(cpugpus[g]);

        const int batchsize = 1;
        const int geometricsteps = 1;
        const int numobjsupport = 0;

        const float probef1 = 0.0;

        float* objsupport = nullptr;
        float* probesupport = nullptr;
        float* background = nullptr;

        Pie* pie = CreatePie((float*)cpudif,
                dim3(dsizex,dsizex,numrois),
                (complex*)cpuprobe, dim3(psizex,psizex,psizez),
                (complex*)cpuobj, dim3(osizex, osizey),
                (ROI*)cpurois, numrois,
                batchsize, rfactors,
                gpus, objsupport,
                probesupport, numobjsupport,
                (float*)sigmask, geometricsteps, background, probef1,
                step_object, step_probe,
                reg_obj, reg_probe);

        PieRun(*pie, numiter);

        DestroyPie(pie);
    }

    void raarcall(void* cpuobj, void* cpuprobe, void* cpudif, int psizex, int osizex, int osizey, int dsizex, void* cpurois, int numrois,
            int bsize, int numiter, int ngpus, int* cpugpus, float* rfactors, float objbeta, float probebeta, int psizez,
            float* objsupport, float* probesupport, int numobjsupport, float* sigmask, int geometricsteps, float epsilon, float* background, float probef1)
    {
        ssc_info(format("Starting RAAR - p: {} o: {} r: {} b: {} n: {}",
                    psizex, osizex, numrois, bsize, numiter));
        {
            std::vector<int> gpus;
            for(int g=0; g<ngpus; g++)
                gpus.push_back(cpugpus[g]);

            RAAR* raar = CreateRAAR((float*)cpudif, dim3(dsizex,dsizex,numrois), (complex*)cpuprobe, dim3(psizex,psizex,psizez), (complex*)cpuobj, dim3(osizex, osizey),
                    (ROI*)cpurois, numrois, bsize, rfactors, gpus, objsupport, probesupport, numobjsupport, sigmask, geometricsteps, background, probef1, epsilon);

            raar->ptycho->objmomentum = objbeta; // why is this not already inside CreateRAAR?
            raar->ptycho->probemomentum = probebeta;

            RAARRun(*raar, numiter); // perhaps objbeta should also be a  parameter here, since it works like tvmu and epsilon

            DestroyRAAR(raar);

            ssc_info("End RAAR.");
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
        ssc_debug("Disabled P2P.");
    };
};

Initializer init;
