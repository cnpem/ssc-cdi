#include <propagator.hpp>

extern "C"{
void Fraunhoffer::Append(const dim3& dim)
{
    ssc_debug(format("Creating new plan: {} {} {}", dim.x, dim.y, dim.z));
    if(workarea == nullptr || dim.x*dim.y*dim.z > workarea->size)
    {
        ssc_debug(format("Reallocating plan memory to size: {} {} {}",
                    dim.x, dim.y, dim.z));
        if(workarea)
            delete workarea;
        workarea = new cImage(dim.x,dim.y,dim.z);

        for(auto plan : plans)
            cufftSetWorkArea(plan, workarea->gpuptr);
    }

    int i = dims.size();
    dims.push_back(dim);

    int n[] = {(int)dim.x,(int)dim.y};

    cufftHandle newplan;

    size_t worksize;
    ssc_cufft_check( cufftCreate(&newplan) );
    ssc_cufft_check( cufftSetAutoAllocation(newplan,0) );
    ssc_cufft_check( cufftMakePlanMany(newplan,2,n,nullptr,0,0,
                nullptr,0,0,CUFFT_C2C,(int)dim.z,&worksize) );
    ssc_cufft_check( cufftSetWorkArea(newplan, workarea->gpuptr) );

    plans.push_back(newplan);

    ssc_assert(worksize <= 8*workarea->size, "CuFFT being hungry!");
    ssc_debug("Done.");
}

bool dim3EQ(const dim3& d1, const dim3& d2){
    return d1.x==d2.x && d1.y==d2.y && d1.z==d2.z;
}

void Fraunhoffer::Propagate(complex* owave, complex* iwave, dim3 shape, float amount)
{
    bool bPlanExists = false;
    auto dir = (amount > 0) ? CUFFT_FORWARD : CUFFT_INVERSE;

    cufftHandle plan;

    for(int i=0; i<dims.size(); i++) if( dim3EQ(shape,dims[i]) )
    {
        bPlanExists = true;
        plan = plans[i];
    }
    if(!bPlanExists)
    {
        Append(shape);
        plan = plans[plans.size()-1];
    }
    ssc_cufft_check(  cufftExecC2C(plan,iwave,owave,dir)  );
}

Fraunhoffer::~Fraunhoffer()
{
    ssc_debug("Deleting propagator.");
    for(auto plan : plans)
        if(plan) cufftDestroy(plan);
    if(workarea)
        delete workarea;
}

__global__ void KApplyASM(complex* wave, float f1, dim3 shape)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idy = blockIdx.y;
    size_t idz = blockIdx.z;

    if(idx >= shape.x)
        return;

    float xx = float( int(idx+shape.x/2)%int(shape.x) )/float(shape.x) - 0.5f;
    float yy = float( int(idy+shape.y/2)%int(shape.y) )/float(shape.y) - 0.5f;

    wave[idz*shape.x*shape.y + idy*shape.x + idx] *= complex::exp1j( -float(M_PI)/f1 * (xx*xx + yy*yy) ) / float(shape.x*shape.y);
}

void ASM::Propagate(complex* owave, complex* iwave, dim3 shape, float amount)
{
    Fraunhoffer::Propagate(owave,iwave,shape,1);
    KApplyASM<<<dim3((shape.x+127)/128,shape.y,shape.z),dim3(128,1,1)>>>(owave, amount, shape);
    Fraunhoffer::Propagate(owave,owave,shape,-1);
}
}
