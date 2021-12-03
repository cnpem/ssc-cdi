// @Author: Giovanni L. Baraldi
// File contains implementations for gpu memory management.

#ifndef _TYPES_H
#define _TYPES_H

#ifdef __CUDACC__
 #define restrict __restrict__
#else
 #define restrict
#endif

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cufft.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <string>

#include "complex.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

//std::stringstream& GetLog();
inline void SaveLog(){};

 /*
#define Log(x) //GetLog() << __FILE__ << " " <<__LINE__ << " " << x << "\n"; 
#define LogB(x) //GetLog() << __FILE__ << " " <<__LINE__ << " " << x << "..."; 
#define LogE() //GetLog() << " Done." << "\n"; 
 //*/

//*
#define LogB(x)
#define LogE()
#define Log(x) 
//*/

//#define SyncDebug cudaDeviceSynchronize()
#define SyncDebug

#define __hevice __host__ __device__ inline

#define ErrorAssert(statement, message){ if(!(statement)){ std::cerr << __LINE__ << " in " << __FILE__ << ":\n" << message << std::endl; Log(message); SaveLog(); exit(-1); } }

#define Warning(statement, message){ if(!(statement)){ std::cerr << __LINE__ << " in " << __FILE__ << ":\n" << message << std::endl; Log(message); SaveLog(); } }

#define HANDLE_ERROR( errexp ) { cudaError_t cudaerror = errexp; ErrorAssert( cudaerror == cudaSuccess, "Cuda error: " << std::string(cudaGetErrorString( cudaerror )) ) }

#define HANDLE_FFTERROR( errexp ) { cufftResult fftres = errexp; ErrorAssert( fftres == CUFFT_SUCCESS, "Cufft error: " << fftres ) }

enum MemoryType
{
        ENoAlloc = 0,
        EAllocGPU = 1,
        EAllocCPU = 2,
        EAllocCPUGPU = 3,
        EAllocManaged = 4,
        EAllocSBManaged = 12,
};

template<typename Type>
inline __device__ __host__ Type sq(const Type& x){ return x*x; }

template<typename Type> 
struct Image2D
{
        MemoryType memorytype;

        bool bIsManaged() const { return memorytype == MemoryType::EAllocManaged; }
        bool bHasAllocCPU() const { return (memorytype&MemoryType::EAllocCPU) == MemoryType::EAllocCPU; }
        bool bHasAllocGPU() const { return (memorytype&MemoryType::EAllocGPU) == MemoryType::EAllocGPU; }

        Image2D(size_t _sizex, size_t _sizey, size_t _sizez = 1, MemoryType memtype = MemoryType::EAllocCPUGPU): 
                sizex(_sizex), sizey(_sizey), sizez(_sizez), size(_sizex *_sizey*_sizez), memorytype(memtype), gpuptr(nullptr), cpuptr(nullptr)
        {
                LogB("Creating Image of size: " << sizex << " " << sizey << " " << sizez << " (" << sizeof(Type) << ")");
                AllocManaged();
                AllocGPU();
                AllocCPU();
                LogE();
        }

        Image2D(Type* newdata, size_t _sizex, size_t _sizey, size_t _sizez = 1, MemoryType memtype = MemoryType::EAllocCPUGPU): 
                Image2D<Type>(_sizex,_sizey,_sizez,memtype)
        {
                if(memtype == MemoryType::ENoAlloc)
                {
                        this->cpuptr = newdata;
                        this->gpuptr = newdata;
                }
                else
                {
                        LogB("Creating Image of size: " << sizex << " " << sizey << " " << sizez << " (" << sizeof(Type) << ") from pointer " << newdata);
                        if(bHasAllocGPU() || bIsManaged())
                                this->CopyFrom(newdata);
                        if(bHasAllocCPU())
                                memcpy(cpuptr, newdata, sizeof(Type)*size);
                        LogE();
                }
        }
        explicit Image2D(Image2D<Type>& other): 
                Image2D(other.sizex,other.sizey,other.sizez,(other.memorytype != MemoryType::ENoAlloc) ? other.memorytype : MemoryType::EAllocCPUGPU)
        {
                CopyFrom(other);
        }
        Image2D(const dim3& dim, MemoryType memtype = MemoryType::EAllocCPUGPU): Image2D(dim.x,dim.y,dim.z,memtype) {};
        Image2D(Type* newdata, const dim3& dim, MemoryType memtype = MemoryType::EAllocCPUGPU): Image2D(newdata, dim.x,dim.y,dim.z,memtype) {};

        virtual ~Image2D() 
        {
                cudaDeviceSynchronize();
                HANDLE_ERROR( cudaGetLastError() ); 
                LogB("Dealloc Image of pointer " << gpuptr);

                DeallocManaged();
                DeallocGPU(); 
                DeallocCPU();

                cudaDeviceSynchronize();
                LogE();
                HANDLE_ERROR( cudaGetLastError() );
        }

        size_t sizex = 0;
        size_t sizey = 0;
        size_t sizez = 1;
        size_t size = 0;

        Type* gpuptr = nullptr;
        Type* cpuptr = nullptr;

        void SetGPUToZero(){ HANDLE_ERROR( cudaMemset(gpuptr, 0, size*sizeof(Type)) ); SyncDebug; }
        void SetGPUToZero(cudaStream_t stream){ HANDLE_ERROR( cudaMemsetAsync(gpuptr, 0, size*sizeof(Type), stream) ); SyncDebug; }
        void SetCPUToZero(){  assert(cpuptr != nullptr); memset(cpuptr, 0, size*sizeof(Type)); }

        static const size_t blocksize = 128;
        dim3 Shape() const { return dim3(sizex,sizey,sizez); };
        dim3 ShapeThread() const { dim3 shp = Shape(); return dim3(blocksize, 1, 1); };
        dim3 ShapeBlock() const { dim3 shp = Shape(); return dim3((shp.x + blocksize - 1) / blocksize, shp.y, shp.z); };

        dim3 LinearThread() const { dim3 shp = Shape(); return dim3(size < blocksize ? ((size+31)%32) : blocksize, 1, 1); };
        dim3 LinearBlock() const { dim3 shp = Shape(); return dim3((size + blocksize - 1) / blocksize, 1, 1); };

        void AllocGPU()
        {
                if(!gpuptr && bHasAllocGPU()) HANDLE_ERROR( cudaMalloc((void**)&gpuptr, sizeof(Type)*size) );
        }
        void DeallocGPU()
        {
                if(gpuptr && bHasAllocGPU()) cudaFree(gpuptr);
                gpuptr = nullptr;
        }
        void AllocCPU()
        {
                if(!cpuptr && bHasAllocCPU()) cpuptr = new Type[size];
        }
        void DeallocCPU()
        { 
                if(cpuptr && bHasAllocCPU()) delete[] cpuptr; //cudaFreeHost(cpuptr);
                cpuptr = nullptr;
        }
        void AllocManaged()
        {
                if(bIsManaged())
                {
                        HANDLE_ERROR( cudaMallocManaged((void**)&gpuptr, sizeof(Type)*size) );
                        cpuptr = gpuptr;
                }
        }
        void DeallocManaged()
        {
                if(gpuptr && bIsManaged())
                {
                        cudaFree(gpuptr);
                        cpuptr = gpuptr = nullptr;
                }
        }

        void CopyFrom(const Type* other, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                if(cpysize == -1)
                        cpysize = this->size;
                        
                ErrorAssert(other != nullptr, "Syncing from empty pointer!");
                ErrorAssert(gpuptr != nullptr && cpysize <= this->size, "Not enough space for sync.!");

                if(stream==0)
                        cudaMemcpy(this->gpuptr, other, cpysize*sizeof(Type), cudaMemcpyDefault);
                else
                        cudaMemcpyAsync(this->gpuptr, other, cpysize*sizeof(Type), cudaMemcpyDefault, stream);
        }
        void CopyTo(Type* outptr, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                ErrorAssert(gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");

                if(cpysize < 0)
                        cpysize = size*sizeof(Type);
                else
                        cpysize = cpysize*sizeof(Type);

                if( stream == 0 )
                        HANDLE_ERROR( cudaMemcpy((void*)outptr, (void*)gpuptr, cpysize, cudaMemcpyDefault) )
                else
                        HANDLE_ERROR( cudaMemcpyAsync((void*)outptr, (void*)gpuptr, cpysize, cudaMemcpyDefault, stream) )
                SyncDebug;
        }

        void CopyFrom(const Image2D<Type>& other, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                CopyFrom(other.gpuptr, stream, cpysize);
        }
        void CopyTo(Image2D<Type>& other, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                CopyTo(other.gpuptr, stream, cpysize);
        }

        void LoadToGPU(cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                ErrorAssert(cpuptr && gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");

                if(gpuptr == cpuptr)
                        return;

                if(cpysize < 0)
                        cpysize = size*sizeof(Type);
                else
                        cpysize = cpysize*sizeof(Type);

                if( stream == 0 )
                        HANDLE_ERROR( cudaMemcpy((void*)gpuptr, (void*)cpuptr, sizeof(Type)*size, cudaMemcpyHostToDevice) )
                else
                        HANDLE_ERROR( cudaMemcpyAsync((void*)gpuptr, (void*)cpuptr, sizeof(Type)*size, cudaMemcpyHostToDevice, stream) )
                SyncDebug;
        }
        void LoadFromGPU(cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                ErrorAssert(cpuptr && gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");
                if(gpuptr == cpuptr)
                        return;

                if(cpysize < 0)
                        cpysize = size*sizeof(Type);
                else
                        cpysize = cpysize*sizeof(Type);

                if( stream == 0 )
                        HANDLE_ERROR( cudaMemcpy((void*)cpuptr, (void*)gpuptr, cpysize, cudaMemcpyDeviceToHost) )
                else
                        HANDLE_ERROR( cudaMemcpyAsync((void*)cpuptr, (void*)gpuptr, cpysize, cudaMemcpyDeviceToHost, stream) )
                SyncDebug;
        }
};


typedef Image2D<float> rImage;
typedef Image2D<complex> cImage;

struct CFilter
{
	CFilter() = default;
	explicit CFilter(int _type, float _reg): type((EType)_type), reg(_reg) {} 

	enum EType
	{
		none=0,
		gaussian=1,
		lorentz=2,
		cosine=3,
		rectangle=4,
		FSC
	};

	float reg = 0;
	EType type = EType::none;

	__host__ __device__ inline 
	float Apply(float input)
	{
		if(type == EType::gaussian)
			input *= exp(	-0.693f*reg*sq( input )	);

		else if(type == EType::lorentz)
			input /= 1.0f + reg*sq( input );

		else if(type == EType::cosine)
			input *= cosf(	float(M_PI)*0.5f*input	);

		else if(type == EType::rectangle)
                {
                        float param = fmaxf(input * reg * float(M_PI) * 0.5f, 1E-4f);
			input *= sinf(param) / param;
                }

		return input;
	}
};

#endif