// @Author: Giovanni L. Baraldi
// File contains implementations for single and multigpu memory management, along with some commom operations.

/** @file */

#ifndef _TYPES_H
#define _TYPES_H

#ifdef __CUDACC__
#define restrict __restrict__
#else
#define restrict
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "cufft.h"

#if __has_include("thrust/device_vector.h")
#include "thrust/equal.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/complex.h>
#define THRUST_INCLUDED
#else
#pragma message(">>> Nvidia Thrust NOT included <<<")
#endif

#include "complex.hpp"
#include "operations.hpp"
#include "logger.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


//#define SyncDebug cudaDeviceSynchronize()
#define SyncDebug

#define __hevice __host__ __device__ inline

__global__ void KSwapZ(float* vec, size_t N);
__global__ void KSwapZ_4gpu(float* vec0, float* vec1, float* vec2, float* vec3, int gpu, size_t N);

/**
 * Type of allocation to be passed for each Array.
 * */
enum MemoryType {
    ENoAlloc = 0,          //!< Dont allocated memory, useful for wrapping an existent pointer.
    EAllocGPU = 1,         //!< GPU-Only allocation
    EAllocCPU = 2,         //!< CPU-Only allocation.
    EAllocCPUGPU = 3,      //!< Standard, allocate both im GPU and CPU.
    EAllocManaged = 4,     //!< Use CUDA's mamaged memory system, in this case cpuptr==gpuptr.
    EAllocSBManaged = 12,  //!< For future realase, dont use this yet!
};

namespace _AuxReduction {
template <typename T>
struct variable_norm2 {
    __hevice float operator()(const T& x) const { return float(x) * float(x); }
};
template <>
struct variable_norm2<complex> {
    __hevice float operator()(const complex& c) const { return c.x * c.x + c.y * c.y; }
};

template <typename T1, typename T2>
struct productstruct {
    __hevice T1 operator()(const T1& x1, const T2& x2) const { return x1 * x2; }
};
template <typename T1>
struct productstruct<T1, complex> {
    __hevice complex operator()(const T1& x1, const complex& c) const { return c * x1; }
};
template <typename T2>
struct productstruct<complex, T2> {
    __hevice complex operator()(const complex& c, const T2& x2) const { return c * x2; }
};
template <>
struct productstruct<complex, complex> {
    __hevice complex operator()(const complex& c1, const complex& c2) const { return c1 * c2; }
};

// template<typename T> struct addstruct{ __hevice T operator()(const T& x1, const T& x2) const { return x1+x2; } };
// template<typename T1, typename T2> struct maximumstruct{ __hevice T1 operator()(const T1& x1, const T2& x2) const {
// return max(x1,x2); } };
}  // namespace _AuxReduction

/**
 * Base class for single GPU basic operations and memory management.
 * */
template <typename Type>
struct Image2D {
    MemoryType memorytype;

    bool bIsManaged() const { return memorytype == MemoryType::EAllocManaged; }
    bool bHasAllocCPU() const { return (memorytype & MemoryType::EAllocCPU) == MemoryType::EAllocCPU; }
    bool bHasAllocGPU() const { return (memorytype & MemoryType::EAllocGPU) == MemoryType::EAllocGPU; }

    /**
     * Constructor
     */
    Image2D(size_t _sizex, size_t _sizey, size_t _sizez = 1, MemoryType memtype = MemoryType::EAllocCPUGPU)
        : sizex(_sizex),
          sizey(_sizey),
          sizez(_sizez),
          size(_sizex * _sizey * _sizez),
          memorytype(memtype),
          gpuptr(nullptr),
          cpuptr(nullptr) {
        ssc_debug(format("Creating Image of size: {} {} {} ({})", sizex, sizey, sizez, sizeof(Type)));
        AllocManaged();
        AllocGPU();
        AllocCPU();
    }
    /**
     * Makes a copy from a given pointer during construction.
     * */
    Image2D(Type* newdata, size_t _sizex, size_t _sizey, size_t _sizez = 1,
            MemoryType memtype = MemoryType::EAllocCPUGPU)
        : Image2D<Type>(_sizex, _sizey, _sizez, memtype) {
        if (memtype == MemoryType::ENoAlloc) {
            this->cpuptr = newdata;
            this->gpuptr = newdata;
        } else {
            ssc_debug(format("Creating Image of size: {} {} {} ({}) from pointer {}",
                        sizex, sizey, sizez, sizeof(Type), (void*)newdata));
            if (bHasAllocGPU() || bIsManaged()) this->CopyFrom(newdata);
            if (bHasAllocCPU()) memcpy(cpuptr, newdata, sizeof(Type) * GetSize());
        }
    }
    /**
     * Makes a copy of given array.
     */
    explicit Image2D(Image2D<Type>& other)
        : Image2D(other.sizex, other.sizey, other.sizez,
                  (other.memorytype != MemoryType::ENoAlloc) ? other.memorytype : MemoryType::EAllocCPUGPU) {
        CopyFrom(other);
    }
    /**
     * Constructor
     */
    Image2D(const dim3& dim, MemoryType memtype = MemoryType::EAllocCPUGPU) : Image2D(dim.x, dim.y, dim.z, memtype){};
    /**
     * Constructor
     */
    Image2D(Type* newdata, const dim3& dim, MemoryType memtype = MemoryType::EAllocCPUGPU)
        : Image2D(newdata, dim.x, dim.y, dim.z, memtype){};

    /**
     * Destructor
     */
    virtual ~Image2D() {
        cudaDeviceSynchronize();
        ssc_debug(format("Dealloc Image of pointer {}", (void*)gpuptr));

        DeallocManaged();
        DeallocGPU();
        DeallocCPU();

        cudaDeviceSynchronize();
        ssc_cuda_check(cudaGetLastError());
    }

    size_t sizex = 0;
    size_t sizey = 0;
    size_t sizez = 1;

    size_t size = 0;  //!< sizex*sizey*sizez

    Type* gpuptr = nullptr;  //!< Pointer in GPU Memory.
    Type* cpuptr = nullptr;  //!< Pointer in CPU Memory.
    /**
     * Zero-fill.
     * */
    void SetGPUToZero() {
        ssc_cuda_check(cudaMemset(gpuptr, 0, size * sizeof(Type)));
        SyncDebug;
    }
    /**
     * Stramem zero-fill.
     * */
    void SetGPUToZero(cudaStream_t stream) {
        ssc_cuda_check(cudaMemsetAsync(gpuptr, 0, size * sizeof(Type), stream));
        SyncDebug;
    }
    /**
     * Zero-fill.
     * */
    void SetCPUToZero() {
        assert(cpuptr != nullptr);
        memset(cpuptr, 0, size * sizeof(Type));
    }

    static const size_t blocksize = 64;  //!< Default number of threads per block.

    dim3 Shape() const { return dim3(sizex, sizey, sizez); };
    /**
     * Get a dim3() object for kernel launch.
     * */
    dim3 ShapeThread() const {
        dim3 shp = Shape();
        return dim3(blocksize, 1, 1);
    };
    /**
     * Get a dim3() object for kernel launch.
     * */
    dim3 ShapeBlock() const {
        dim3 shp = Shape();
        return dim3((shp.x + blocksize - 1) / blocksize, shp.y, shp.z);
    };

    dim3 LinearThread() const {
        dim3 shp = Shape();
        return dim3(size < blocksize ? ((size + 31) % 32) : blocksize, 1, 1);
    };
    dim3 LinearBlock() const {
        dim3 shp = Shape();
        return dim3((size + blocksize - 1) / blocksize, 1, 1);
    };

    void AllocGPU() {
        if (!gpuptr && bHasAllocGPU()) ssc_cuda_check(cudaMalloc((void**)&gpuptr, sizeof(Type) * size));
    }
    void DeallocGPU() {
        if (gpuptr && bHasAllocGPU()) cudaFree(gpuptr);
        gpuptr = nullptr;
    }
    void AllocCPU() {
        if (!cpuptr && bHasAllocCPU()) cpuptr = new Type[size];
    }
    void DeallocCPU() {
        if (cpuptr && bHasAllocCPU()) delete[] cpuptr;  // cudaFreeHost(cpuptr);
        cpuptr = nullptr;
    }
    void AllocManaged() {
        if (bIsManaged()) {
            ssc_cuda_check(cudaMallocManaged((void**)&gpuptr, sizeof(Type) * size));
            cpuptr = gpuptr;
        }
    }
    void DeallocManaged() {
        if (gpuptr && bIsManaged()) {
            cudaFree(gpuptr);
            cpuptr = gpuptr = nullptr;
        }
    }

    template <typename Type2 = Type>
    bool operator==(const Image2D<Type2>& other) const {
        if (size != other.size || sizex != other.sizex ||
                sizey != other.sizey || sizez != other.sizez)
            return false;
        return thrust::equal(thrust::device,
                this->gpuptr, this->gpuptr + this->size * sizeof(Type),
                other.gpuptr);
    }

    template <typename Type2 = Type>
    Image2D<Type>& operator+=(const Image2D<Type2>& other) {
        ssc_assert(this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for addition.");
        BasicOps::KB_Add<Type, Type2>
            <<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
        return *this;
    }
    template <typename Type2 = Type>
    Image2D<Type>& operator-=(const Image2D<Type2>& other) {
        ssc_assert(this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for subtraction.");
        BasicOps::KB_Sub<Type, Type2>
            <<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
        return *this;
    }
    template <typename Type2 = Type>
    Image2D<Type>& operator*=(const Image2D<Type2>& other) {
        ssc_assert(this->size == other.size || this->sizex == other.sizex,
                    "Incompatible GPU shape for multiplication.");
        BasicOps::KB_Mul<Type, Type2>
            <<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
        return *this;
    }
    template <typename Type2 = Type>
    Image2D<Type>& operator/=(const Image2D<Type2>& other) {
        ssc_assert(this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for division.");
        BasicOps::KB_Div<Type, Type2>
            <<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
        return *this;
    }

    template <typename Type2 = Type>
    Image2D<Type>& operator+=(Type2 other) {
        BasicOps::KB_Add<Type, Type2><<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, other, this->size);
        return *this;
    }
    template <typename Type2 = Type>
    Image2D<Type>& operator-=(Type2 other) {
        BasicOps::KB_Sub<Type, Type2><<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, other, this->size);
        return *this;
    }
    template <typename Type2 = Type>
    Image2D<Type>& operator*=(Type2 other) {
        BasicOps::KB_Mul<Type, Type2><<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, other, this->size);
        return *this;
    }
    template <typename Type2 = Type>
    Image2D<Type>& operator/=(Type2 other) {
        BasicOps::KB_Div<Type, Type2><<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, other, this->size);
        return *this;
    }

    template <typename Type2>
    void DataAs(Type2* fill, size_t tsize = 0, float threshold = -1) const {
        if (tsize == 0) tsize = this->size;

        // if(!std::is_same<Type,Type2>::value)

        BasicOps::KConvert<Type2, Type><<<(tsize + 31) / 32, 32>>>(fill, this->gpuptr, tsize, threshold);
    }

    void DataAs(void* fill, EType::TypeEnum type, size_t tsize = 0, float threshold = -1) const {
        switch (type) {
            case EType::TypeEnum::UINT8:
                DataAs<uint8_t>((uint8_t*)fill, tsize, threshold);
                break;
            case EType::TypeEnum::UINT16:
                DataAs<uint16_t>((uint16_t*)fill, tsize, threshold);
                break;
            case EType::TypeEnum::INT32:
                DataAs<int32_t>((int32_t*)fill, tsize, threshold);
                break;
            case EType::TypeEnum::FLOAT32:
                DataAs<float>((float*)fill, tsize, threshold);
                break;
            case EType::TypeEnum::DOUBLE:
                DataAs<double>((double*)fill, tsize, threshold);
                break;
            case EType::TypeEnum::HALF:
                DataAs<__half>((__half*)fill, tsize, threshold);
                break;
            default:
                ssc_error("Invalid data type!");
                exit(-1);
                break;
        }
    }

    /**
     * Copies memory from given pointer. Cpysize == -1 -> Cpysize = this->size
     * */
    void CopyFrom(const Type* other, cudaStream_t stream = 0, int64_t cpysize = -1) {
        if (cpysize == -1) cpysize = this->size;

        ssc_assert(other != nullptr, "Syncing from empty pointer!");
        ssc_assert(gpuptr != nullptr && cpysize <= this->size, "Not enough space for sync.!");

        if (stream == 0)
            cudaMemcpy(this->gpuptr, other, cpysize * sizeof(Type), cudaMemcpyDefault);
        else
            cudaMemcpyAsync(this->gpuptr, other, cpysize * sizeof(Type), cudaMemcpyDefault, stream);
    }
    /**
     * Copies memory to given pointer. Cpysize == -1 -> Cpysize = this->size
     * */
    void CopyTo(Type* outptr, cudaStream_t stream = 0, int64_t cpysize = -1) {
        ssc_assert(gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");

        if (cpysize < 0)
            cpysize = size * sizeof(Type);
        else
            cpysize = cpysize * sizeof(Type);

        if (stream == 0)
            ssc_cuda_check(cudaMemcpy((void*)outptr, (void*)gpuptr, cpysize, cudaMemcpyDefault));
        else
            ssc_cuda_check(cudaMemcpyAsync((void*)outptr, (void*)gpuptr, cpysize, cudaMemcpyDefault, stream));
        SyncDebug;
    }

    void CopyRoiTo(Type* outptr, dim3 offset, dim3 roi_size) {
        cudaMemcpy3DParms params = {0};

        // Set source parameters
        params.srcPtr.ptr = gpuptr + offset.z * sizex * sizey + offset.y * sizex + offset.x;
        params.srcPtr.pitch = sizex * sizeof(Type);
        params.srcPtr.xsize = roi_size.x * sizeof(Type);
        params.srcPtr.ysize = roi_size.y;

        params.srcPos.x = 0;
        params.srcPos.y = 0;
        params.srcPos.z = 0;

        // Set destination parameters
        params.dstPtr.ptr = outptr;
        params.dstPtr.pitch = roi_size.x * sizeof(Type);
        params.dstPtr.xsize = roi_size.x * sizeof(Type);
        params.dstPtr.ysize = roi_size.y;

        params.dstPos.x = 0;
        params.dstPos.y = 0;
        params.dstPos.z = 0;

        // Set copy dimensions
        params.extent.width = roi_size.x * sizeof(Type);
        params.extent.height = roi_size.y;
        params.extent.depth = roi_size.z;

        // Set copy direction
        params.kind = cudaMemcpyDeviceToDevice;

        // Perform the copy
        cudaMemcpy3D(&params);
    }

    void CopyRoiTo(Image2D<Type>& other, dim3 offset, dim3 roi_size) {
        CopyRoiTo(other.gpuptr, offset, roi_size);
    }

    /**
     * Copies from given array.
     */
    void CopyFrom(const Image2D<Type>& other, cudaStream_t stream = 0, int64_t cpysize = -1) {
        CopyFrom(other.gpuptr, stream, cpysize);
    }
    /**
     * Copies to given array.
     * */
    void CopyTo(Image2D<Type>& other, cudaStream_t stream = 0, int64_t cpysize = -1) {
        CopyTo(other.gpuptr, stream, cpysize);
    }

    /**
     * If the allocation is not using unified address, copies the contents of cpuptr to gpuptr.
     * */
    void LoadToGPU(cudaStream_t stream = 0, int64_t cpysize = -1) {
        ssc_assert(cpuptr && gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");

        if (gpuptr == cpuptr) return;

        if (cpysize < 0)
            cpysize = size * sizeof(Type);
        else
            cpysize = cpysize * sizeof(Type);

        if (stream == 0)
            ssc_cuda_check(cudaMemcpy((void*)gpuptr, (void*)cpuptr, sizeof(Type) * size, cudaMemcpyHostToDevice));
        else
            ssc_cuda_check(
                cudaMemcpyAsync((void*)gpuptr, (void*)cpuptr, sizeof(Type) * size, cudaMemcpyHostToDevice, stream));
        SyncDebug;
    }
    /**
     * If the allocation is not using unified address, copies the contents of gpuptr to cpuptr.
     * */
    void LoadFromGPU(cudaStream_t stream = 0, int64_t cpysize = -1) {
        ssc_assert(cpuptr && gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");
        if (gpuptr == cpuptr) return;

        if (cpysize < 0)
            cpysize = size * sizeof(Type);
        else
            cpysize = cpysize * sizeof(Type);

        if (stream == 0)
            ssc_cuda_check(cudaMemcpy((void*)cpuptr, (void*)gpuptr, cpysize, cudaMemcpyDeviceToHost));
        else
            ssc_cuda_check(cudaMemcpyAsync((void*)cpuptr, (void*)gpuptr, cpysize, cudaMemcpyDeviceToHost, stream));
        SyncDebug;
    }

    size_t GetSize() const { return size; };
    /**
     * Natural exponential.
     * */
    void exp(bool bNeg = false) { BasicOps::KB_Exp<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, size, bNeg); };
    /**
     * Natural logarithm.
     * */
    void ln() { BasicOps::KB_Log<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, size); };

    /**
     * Computes *this = exp(i*other)
     * */
    void exp1j(Image2D<float>& other) {
        BasicOps::KB_exp1j<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, other.gpuptr, size);
    };
    /**
     * Computes *this = arctan(other.imag/other.real)
     * */
    void angle(Image2D<float>& other) {
        BasicOps::KB_log1j<<<ShapeBlock(), ShapeThread()>>>(other.gpuptr, this->gpuptr, size);
    };

    void SwapZ() {
        ssc_assert(sizex == sizey && sizex == sizez, "Invalid shape for square transpose.");
        KSwapZ<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, sizex);
    };
    /**
     * Returns a cufft plan to 1D-transform this->shape.
     * */
    cufftHandle GetFFTPlan1D(int batch = -1) {
        cufftHandle plan;
        ssc_cufft_check(cufftPlan1d(&plan, sizex, CUFFT_C2C, (batch > 0) ? batch : (sizey * sizez)));
        return plan;
    }
    /**
     * Returns a cufft plan to 2D-transform this->shape.
     * */
    cufftHandle GetFFTPlan2D() {
        cufftHandle plan;
        ssc_cufft_check(cufftPlan2d(&plan, sizey, sizex, CUFFT_C2C));
        return plan;
    }
    /**
     * Returns a cufft plan to 3D-transform this->shape.
     * */
    cufftHandle GetFFTPlan3D() {
        cufftHandle plan;
        ssc_cufft_check(cufftPlan3d(&plan, sizez, sizey, sizex, CUFFT_C2C));
        return plan;
    }
    /**
     * Executes FFT on this array.
     * */
    void FFT(cufftHandle plan) {
        for (int z = 0; z < sizez; z++)
            ssc_cufft_check(cufftExecC2C(plan, gpuptr + z * sizex * sizey, gpuptr + z * sizex * sizey, CUFFT_FORWARD));
    }
    /**
     * Executes IFFT on this array. Not normalized.
     * */
    void IFFT(cufftHandle plan) {
        for (int z = 0; z < sizez; z++)
            ssc_cufft_check(cufftExecC2C(plan, gpuptr + z * sizex * sizey, gpuptr + z * sizex * sizey, CUFFT_INVERSE));
    }

    /**
     * Ensures: a <= *this <= b
     * */
    void Clamp(Type a, Type b) {
        BasicOps::KB_Clamp<<<LinearBlock(), LinearThread()>>>(this->gpuptr, a, b, this->size);
    }
    /**
     * *this = (*this)^P
     * */
    void Power(float P) { BasicOps::KB_Power<Type><<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, P, this->size); }
    /**
     * Dumps absolute value^2 to out.
     * */
    void abs2(Image2D<float>& out) {
        BasicOps::KB_ABS2<Type><<<ShapeBlock(), ShapeThread()>>>(out.gpuptr, this->gpuptr, this->size);
    }

    /**
     * Sum reduction
     *  */
    Type SumCPU() {
        Type s { 0 };
        for (int i = 0; i < this->size; i++) s += this->cpuptr[i];
        return s;
    }

#ifdef THRUST_INCLUDED
    /**
     * Sum reduction
     *  */
    Type SumGPU() { return thrust::reduce(thrust::device, gpuptr, gpuptr + size, Type(0), thrust::plus<Type>()); }
    /**
     * Ranged sum reduction
     *  */
    Type SumGPU(size_t start, size_t end) {
        return thrust::reduce(thrust::device, gpuptr + start, gpuptr + end, Type(0), thrust::plus<Type>());
    }
    /**
     * Computes sum(|.|^2)
     */
    float Norm2() {
        return thrust::transform_reduce(thrust::device, gpuptr, gpuptr + size, _AuxReduction::variable_norm2<Type>(),
                                        0.0f, thrust::plus<float>());
    }

    /**
     * Scalar/dot product. Note there is no conjugation in a complex.complex product due to non commutativity in the
     * operation.
     * */
    template <typename T2>
    Type dot(Image2D<T2>& other) {
        return thrust::inner_product(thrust::device, gpuptr, gpuptr + size, other.gpuptr, Type(0), thrust::plus<Type>(),
                                     _AuxReduction::productstruct<Type, T2>());
    }

    /**
     * Computes the maximum element in array.
     * */
    Type max() {
        return thrust::transform_reduce(thrust::device, gpuptr, gpuptr + size, thrust::identity<Type>(),
                                        std::numeric_limits<Type>::lowest(), thrust::maximum<Type>());
    }

    float maxAbs2() {
         return thrust::transform_reduce(thrust::device, gpuptr, gpuptr + size,
                 [] __device__(const Type& t) {
                    return t.abs2();
                 },
                 std::numeric_limits<float>::lowest(), thrust::maximum<float>());
    }

#else
    /**
     * Sum reduction
     *  */
    Type SumGPU(cudaStream_t stream = 0) {
        LoadFromGPU(stream);
        cudaStreamSynchronize(stream);
        return SumCPU();
    }
#endif
    /**
     * Applies the 1d-fftshift to this array.
     * */
    void FFTShift1() { BasicOps::KFFTshift1<<<ShapeBlock(), ShapeThread()>>>(gpuptr, sizex, sizey); }
    /**
     * Applies the 2d-fftshift to this array.
     * */
    void FFTShift2() { BasicOps::KFFTshift2<<<ShapeBlock(), ShapeThread()>>>(gpuptr, sizex, sizey); }
    /**
     * Applies the 3d-fftshift to this array.
     * */
    void FFTShift3() { BasicOps::KFFTshift3<<<ShapeBlock(), ShapeThread()>>>(gpuptr, sizex, sizey, sizez); }

    template <typename Op>
    void UnaryOperator(Op op) {
        thrust::transform(thrust::device, gpuptr, gpuptr + size, gpuptr, op);
    }

    /**
     * Returns reference array containing the Z-index specified
     * */
    Image2D<Type>* SliceZ(size_t index) const {
        return new Image2D<Type>(gpuptr + sizex * sizey * index, sizex, sizey, 1, MemoryType::ENoAlloc);
    };
};

/**
 * Class meant to be passed to a kernel launch. Provides a ndarray-like structure to cuda kernels.
 * */
template <typename Type>
struct GArray {
    /**
     * Constructor
     *  */
    GArray<Type>(Type* _ptr, const dim3& _shape) : ptr(_ptr), shape(_shape){};
    /**
     * Constructor
     *  */
    GArray<Type>(Type* _ptr, size_t x, size_t y, size_t z) : ptr(_ptr), shape(dim3(x, y, z)){};

    /**
     * Generates a kernel array form given Image/Volume
     * */
    GArray<Type>(const Image2D<Type>& img) : ptr(img.gpuptr), shape(img.sizex, img.sizey, img.sizez){};

    /**
     * Shapeoffset is meant to create a slice:
     *  -> ptr += shape
     *  -> newshape = shape - shapeoffset
     * It is the caller's responsibility to deal with a possible noncontiguous memory issue.
     * */
    explicit GArray<Type>(const Image2D<Type>& img, dim3 shapeoffset) : GArray<Type>(img) {
        assert(shape.x >= shapeoffset.x && shape.y >= shapeoffset.y && shape.z >= shapeoffset.z);
        ptr += shapeoffset.x + shape.x * (shapeoffset.y + shape.y * shapeoffset.z);
        shape.x -= shapeoffset.x;
        shape.y -= shapeoffset.y;
        shape.z -= shapeoffset.z;
    }

    Type* ptr;
    dim3 shape;

    __hevice Type& operator[](size_t i) { return ptr[i]; };
    __hevice const Type& operator[](size_t i) const { return ptr[i]; };
    __hevice Type& operator()(size_t k, size_t j, size_t i) { return ptr[(k * shape.y + j) * shape.x + i]; };
    __hevice const Type& operator()(size_t k, size_t j, size_t i) const {
        return ptr[(k * shape.y + j) * shape.x + i];
    };
    __hevice Type& operator()(size_t j, size_t i) { return ptr[j * shape.x + i]; };
    __hevice const Type& operator()(size_t j, size_t i) const { return ptr[j * shape.x + i]; };
};


typedef Image2D<float> rImage;
typedef Image2D<complex> cImage;
typedef Image2D<complex16> hcImage;

// Note: The outer brackets are necessary for shallow declaration of g.
// Removing them may cause compile errors.

#define MEXEC(func, array1, ...)                            \
    {                                                       \
        for (int g = 0; g < array1.ngpus; g++) {            \
            cudaSetDevice(array1.indices[g]);               \
            dim3& blk = array1.blk[g];                      \
            dim3& thr = array1.thr[g];                      \
            func<<<blk, thr>>>(array1.Ptr(g), __VA_ARGS__); \
            ssc_cuda_check(cudaGetLastError());               \
        }                                                   \
    }

#define MGPULOOP(execcode)                 \
    {                                      \
        for (int g = 0; g < GetN(); g++) { \
            Set(g);                        \
            execcode;                      \
        }                                  \
    }
#define MultiKernel(gpus, func, blk, thr, ...)  \
    {                                           \
        for (int g = 0; g < gpus.size(); g++) { \
            cudaSetDevice(gpus[g]);             \
            func<<<blk, thr>>>(__VA_ARGS__);    \
        }                                       \
    }

inline void SetDevice(const std::vector<int>& gpuindices, int g) { cudaSetDevice(gpuindices[g]); }

inline void SyncDevices(const std::vector<int>& gpuindices) {
    for (int device : gpuindices) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
}

/**
 * Base class for multigpu arrays.
 * */
struct MultiGPU {
    MultiGPU(const std::vector<int>& gpus) : ngpus(gpus.size()), gpuindices(gpus){};

    const int ngpus = 1;
    const std::vector<int> gpuindices;

    void Set(int g) { cudaSetDevice(this->gpuindices[g]); }
    int GetN() const { return this->ngpus; }
    void SyncDevices() {
        for (int g : gpuindices) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
    };
};

/**
 * Array class for multigpu basic operations and memory management.
 * */
template <typename Type>
struct MImage : public MultiGPU {
    MemoryType memorytype;

    bool bIsManaged() const { return memorytype == MemoryType::EAllocManaged; }
    bool bHasAllocCPU() const { return (memorytype & MemoryType::EAllocCPU) == MemoryType::EAllocCPU; }
    bool bHasAllocGPU() const { return (memorytype & MemoryType::EAllocGPU) == MemoryType::EAllocGPU; }

    Image2D<Type>* arrays[16];
    size_t offsets[16];

    bool bBroadcast = false;

    Type* Ptr(int g) const { return arrays[g]->gpuptr; }
    Image2D<Type>& operator[](int n) { return *(arrays[n]); };

    /**
     * Constructor. If broadcast = true, makes each gpu have a full copy of the array. Otherwise, each gpu gets a chunk
     * of the memory.
     * */
    MImage(size_t _sizex, size_t _sizey, size_t _sizez, bool _bBroadcast, const std::vector<int>& gpus,
           MemoryType memtype = MemoryType::EAllocCPUGPU)
        : MultiGPU(gpus),
          sizex(_sizex),
          sizey(_sizey),
          sizez(_sizez),
          size(_sizex * _sizey * _sizez),
          bBroadcast(_bBroadcast),
          memorytype(memtype) {
        ssc_assert(gpus.size() <= 16, "Too many gpus!");
        ssc_debug(format("Creating MImage of size: {} {} {} ({})",
                    sizex, sizey, sizez, sizeof(Type)));

        size_t zstep, zdistrib;

        if (bBroadcast) {
            zstep = 0;
            zdistrib = sizez;
        } else {
            if (gpus.size() > sizez)
                ssc_debug("Warning: More GPUs than slices!");
            zstep = (sizez + gpus.size() - 1) / gpus.size();
            zdistrib = zstep;
        }

        for (int g = 0; g < this->ngpus; g++) {
            if (zstep * (g + 1) > sizez) {
                if (sizez > zstep * g)
                    zdistrib = sizez - zstep * g;
                else
                    zdistrib = 1;
            }

            Set(g);

            arrays[g] = new Image2D<Type>(sizex, sizey, zdistrib, memtype);
            offsets[g] = sizex * sizey * zstep * g;

            if (sizez <= zstep * g) arrays[g]->sizez = 0;
        }
    }

    /**
     * Constructor. If broadcast = true, makes each gpu have a full copy of the array. Otherwise, each gpu gets a chunk
     * of the memory.
     * */
    MImage(Type* newdata, size_t _sizex, size_t _sizey, size_t _sizez, bool _bBroadcast, const std::vector<int>& gpus,
           MemoryType memtype = MemoryType::EAllocCPUGPU)
        : MImage<Type>(_sizex, _sizey, _sizez, _bBroadcast, gpus, memtype) {
        for (int g = 0; g < this->ngpus; g++) {
            Set(g);
            arrays[g]->CopyFrom(newdata + offsets[g]);
        }
        ssc_debug(format("^ from pointer {}", (void*)newdata));
    }
    /**
     * Constructor. If broadcast = true, makes each gpu have a full copy of the array. Otherwise, each gpu gets a chunk
     * of the memory.
     * */
    MImage(dim3 dim, bool _bBroadcast, const std::vector<int>& gpus, MemoryType memtype = MemoryType::EAllocCPUGPU)
        : MImage<Type>(dim.x, dim.y, dim.z, _bBroadcast, gpus, memtype){};
    /**
     * Constructor. If broadcast = true, makes each gpu have a full copy of the array. Otherwise, each gpu gets a chunk
     * of the memory.
     * */
    MImage(Type* newdata, dim3 dim, bool _bBroadcast, const std::vector<int>& gpus,
           MemoryType memtype = MemoryType::EAllocCPUGPU)
        : MImage<Type>(newdata, dim.x, dim.y, dim.z, _bBroadcast, gpus, memtype){};

    MImage(MImage<Type>& other)
        : MImage<Type>(other.sizex, other.sizey, other.sizez, other.bBroadcast, other.gpuindices, other.memorytype) {
        CopyFrom(other);
    };

    virtual ~MImage() {
        ssc_debug(format("Deleting MImage of shape: {} {} {} ({})",
                    sizex, sizey, sizez, sizeof(Type)));
        MGPULOOP(delete arrays[g]; arrays[g] = nullptr;);
        ssc_cuda_check(cudaGetLastError());
    }
    dim3 Shape() const { return dim3(sizex, sizey, sizez); };

    size_t sizex = 0;
    size_t sizey = 0;
    size_t sizez = 1;
    size_t size = 0;

    void SetGPUToZero() { MGPULOOP(arrays[g]->SetGPUToZero();); };
    void Clamp(Type a, Type b) { MGPULOOP(arrays[g]->Clamp(a, b);); }

    template <typename Type2 = Type>
    bool operator==(const MImage<Type2>& other) {
        for(int g = 0; g < GetN(); ++g) {
            Set(g);
            if (*(arrays[g]) != *(other.arrays[g])) {
                return false;
            }
        }
        return true;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator+=(const MImage<Type2>& other) {
        MGPULOOP(*(arrays[g]) += *(other.arrays[g]););
        return *this;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator-=(const MImage<Type2>& other) {
        MGPULOOP(*(arrays[g]) -= *(other.arrays[g]););
        return *this;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator*=(const MImage<Type2>& other) {
        MGPULOOP(*(arrays[g]) *= *(other.arrays[g]););
        return *this;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator/=(const MImage<Type2>& other) {
        MGPULOOP(*(arrays[g]) /= *(other.arrays[g]););
        return *this;
    }

    template <typename Type2 = Type>
    MImage<Type>& operator+=(Type2 other) {
        MGPULOOP(*(arrays[g]) += other;);
        return *this;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator-=(Type2 other) {
        MGPULOOP(*(arrays[g]) -= other;);
        return *this;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator*=(Type2 other) {
        MGPULOOP(*(arrays[g]) *= other;);
        return *this;
    }
    template <typename Type2 = Type>
    MImage<Type>& operator/=(Type2 other) {
        MGPULOOP(*(arrays[g]) /= other;);
        return *this;
    }

    void exp(bool bNeg = false) {
        for (int g = 0; g < GetN(); g++) {
            Set(g);
            arrays[g]->exp(bNeg);
        }
    };
    void ln() {
        for (int g = 0; g < GetN(); g++) {
            Set(g);
            arrays[g]->ln();
        }
    };

    void exp1j(MImage<float>& other) { MGPULOOP(arrays[g]->exp1j(*(other.arrays[g]))) }
    void angle(MImage<float>& other) { MGPULOOP(arrays[g]->angle(*(other.arrays[g]))) }

    void CopyFrom(const Type* other) { MGPULOOP(arrays[g]->CopyFrom(other + offsets[g]);); }
    void CopyTo(Type* other) {
        // ssc_assert(!bBroadcast, "Source copy has multiple instances!");
        ssc_debug(format("Copying MImage to pointer. Broadcast {}", (void*)bBroadcast));
        if (!bBroadcast) {
            MGPULOOP(arrays[g]->CopyTo(other + offsets[g]););
        } else {
            Set(0);
            arrays[0]->CopyTo(other);
        }
    }

    void CopyFrom(const MImage<Type>& other) {
        ssc_assert(other.GetSize() >= GetSize() && other.sizez / other.ngpus >= sizez / ngpus, "Source will not fit!");
        MGPULOOP(arrays[g]->CopyFrom(other.arrays[g][0]););
    }
    void CopyTo(MImage<Type>& other) {
        ssc_assert(other.GetSize() >= GetSize() && other.sizez / other.ngpus >= sizez / ngpus,
                    "Destination will not fit!");
        MGPULOOP(arrays[g]->CopyTo(other.arrays[g][0]););
    }

    void LoadToGPU() { MGPULOOP(arrays[g]->LoadToGPU();); }
    void LoadFromGPU() { MGPULOOP(arrays[g]->LoadFromGPU();); }

    dim3 ShapeBlock() const { return arrays[0]->ShapeBlock(); }
    dim3 ShapeThread() const { return arrays[0]->ShapeThread(); }

    size_t GetSize() const { return size; };

    Type SumGPU() {
        Type s { 0 };
        MGPULOOP(s += arrays[g]->SumGPU(););
        return s;
    }
    Type SumGPU(size_t start, size_t end) {
        Type s { 0 };
        MGPULOOP(s += arrays[g]->SumGPU(start, end););
        return s;
    }
    Type SumCPU() {
        MGPULOOP(arrays[g]->LoadFromGPU(););
        MGPULOOP(cudaDeviceSynchronize(););

        Type s { 0 };
        MGPULOOP(s += arrays[g]->SumCPU(););
        return s;
    }

    void FFTShift1() { MGPULOOP(arrays[g]->FFTShift1();); }
    void FFTShift2() { MGPULOOP(arrays[g]->FFTShift2();); }
    void FFTShift3() { MGPULOOP(arrays[g]->FFTShift3();); }

    Type max() {
        Type m = std::numeric_limits<Type>::lowest();
        MGPULOOP(m = thrust::maximum<Type>()(arrays[g]->max(), m););
        return m;
    }

    float maxAbs2() {
        float m = std::numeric_limits<float>::lowest();
        MGPULOOP(m = thrust::maximum<float>()(arrays[g]->maxAbs2(), m););
        return m;
    }


    template <typename Op>
    void UnaryOperator(Op op) {
        MGPULOOP(arrays[g]->UnaryOperator(op););
    }

    void SwapZ() {
        ssc_assert(ngpus == 1 || (sizez % 4 == 0 && ngpus == 4), "Cant SwapZ!");

        if (ngpus == 4) {
            for (int g = 0; g < ngpus; g++) {
                Set(g);
                cudaDeviceSynchronize();
            }

            for (int g = 0; g < GetN(); g++) {
                Set(g);
                dim3 blk = arrays[g]->ShapeBlock();
                dim3 thr = arrays[g]->ShapeThread();
                KSwapZ_4gpu<<<blk, thr>>>(Ptr(0), Ptr(1), Ptr(2), Ptr(3), g, sizex);
            }
            MGPULOOP(cudaDeviceSynchronize(););
        } else
            arrays[0]->SwapZ();
    }

    /**
     * Makes a sum of the array elements in the "gpu dimension". Stores the result in gpu[0].
     * */
    void ReduceSync() {
        if (ngpus < 2) return;

        for (int s = 1; s < ngpus; s *= 2) {
            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus) {
                    Set(g + s);
                    cudaDeviceSynchronize();
                    Set(g);
                    cudaDeviceSynchronize();

                    arrays[g][0] += arrays[g + s][0];
                }
        }
        Set(0);
        cudaDeviceSynchronize();
    }

    /**
     * Broadcasts the contents of gpu[0] to all other gpus. To be used, e.g., after ReduceSync().
     * */
    void BroadcastSync() {
        if (ngpus < 2) return;

        int potgpus = ngpus - 1;  // convert ngpus to power-of-two
        potgpus |= potgpus >> 1;
        potgpus |= potgpus >> 2;
        potgpus |= potgpus >> 4;
        potgpus += 1;

        for (int s = potgpus / 2; s >= 1; s /= 2) {
            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus) {
                    Set(g);
                    cudaDeviceSynchronize();

                    arrays[g]->CopyTo(arrays[g + s]->gpuptr);
                }
        }

        MGPULOOP(cudaDeviceSynchronize(););
    }

    /**
     * Computes:
     * velocity = beta*velocity + ReduceSync(acc)/ReduceSync(div) - *this
     * *this += velocity
     * Broadcast(this)
     * Meant for ptychography: object(all_gpus) = Sum_gpus(P+*Phi)/Sum_gpus(|P|^2) + beta*momentum
     * */
    void WeightedLerpSync(MImage<Type>& acc, MImage<float>& div, float stepsize, float momentum,
                          Image2D<Type>& velocity, float epsilon) {
        acc.ReduceSync();
        div.ReduceSync();

        Set(0);

        float maxAccum = div.arrays[0]->max();
        auto interpfunction = [maxAccum, epsilon] __device__ __host__(float in) -> float {
            return in * (1 - epsilon) + maxAccum * epsilon;
        };
        div.arrays[0]->UnaryOperator(interpfunction);

        Sync::KWeightedLerp<<<ShapeBlock(), ShapeThread()>>>(Ptr(0), acc.Ptr(0), div.Ptr(0), velocity.gpuptr, size, stepsize, momentum);

        BroadcastSync();
    }

    int __StupidLog2(int s) {
        int t;
        if (s == 1)
            t = 0;
        else if (s == 2)
            t = 1;
        else if (s == 4)
            t = 2;
        else if (s == 8)
            t = 3;
        else if (s == 16)
            t = 4;

        return t;
    }

    /**
     * Same as ReduceSync() except only synchronize unmasked pixels, for performance reasons.
     * */
    void ReduceSyncMasked(Image2D<uint32_t>** syncmask) {
        if (ngpus < 2) return;

        dim3 blk = ShapeBlock();
        dim3 thr = ShapeThread();

        for (int s = 1; s < ngpus; s *= 2) {
            int t = __StupidLog2(s);

            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus) {
                    Set(g + s);
                    cudaDeviceSynchronize();
                    Set(g);
                    cudaDeviceSynchronize();

                    // arrays[g][0] += arrays[g+s][0];
                    Sync::KMaskedSum<Type><<<blk, thr>>>(Ptr(g), Ptr(g + s), size, syncmask[t + g * 4]->gpuptr);
                }
        }
        Set(0);
        cudaDeviceSynchronize();
    }

    /**
     * Same as Broadcast() except only synchronize unmasked pixels, for performance reasons.
     * */
    void BroadcastSyncMasked(Image2D<uint32_t>** syncmask) {
        if (ngpus < 2) return;

        int potgpus = ngpus - 1;  // convert ngpus to power-of-two
        potgpus |= potgpus >> 1;
        potgpus |= potgpus >> 2;
        potgpus |= potgpus >> 4;
        potgpus += 1;

        dim3 blk = ShapeBlock();
        dim3 thr = ShapeThread();

        for (int s = potgpus / 2; s >= 1; s /= 2) {
            int t = __StupidLog2(s);

            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus) {
                    Set(g);
                    cudaDeviceSynchronize();

                    // arrays[g]->CopyTo(arrays[g+s]->gpuptr);
                    Sync::KMaskedBroadcast<Type>
                        <<<blk, thr>>>(Ptr(g + s), Ptr(g), this->size, syncmask[t + g * 4]->gpuptr);
                }
        }

        MGPULOOP(cudaDeviceSynchronize(););
    }

    Image2D<uint32_t>** GenSyncMask() {
        if (ngpus < 2) return nullptr;

        Image2D<uint32_t>** maskmatrix = new Image2D<uint32_t>*[4 * ngpus];
        memset(maskmatrix, 0, 4 * ngpus * sizeof(Image2D<uint32_t>*));

        for (int s = 1; s < ngpus; s *= 2) {
            int t = __StupidLog2(s);

            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus) {
                    Set(g);
                    maskmatrix[t + g * 4] = new Image2D<uint32_t>((size + 31) / 32, 1, 1);
                }
        }

        return maskmatrix;
    }
    void DeleteSyncMask(Image2D<uint32_t>** maskmatrix) {
        if (maskmatrix == nullptr) return;

        for (int s = 1; s < ngpus; s *= 2) {
            int t = __StupidLog2(s);

            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus)
                    if (maskmatrix[t + g * 4]) {
                        Set(g);
                        delete maskmatrix[t + g * 4];
                        maskmatrix[t + g * 4] = nullptr;
                    }
        }

        delete[] maskmatrix;
    }

    void UpdateSyncMask(Image2D<uint32_t>** maskmatrix, float thresh) {
        if (ngpus < 2) return;

        dim3 blk = dim3((this->size + 31) / 32, 1, 1);
        dim3 thr = dim3(32, 1, 1);

        for (int s = 1; s < ngpus; s *= 2) {
            int t = __StupidLog2(s);

            for (int g = 0; g < ngpus; g += 2 * s)
                if (g + s < ngpus) {
                    Set(g);
                    maskmatrix[t + g * 4]->SetGPUToZero();
                    cudaDeviceSynchronize();

                    Set(g + s);
                    cudaDeviceSynchronize();

                    Sync::KSetMask<Type><<<blk, thr>>>(maskmatrix[t + g * 4]->gpuptr, Ptr(g + s), this->size, thresh);
                }
        }

        MGPULOOP(cudaDeviceSynchronize(););
    }

    /**
     * Same as WeightedLerpSyncMasked() except only synchronize unmasked pixels, for performance reasons.
     * */
    void WeightedLerpSyncMasked(MImage<Type>& acc, MImage<float>& div, float lerp, Image2D<uint32_t>** syncmask) {
        acc.ReduceSyncMasked(syncmask);
        div.ReduceSyncMasked(syncmask);

        Set(0);
        Sync::KWeightedLerp<<<ShapeBlock(), ShapeThread()>>>(Ptr(0), acc.Ptr(0), div.Ptr(0), size, lerp);

        BroadcastSyncMasked(syncmask);
    }

    std::vector<cufftHandle> GetFFTPlan1D(int batch = -1) {
        std::vector<cufftHandle> plans;
        MGPULOOP(plans.push_back(arrays[g]->GetFFTPlan1D(batch)));
        return plans;
    }
    std::vector<cufftHandle> GetFFTPlan2D() {
        std::vector<cufftHandle> plans;
        MGPULOOP(plans.push_back(arrays[g]->GetFFTPlan2D()));
        return plans;
    }
    std::vector<cufftHandle> GetFFTPlan3D() {
        std::vector<cufftHandle> plans;
        MGPULOOP(plans.push_back(arrays[g]->GetFFTPlan3D()));
        return plans;
    }
    void FFT(std::vector<cufftHandle> plans) { MGPULOOP(arrays[g]->FFT(plans[g])); }
    void IFFT(std::vector<cufftHandle> plans) { MGPULOOP(arrays[g]->IFFT(plans[g])); }

/**
 * Todo: Parallel execution of blocking calls such as norm and sumgpu.
 * */
#ifdef THRUST_INCLUDED
    float Norm2() {
        float n2 = 0;
        MGPULOOP(n2 += arrays[g]->Norm2());
        return n2;
    }
    template <typename T2>
    Type dot(MImage<T2>& other) {
        Type n2 = 0;
        MGPULOOP(n2 += arrays[g]->dot(other.arrays[g][0]));
        return n2;
    }
#endif

    /**
     * Returns reference array containing the Z-index specified
     * */
    MImage<Type>* SliceZ(size_t index) const {
        MImage<Type>* myimage = new MImage<Type>(sizex, sizey, 1, true, gpuindices, MemoryType::ENoAlloc);
        for (int g = 0; g < GetN(); g++) myimage->arrays[g]->gpuptr = arrays[g]->gpuptr + sizex * sizey * index;
        return myimage;
    };
};

typedef MImage<float> rMImage;
typedef MImage<complex> cMImage;
typedef MImage<complex16> hcMImage;


#endif
