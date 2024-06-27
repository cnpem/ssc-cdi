// @Author: Giovanni L. Baraldi, Gilberto Martinez
// File contains kernels for the most basic operations

#ifndef _OPERATIONS_H
#define _OPERATIONS_H

#ifdef __CUDACC__
#define restrict __restrict__
#else
#define restrict
#endif

#include <string>

#include "complex.hpp"

//#define SyncDebug cudaDeviceSynchronize()
#define SyncDebug

inline __device__ int iabs(int n) { return n < 0 ? -n : n; }

static __device__ __host__ float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }

static __device__ half clamp(half f, half a, half b) { return fmaxf(a, fminf(f, b)); }

static __device__ __host__ half clamp(half f, float a = -32768.0f, float b = 32768.0f) {
    return fmaxf(__float2half(a), fminf(f, __float2half(b)));
}

#if __CUDA_ARCH__ >= 530
static __device__ complex16 clamp(complex16 f, float a, float b) {
    return complex16(clamp(f.x, a, b), clamp(f.y, a, b));
}
#endif

static __device__ __host__ complex clamp(const complex& f, float a, float b) {
    return complex(clamp(f.x, a, b), clamp(f.y, a, b));
}
static __device__ __host__ complex clamp(const complex& f, const complex& a, const complex& b) {
    return complex(clamp(f.x, a.x, b.x), clamp(f.y, a.y, b.y));
}

template <typename Type>
inline __device__ __host__ Type sq(const Type& x) {
    return x * x;
}

inline __device__ complex exp1j(float f) {
    complex c;
    __sincosf(f, &c.y, &c.x);
    return c;
}

struct EType{

	enum class TypeEnum 
	{
		INVALID = 0,
		UINT8   = 1,
		UINT16  = 2,
		INT32   = 3,
		HALF    = 4,
		FLOAT32 = 5,
		DOUBLE  = 6,
		NUM_ENUMS
	};

	EType() = default;
	EType(TypeEnum etype): type(etype) {};

	static size_t Size(TypeEnum datatype);

	size_t Size() const { return Size(type); };

	static std::string String(TypeEnum type);

	std::string String() { return String(type); };

	static EType Type(const std::string& nametype);

	TypeEnum type = TypeEnum::INVALID;
};

namespace BasicOps {
inline __device__ size_t GetIndex() {
    return threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
}

template <typename Type1, typename Type2>
__global__ void KB_Add(Type1* a, const Type2* b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void KB_Sub(Type1* a, const Type2* b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void KB_Mul(Type1* a, const Type2* b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void KB_Div(Type1* a, const Type2* b, size_t size, size_t size2);

template <typename Type1, typename Type2 = Type1>
__global__ void KB_Add(Type1* a, Type2 n, size_t size);

template <typename Type1, typename Type2 = Type1>
__global__ void KB_Sub(Type1* a, Type2 n, size_t size);

template <typename Type1, typename Type2 = Type1>
__global__ void KB_Mul(Type1* a, Type2 n, size_t size);

template <typename Type1, typename Type2 = Type1>
__global__ void KB_Div(Type1* a, Type2 n, size_t size);

template <typename Type>
__global__ void KB_Log(Type* a, size_t size);

template <typename Type>
__global__ void KB_Exp(Type* a, size_t size, bool bNeg);

template <typename Type>
__global__ void KB_Clamp(Type* a, const Type b, const Type c, size_t size);

template <typename Type>
__global__ void KB_log1j(float* out, const Type* in, size_t size);

template <typename Type>
__global__ void KB_exp1j(Type* out, const float* in, size_t size);

template <>
__global__ void KB_log1j<complex>(float* out, const complex* in, size_t size);

template <>
__global__ void KB_exp1j<complex>(complex* out, const float* in, size_t size);

template <typename Type>
__global__ void KB_Power(Type* a, float P, size_t size);

template <>
__global__ void KB_Power<float>(float* out, float P, size_t size);

template <>
__global__ void KB_Power<complex>(complex* out, float P, size_t size);

template <typename Type>
__global__ void KB_ABS2(float* out, Type* a, size_t size);

template <>
__global__ void KB_ABS2<complex>(float* out, complex* in, size_t size);

template <>
__global__ void KB_ABS2<float>(float* out, float* in, size_t size);

template <typename Type2, typename Type1>
__global__ void KConvert(Type2* out, Type1* in, size_t size, float threshold);

#if __CUDA_ARCH__ >= 530
template <>
__global__ void KConvert<complex, complex16>(complex* out, complex16* in, size_t size, float threshold);
template <>
__global__ void KConvert<complex16, complex>(complex16* out, complex* in, size_t size, float threshold);
#endif

template <typename Type>
inline __device__ Type convert(float var, float threshold) {
    return var;
}

/*we divide by 1024 just for normalization*/

#if __CUDA_ARCH__ >= 530
template <>
inline __device__ half convert(float var, float threshold) {
    return __float2half(var);
}
#endif

template <>
inline __device__ uint16_t convert(float var, float threshold) {
    var = __saturatef(var / (2 * threshold) + 0.5f);
    return ((1 << 16) - 1) * var;
}

template <>
inline __device__ uint8_t convert(float var, float threshold) {
    var = __saturatef(var / (2 * threshold) + 0.5f);
    return ((1 << 8) - 1) * var;
}

// float just has 24 bits (e.g 1 << 24)
template <>
inline __device__ int convert(float var, float threshold) {
    var = __saturatef(var / (2 * threshold) + 0.5f);
    return ((1 << 24) - 1) * var;
}

template <typename Type>
inline __device__ void set_pixel(void* recon, float var, int i, int j, int wdI, float threshold) {
    ((Type*)recon)[j * wdI + i] = convert<Type>(var, threshold);
}

inline __device__ void set_pixel(void* recon, float var, int i, int j, int wdI, float threshold,
                                 EType::TypeEnum raftDatatype) {
    switch (raftDatatype) {
        case EType::TypeEnum::UINT8:
            set_pixel<uint8_t>(recon, var, i, j, wdI, threshold);
            break;
        case EType::TypeEnum::UINT16:
            set_pixel<uint16_t>(recon, var, i, j, wdI, threshold);
            break;
        case EType::TypeEnum::INT32:
            set_pixel<int>(recon, var, i, j, wdI, threshold);
            break;
        case EType::TypeEnum::FLOAT32:
            set_pixel<float>(recon, var, i, j, wdI, threshold);
            break;
#if __CUDA_ARCH__ >= 530
        case EType::TypeEnum::HALF:
            set_pixel<half>(recon, var, i, j, wdI, threshold);
            break;
#endif
    }
}

template <typename Type>
__global__ void KFFTshift1(Type* img, size_t sizex, size_t sizey);

template <typename Type>
__global__ void KFFTshift2(Type* img, size_t sizex, size_t sizey);

template <typename Type>
__global__ void KFFTshift3(Type* img, size_t sizex, size_t sizey, size_t sizez);


}  // namespace BasicOps

namespace Reduction {
/*
inline __device__ void KSharedReduce32(float* intermediate)
{
        if(threadIdx.x < 16)
        {
                for(int group = 16; group > 0; group /= 2)
                {
                        if(threadIdx.x < group)
                        {
                                intermediate[threadIdx.x] += intermediate[threadIdx.x+group];
                                __syncwarp((1<<group)-1);
                        }
                }
        }
}

inline __device__ void KSharedReduce(float* intermediate, int size)
{
        if(threadIdx.x < 32)
        {
                float s = 0;
                for(int idx = threadIdx.x; idx < size; idx += 32)
                        s += intermediate[threadIdx.x];

                intermediate[threadIdx.x] = s;
                __syncwarp();
                KSharedReduce32(intermediate);
        }
}

inline __global__ void KGlobalReduce(float* out, const float* in, size_t size)
{
        __shared__ float intermediate[32];
        if(threadIdx.x<32)
                intermediate[threadIdx.x] = 0;
        __syncthreads();

        float mine = 0;

        for(size_t index = threadIdx.x + blockIdx.x*blockDim.x; index < size; index += blockDim.x*gridDim.x)
                mine += in[index];

        atomicAdd(intermediate + threadIdx.x%32, mine);

        __syncthreads();

        KSharedReduce32(intermediate);
        if(threadIdx.x==0)
                out[blockIdx.x] = intermediate[0];
} */
template <typename Type>
inline __device__ void KSharedReduce32(Type* intermediate) {
    if (threadIdx.x < 16) {
        for (int group = 16; group > 0; group /= 2) {
            if (threadIdx.x < group) {
                intermediate[threadIdx.x] += intermediate[threadIdx.x + group];
                __syncwarp((1 << group) - 1);
            }
        }
    }
}

template <typename Type>
inline __device__ void KSharedReduce(Type* intermediate, int size) {
    if (threadIdx.x < 32) {
        Type s {0};
        for (int idx = threadIdx.x; idx < size; idx += 32) s += intermediate[threadIdx.x];

        intermediate[threadIdx.x] = s;
        __syncwarp();
        KSharedReduce32(intermediate);
    }
}

template <typename Type>
__global__ void KGlobalReduce(Type* out, const Type* in, size_t size);

}  // namespace Reduction

namespace Sync {
template <typename Type>
__global__ void KWeightedLerp(Type* val, const Type* acc, const float* div, 
            Type* velocity, size_t size, float stepsize, float momentum);

template <typename Type>
__global__ void KMaskedSum(Type* cval, const Type* acc, size_t size, const uint32_t* mask2);

template <typename Type>
__global__ void KMaskedBroadcast(Type* cval, const Type* acc, size_t size, const uint32_t* mask2);

template <typename Type>
__global__ void KSetMask(uint32_t* mask, const Type* value, size_t size, float thresh);

template <>
inline __global__ void KSetMask<float>(uint32_t* mask, const float* fval, size_t size, float thresh);
}  // namespace Sync

namespace Filters {
inline __device__ int addrclamp(int x, int a, int b) { return (x < a) ? a : ((x >= b) ? (b - 1) : x); }
inline __device__ float SoftThresh(float v, float mu) { return fmaxf(fminf(v, mu), -mu); }
inline __device__ complex SoftThresh(complex v, float mu) { return complex(SoftThresh(v.x, mu), SoftThresh(v.y, mu)); }

constexpr int bd = 4;

template <typename Type>
__forceinline__ __device__ void DTV2D(Type shout[][16 + 2 * bd], const Type shin[][16 + 2 * bd],
                                      const Type sh0[][16 + 2 * bd], float mu) {
    for (int lidy = threadIdx.y + 1; lidy < 16 + 2 * bd - 1; lidy += blockDim.y)
        for (int lidx = threadIdx.x + 1; lidx < 16 + 2 * bd - 1; lidx += blockDim.x) {
            Type center = shin[lidy][lidx];

            Type coef1 = SoftThresh(shin[lidy][lidx + 1] - center, mu);
            Type coef2 = SoftThresh(shin[lidy][lidx - 1] - center, mu);
            Type coef3 = SoftThresh(shin[lidy + 1][lidx] - center, mu);
            Type coef4 = SoftThresh(shin[lidy - 1][lidx] - center, mu);

            Type coef5 = SoftThresh(shin[lidy + 1][lidx + 1] - center, mu);
            Type coef6 = SoftThresh(shin[lidy + 1][lidx - 1] - center, mu);
            Type coef7 = SoftThresh(shin[lidy - 1][lidx + 1] - center, mu);
            Type coef8 = SoftThresh(shin[lidy - 1][lidx - 1] - center, mu);

            shout[lidy][lidx] = center + (coef1 + coef2 + coef3 + coef4) * 0.2f +
                                (coef5 + coef6 + coef7 + coef8) * 0.1f + (sh0[lidy][lidx] - shin[lidy][lidx]) * 0.2f;
        }
}

/**
 * If gradout == nullptr, store img=tv(img). Else, store gradout += tv(img)-img.
 * */
template <typename Type>
__global__ void KTV2D(Type* img, float mu, dim3 shape, Type* gradout);

template <typename Type>
__forceinline__ __device__ void DMappedTV(Type shout[][16 + 2 * bd], const Type shin[][16 + 2 * bd],
                                          const Type sh0[][16 + 2 * bd], const float mumap[][16 + 2 * bd]) {
    for (int lidy = threadIdx.y + 1; lidy < 16 + 2 * bd - 1; lidy += blockDim.y)
        for (int lidx = threadIdx.x + 1; lidx < 16 + 2 * bd - 1; lidx += blockDim.x) {
            Type center = shin[lidy][lidx];
            float mu = mumap[lidy][lidx];

            Type coef1 = SoftThresh(shin[lidy][lidx + 1] - center, mu);
            Type coef2 = SoftThresh(shin[lidy][lidx - 1] - center, mu);
            Type coef3 = SoftThresh(shin[lidy + 1][lidx] - center, mu);
            Type coef4 = SoftThresh(shin[lidy - 1][lidx] - center, mu);

            Type coef5 = SoftThresh(shin[lidy + 1][lidx + 1] - center, mu);
            Type coef6 = SoftThresh(shin[lidy + 1][lidx - 1] - center, mu);
            Type coef7 = SoftThresh(shin[lidy - 1][lidx + 1] - center, mu);
            Type coef8 = SoftThresh(shin[lidy - 1][lidx - 1] - center, mu);

            shout[lidy][lidx] = center + (coef1 + coef2 + coef3 + coef4) * 0.2f +
                                (coef5 + coef6 + coef7 + coef8) * 0.1f + (sh0[lidy][lidx] - shin[lidy][lidx]) * 0.2f;
        }
}

template <typename Type>
__global__ void KMappedTV(Type* grad, const Type* imgin, float phi, dim3 shape, const float* lambdamap);

}  // namespace Filters

#endif
