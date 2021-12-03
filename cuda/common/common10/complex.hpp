// @Author: Giovanni L. Baraldi
// File contains implementations for f32 and f16 complex numbers and some useful operations.

#ifndef _COMPLEX24_H
#define _COMPLEX24_H

#ifdef __CUDACC__
 #define restrict __restrict__
#else
 #define restrict
#endif

#include "cuComplex.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#define __hevice __host__ __device__ inline

struct complex: public cuComplex
{
        complex() = default;
        __host__ __device__ complex(float f){ x=f; y=0; };
        __host__ __device__ complex(float f1, float f2){ x=f1; y=f2; };
        __hevice complex& operator+=(const complex& other){ x += other.x; y += other.y; return *this; };
        __hevice complex& operator-=(const complex& other){ x -= other.x; y -= other.y; return *this; };
        __hevice complex& operator*=(const complex& other){ float tx=x; float ox = other.x; x = x*other.x - y*other.y; y = tx*other.y + y*ox; return *this; };
        __hevice complex& operator/=(const complex& other){ float ab = 1.0f/other.abs2(); *this *= other.conj()*ab; return *this; };

        __hevice complex operator+(const complex& other) const { complex cc; cc.x = x + other.x; cc.y = y + other.y; return cc; };
        __hevice complex operator-(const complex& other) const { complex cc; cc.x = x - other.x; cc.y = y - other.y; return cc; };
        __hevice complex operator*(const complex& other) const { complex cc; cc.x = x*other.x - y*other.y; cc.y = x*other.y + y*other.x; return cc; };
        __hevice complex operator/(const complex& other) const { complex cc=*this; return cc/other; };

        __hevice complex& operator+=(float other){ x += other; return *this; };
        __hevice complex& operator-=(float other){ x -= other; return *this; };
        __hevice complex& operator*=(float other){ x *= other; y *= other; return *this; };
        __hevice complex& operator/=(float other){ x /= other; y /= other; return *this; };

        __hevice complex operator+(float other) const { complex cc; cc.x = x + other; cc.y = y; return cc; };
        __hevice complex operator-(float other) const { complex cc; cc.x = x - other; cc.y = y; return cc; };
        __hevice complex operator*(float other) const { complex cc; cc.x = x * other; cc.y = y * other; return cc; };
        __hevice complex operator/(float other) const { complex cc; cc.x = x / other; cc.y = y / other; return cc; };

        __hevice void times_pluss_i(){ float interm = x; x=-y; y=interm; };
        __hevice void times_minus_i(){ float interm = x; x=y; y=-interm; };

        __hevice float abs2() const { return x*x + y*y; };
        __hevice float abs() const 
        { 
                #ifdef __CUDACC__
                        return hypotf(x,y); 
                #else
                        return sqrtf(abs2()); 
                #endif
        };
        __hevice complex conj() const { complex cc; cc.x = x; cc.y = -y; return cc; }

        __device__ float angle() const { return atan2f(y,x); }
        __device__ static complex exp1j(float a){ complex w; __sincosf(a, &w.y, &w.x); return w; };

        __device__ complex(const struct complex16& c16);
        __host__ __device__ const complex& ToC32() const { return *this; };
};
inline __device__ void atomicAdd(complex* ptr, const complex& val){ atomicAdd((float*)ptr, val.x); atomicAdd(1+(float*)ptr, val.y); }; 


#if __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
struct complex16: public __half2
{
	complex16() = default;
	__device__ explicit complex16(half f){ *this = __halves2half2(f,0); };
	__device__ explicit complex16(float f1, float f2){ *this =__floats2half2_rn(f1,f2); };
	__device__ explicit complex16(float f){ *this =__floats2half2_rn(f,0); };
	__device__ explicit complex16(const complex& c){ *this =__float22half2_rn(c); };
	__device__ complex16(half2 f2){ *(half2*)this = f2; };
	
	__device__ complex16& operator+=(const complex16& other){ *this = __hadd2(*this,other); return *this; };
	__device__ complex16& operator-=(const complex16& other){ *this = __hsub2(*this,other); return *this; };
	__device__ complex16& operator*=(const complex16& other){ half mix = __hfma(x,other.x,-__hmul(y,other.y)); y = __hfma(y,other.x,__hmul(x,other.y)); x = mix; return *this; };
	//__hevice complex16& operator/=(const complex16& other){ half ab = __float2half(1.0f)/other.abs2(); *this *= other.conj()*ab; return *this; };
	
	__device__ complex16 operator+(const complex16& other) const { return __hadd2(*this,other); };
	__device__ complex16 operator-(const complex16& other) const { return __hsub2(*this,other); };
	__device__ complex16 operator*(const complex16& other) const { complex16 cc; cc.x = __hfma(x,other.x,-__hmul(y,other.y)); cc.y = __hfma(y,other.x,__hmul(x,other.y)); return cc; };
	//__hevice complex16 operator/(const complex16& other) const { complex16 cc=*this; return cc/other; };
	
	__device__ complex16& operator*=(half other){ *this = __hmul2(*this,__half2half2(other)); return *this; };
	__device__ complex16 operator*(half other) const { return __hmul2(*this,__half2half2(other)); };

	__device__ complex16& operator*=(float other){ *this = __hmul2(*this,__half2half2(__float2half(other))); return *this; };
	__device__ complex16 operator*(float other) const { return __hmul2(*this,__half2half2(__float2half(other))); };

	//__hevice complex16& operator/=(half other){ x /= other; y /= other; return *this; };
	
        __device__ float angle() const { return this->ToC32().angle(); }
        __device__ static complex16 exp1j(float a){ return complex16(complex::exp1j(a)); };


	__device__ void times_pluss_i(){ half interm = x; x=-y; y=interm; };
	__device__ void times_minus_i(){ half interm = x; x=y; y=-interm; };
	
	__device__ half abs2() const { return __hfma(x,x,__hmul(y,y)); };
	__device__ complex16 conj() const { complex16 cc; cc.x = x; cc.y = -y; return cc; }
	
	__device__ complex ToC32() const{ complex c; *(float2*)&c = __half22float2(*this); return c; }
};

inline __device__ complex::complex(const complex16& c16){ *(float2*)this = __half22float2(c16); };
#endif

#endif