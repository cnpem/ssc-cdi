#include <cmath>
#include <cstddef>
#include <string>
#include "complex.hpp"
#include "operations.hpp"
#include "types.hpp"

/*======================================================================*/
/* struct EType (in 'inc/common/operations.hpp') functions definitions */

size_t EType::Size(EType::TypeEnum datatype)
{
	static const size_t datasizes[] = {0, 1, 2, 4, 2, 4, 8};
	return datasizes[static_cast<int>(datatype)];
}

std::string EType::String(EType::TypeEnum type)
{
	static const std::string datanames[] = {"INVALID",
											"UINT8",
											"UINT16",
											"INT32",
											"HALF",
											"FLOAT32",
											"DOUBLE"};

	return datanames[static_cast<int>(type)];
}

EType EType::Type(const std::string &nametype)
{
	EType etype;

	for (int i = 0; i < static_cast<int>(EType::TypeEnum::NUM_ENUMS); i++)
		if (nametype == EType::String(static_cast<EType::TypeEnum>(i)))
			etype.type = static_cast<EType::TypeEnum>(i);

	return etype;
}

/*============================================================================*/
/* namespace BasicOps (in 'inc/common/operations.hpp') functions definitions */

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Add(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();

	if (index >= size)
		return;

	a[index] += b[index % size2];
}


template
__global__ void BasicOps::KB_Add(complex *a, const complex *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Add(complex *a, const float *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Add(float *a, const float *b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Sub(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] -= b[index % size2];
}

template
__global__ void BasicOps::KB_Sub(float *a, const float *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Sub(complex *a, const complex *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Sub(complex *a, const float *b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Mul(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] *= b[index % size2];
}

template
__global__ void BasicOps::KB_Mul(float *a, const float *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Mul(complex *a, const complex *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Mul(complex *a, const float *b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Div(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] /= b[index % size2];
}

template
__global__ void BasicOps::KB_Div(float *a, const float *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Div(complex *a, const complex *b, size_t size, size_t size2);

template
__global__ void BasicOps::KB_Div(complex *a, const float *b, size_t size, size_t size2);

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Add(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] += n;
}

template
__global__ void BasicOps::KB_Add(complex *a, const float n, size_t size);

template
__global__ void BasicOps::KB_Add(complex *a, const complex n, size_t size);

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Sub(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] -= n;
}

template
__global__ void BasicOps::KB_Sub(complex *a, const float n, size_t size);

template
__global__ void BasicOps::KB_Sub(complex *a, const complex n, size_t size);

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Mul(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] *= n;
}

template
__global__ void BasicOps::KB_Mul(float *a, const float n, size_t size);

template
__global__ void BasicOps::KB_Mul(complex *a, const float n, size_t size);

template
__global__ void BasicOps::KB_Mul(complex *a, const complex n, size_t size);


template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Div(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	if (index >= size)
		return;

	a[index] /= n;
}

template
__global__ void BasicOps::KB_Div(float *a, const float n, size_t size);

template
__global__ void BasicOps::KB_Div(complex *a, const float n, size_t size);

template
__global__ void BasicOps::KB_Div(complex *a, const complex n, size_t size);

template <typename Type>
__global__ void BasicOps::KB_Log(Type *a, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] = logf(fmaxf(a[index], 1E-10f));
}

template <typename Type>
__global__ void BasicOps::KB_Exp(Type *a, size_t size, bool bNeg)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	float var = a[index];
	a[index] = expf(bNeg ? (-var) : var);
}

template <typename Type>
__global__ void BasicOps::KB_Clamp(Type *a, const Type b, const Type c, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] = clamp(a[index], b, c);
}

template <>
__global__ void BasicOps::KB_log1j<complex>(float *out, const complex *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = in[index].angle();
}

template <>
__global__ void BasicOps::KB_exp1j<complex>(complex *out, const float *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex::exp1j(in[index]);
}

template <>
__global__ void BasicOps::KB_Power<float>(float *out, float P, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = expf(logf(fmaxf(fabsf(out[index]), 1E-25f)) * P);
}

template <>
__global__ void BasicOps::KB_Power<complex>(complex *out, float P, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex::exp1j(out[index].angle() * P) * expf(logf(fmaxf(out[index].abs(), 1E-25f)) * P);
}

template <>
__global__ void BasicOps::KB_ABS2<complex>(float *out, complex *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = in[index].abs2();
}

template <>
__global__ void BasicOps::KB_ABS2<float>(float *out, float *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = in[index] * in[index];
}


template <typename Type2, typename Type1>
__global__ void BasicOps::KConvert(Type2 *out, Type1 *in, size_t size, float threshold)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	float res = (float)in[index];

	if (std::is_same<Type2, __half>::value == true)
		res *= 1024.0f;
	else if (std::is_floating_point<Type2>::value == false)
		res = fminf(res / threshold, 1.0f) * ((1UL << (8 * (sizeof(Type2) & 0X7))) - 1);

	out[index] = Type2(res);
}

template <>
__global__ void BasicOps::KConvert<complex, complex16>(complex *out, complex16 *in, size_t size, float threshold)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex(in[index]);
}
template <>
__global__ void BasicOps::KConvert<complex16, complex>(complex16 *out, complex *in, size_t size, float threshold)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex16(in[index]);
}


template <typename Type>
__global__ void BasicOps::KFFTshift1(Type *img, size_t sizex, size_t sizey)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = blockIdx.z;

	if (idx < sizex / 2 && idy < sizey)
	{
		size_t index1 = idz * sizex * sizey + idy * sizex + idx;
		size_t index2 = idz * sizex * sizey + idy * sizex + idx + sizex / 2;

		Type temp = img[index1];
		img[index1] = img[index2];
		img[index2] = temp;
	}
}

template
__global__ void BasicOps::KFFTshift1(complex *img, size_t sizex, size_t sizey);

template
__global__ void BasicOps::KFFTshift1(float *img, size_t sizex, size_t sizey);


template <typename Type>
__global__ void BasicOps::KFFTshift2(Type *img, size_t sizex, size_t sizey)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = blockIdx.z;

	if (idx < sizex && idy < sizey / 2)
	{
		size_t index1 = idz * sizex * sizey + idy * sizex + idx;
		size_t index2 = idz * sizex * sizey + (idy + sizey / 2) * sizex + (idx + sizex / 2) % sizex;

		Type temp = img[index1];
		img[index1] = img[index2];
		img[index2] = temp;
	}
}

template
__global__ void BasicOps::KFFTshift2(complex *img, size_t sizex, size_t sizey);

template
__global__ void BasicOps::KFFTshift2(float *img, size_t sizex, size_t sizey);

template <typename Type>
__global__ void BasicOps::KFFTshift3(Type *img, size_t sizex, size_t sizey, size_t sizez)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = blockIdx.z;

	if (idx < sizex && idy < sizey / 2)
	{
		size_t index1 = idz * sizex * sizey + idy * sizex + idx;
		size_t index2 = ((idz + sizez / 2) % sizez) * sizex * sizey + (idy + sizey / 2) * sizex + (idx + sizex / 2) % sizex;

		Type temp = img[index1];
		img[index1] = img[index2];
		img[index2] = temp;
	}
}


template
__global__ void BasicOps::KFFTshift3(complex *img, size_t sizex, size_t sizey, size_t sizez);

template
__global__ void BasicOps::KFFTshift3(float *img, size_t sizex, size_t sizey, size_t sizez);

/*=============================================================================*/
/* namespace Reduction (in 'inc/common/operations.hpp') functions definitions */

template <typename Type>
__global__ void Reduction::KGlobalReduce(Type *out, const Type *in, size_t size)
{
	__shared__ Type intermediate[32];
	if (threadIdx.x < 32)
		intermediate[threadIdx.x] = 0;
	__syncthreads();

	Type mine = 0;

	for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x)
		mine += in[index];

	atomicAdd(intermediate + threadIdx.x % 32, mine);

	__syncthreads();

	Reduction::KSharedReduce32(intermediate);
	if (threadIdx.x == 0)
		out[blockIdx.x] = intermediate[0];
}

/*========================================================================*/
/* namespace Sync (in 'inc/commons/operations.hpp') functions definitions */

template <typename Type>
__global__ void Sync::KWeightedLerp(Type* val, const Type* acc, const float* div,
            Type* velocity, size_t size, float stepsize, float momentum) {
    size_t index = BasicOps::GetIndex();
    if (index >= size) return;

    /*Type weighed = acc[index] / (div[index]+1E-10f);
    Type biasedgrad = weighed - val[index] + velocity[index]*momentum; */
    Type biasedgrad = acc[index] / div[index] + velocity[index] * momentum;

    velocity[index] = biasedgrad;
    val[index] += biasedgrad * stepsize;
}

template
__global__ void Sync::KWeightedLerp<complex>(complex* val, const complex* acc, const float* div,
        complex* velocity, size_t size, float stepsize, float momentum);

template <typename Type>
__global__ void Sync::KMaskedSum(Type *cval, const Type *acc, size_t size, const uint32_t *mask2)
{
	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	uint32_t mask = mask2[index / 32];
	bool value = (mask >> (index & 0x1F)) & 0x1;

	if (value)
		cval[index] += acc[index];
}

template <typename Type>
__global__ void Sync::KMaskedBroadcast(Type *cval, const Type *acc, size_t size, const uint32_t *mask2)
{
	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	uint32_t mask = mask2[index / 32];
	bool value = (mask >> (index & 0x1F)) & 0x1;

	if (value)
		cval[index] = acc[index];
}

template <typename Type>
__global__ void Sync::KSetMask(uint32_t *mask, const Type *value, size_t size, float thresh);

template <>
__global__ void Sync::KSetMask<float>(uint32_t *mask, const float *fval, size_t size, float thresh)
{
	__shared__ uint32_t shvalue[1];
	if (threadIdx.x < 32)
		shvalue[threadIdx.x] = 0;

	__syncthreads();

	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	uint32_t value = (fval[index] > thresh) ? 1 : 0;
	value = value << threadIdx.x;

	atomicOr(shvalue, value);

	__syncthreads();
	if (threadIdx.x == 0)
		mask[index / 32] = shvalue[0];
}

/*========================================================================*/
/* namespace Filters (in 'inc/common/operations.hpp') functions definitions */

/**
 * If gradout == nullptr, store img=tv(img). Else, store gradout += tv(img)-img.
 * */
template <typename Type>
__global__ void Filters::KTV2D(Type* img, float mu, dim3 shape, Type* gradout) {
    __shared__ Type shimgf[16 + 2 * bd][16 + 2 * bd];
    __shared__ Type shimgi[16 + 2 * bd][16 + 2 * bd];
    __shared__ Type shimg0[16 + 2 * bd][16 + 2 * bd];

    for (int lidy = threadIdx.y; lidy < 16 + 2 * bd; lidy += blockDim.y)
        for (int lidx = threadIdx.x; lidx < 16 + 2 * bd; lidx += blockDim.x) {
            const int gidx = blockDim.x * blockIdx.x + lidx;
            const int gidy = blockDim.y * blockIdx.y + lidy;

            Type value = img[blockIdx.z * shape.x * shape.y + Filters::addrclamp(gidy - bd, 0, shape.y) * shape.x +
                             Filters::addrclamp(gidx - bd, 0, shape.x)];
            shimgi[lidy][lidx] = value;
            shimgf[lidy][lidx] = value;
            shimg0[lidy][lidx] = value;
        }

    for (int iter = 0; iter < 10; iter++) {
        __syncthreads();
        Filters::DTV2D<Type>(shimgf, shimgi, shimg0, mu);
        __syncthreads();
        Filters::DTV2D<Type>(shimgi, shimgf, shimg0, mu);
    }
    __syncthreads();

    const int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gidx < shape.x && gidy < shape.y) {
        size_t index = blockIdx.z * shape.y * shape.x + gidy * shape.x + gidx;
        if (gradout == nullptr)
            img[index] = shimgi[threadIdx.y + bd][threadIdx.x + bd];
        else
            gradout[index] += shimgi[threadIdx.y + bd][threadIdx.x + bd] - shimg0[threadIdx.y + bd][threadIdx.x + bd];
    }
}

template <typename Type>
__global__ void Filters::KMappedTV(Type* grad, const Type* imgin, float phi, dim3 shape, const float* lambdamap) {
    __shared__ Type shimgf[16 + 2 * bd][16 + 2 * bd];
    __shared__ Type shimgi[16 + 2 * bd][16 + 2 * bd];
    __shared__ Type shimg0[16 + 2 * bd][16 + 2 * bd];
    __shared__ float shlambda[16 + 2 * bd][16 + 2 * bd];

    for (int lidy = threadIdx.y; lidy < 16 + 2 * bd; lidy += blockDim.y)
        for (int lidx = threadIdx.x; lidx < 16 + 2 * bd; lidx += blockDim.x) {
            const int gidx = blockDim.x * blockIdx.x + lidx;
            const int gidy = blockDim.y * blockIdx.y + lidy;

            Type value = imgin[blockIdx.z * shape.x * shape.y + Filters::addrclamp(gidy - bd, 0, shape.y) * shape.x +
                               Filters::addrclamp(gidx - bd, 0, shape.x)];
            shimgi[lidy][lidx] = value;
            shimgf[lidy][lidx] = value;
            shimg0[lidy][lidx] = value;

            float sp2 = lambdamap[blockIdx.z * shape.x * shape.y + Filters::addrclamp(gidy - bd, 0, shape.y) * shape.x +
                                  Filters::addrclamp(gidx - bd, 0, shape.x)];
            shlambda[lidy][lidx] = phi / (sp2 + phi / value.abs());
        }

    __syncthreads();

    for (int iter = 0; iter < 10; iter++) {
        __syncthreads();
        Filters::DMappedTV<Type>(shimgf, shimgi, shimg0, shlambda);
        __syncthreads();
        Filters::DMappedTV<Type>(shimgi, shimgf, shimg0, shlambda);
    }
    __syncthreads();

    const int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gidx < shape.x && gidy < shape.y)
        grad[blockIdx.z * shape.y * shape.x + gidy * shape.x + gidx] +=
            shimgi[threadIdx.y + bd][threadIdx.x + bd] - shimg0[threadIdx.y + bd][threadIdx.x + bd];
}


template
__global__ void Filters::KMappedTV(complex* grad, const complex* imgin, float phi, dim3 shape, const float* lambdamap);

