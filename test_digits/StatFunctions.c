#include "StatFunctions.h"
#include <float.h>
#include <math.h>

#ifndef assert
	#include <assert.h>
#endif

#define LOCAL_CONCAT_(x, y) x##_##y
#define LOCAL_CONCAT(x, y) LOCAL_CONCAT_(x, y)
#define LOCAL(name) LOCAL_CONCAT(name, __LINE__)

/// Compute numpy.diff() at the specified index.
#define DiffFromNextToCurrentUsingAccessor(p, n, i, result) \
	do \
	{ \
		assert(p); \
		assert(n); \
		assert((i) < (n)); \
		*(result) = GetItemAtIndex((p), (n), (i) + 1) - GetItemAtIndex((p), (n), (i)); \
	} while (0)

static float DiffFromNextToCurrent(const float* const p, const size_t n, const size_t i)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	DiffFromNextToCurrentUsingAccessor(p, n, i, &result);
#undef GetItemAtIndex
	return result;
}

/// Compute numpy.gradient() at the specified index.
#define GradientAtIndexUsingAccessor(p, n, i, result) \
	do \
	{ \
		assert(p); \
		assert(n); \
		assert((i) < (n)); \
		/* https://stackoverflow.com/a/24633888/2958047 */ \
		if ((i) == 0) \
			*(result) = GetItemAtIndex((p), (n), 1) - GetItemAtIndex((p), (n), 0); \
		else if ((i) + 1 == (n)) \
			*(result) = GetItemAtIndex((p), (n), (i)) - GetItemAtIndex((p), (n), (i)-1); \
		else \
			*(result) = (GetItemAtIndex((p), (n), (i) + 1) - GetItemAtIndex((p), (n), (i)-1)) / 2.0f; \
	} while (0)

static float GradientAtIndex(const float* const p, const size_t n, const size_t i)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	GradientAtIndexUsingAccessor(p, n, i, &result);
#undef GetItemAtIndex
	return result;
}

static float GradientOfGradientAtIndex(const float* const p, const size_t n, const size_t i)
{
	float result;
#define GetItemAtIndex GradientAtIndex
	GradientAtIndexUsingAccessor(p, n, i, &result);
#undef GetItemAtIndex
	return result;
}

SAMinMaxResultF32 SAMinMaxF32(const float* p, size_t n)
{
	assert(n);
	assert(p);
	SAMinMaxResultF32 result = {p[0], p[0]};
	for (size_t i = 1; i < n; ++i)
	{
		const float value = p[i];
		if (result.minValue > value)
			result.minValue = value;
		if (result.maxValue < value)
			result.maxValue = value;
	}
	return result;
}

/**
 * Fast2Sum part of [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm).
 * \param[out] sum updated sum.
 * \param[out] compensation updated compensaton.
 * \param[in] input input value.
 */
#define SAFast2Sum(sum, compensation, input) \
	do \
	{ \
		const float LOCAL(x) = *(sum); \
		const float LOCAL(y) = (input); \
		volatile float LOCAL(t) = LOCAL(x) + LOCAL(y); /* low-order digits of y are lost */ \
		volatile float LOCAL(z) = LOCAL(t) - LOCAL(x); /* cancel the high-order part of y */ \
		*(compensation) = LOCAL(z) - LOCAL(y);         /* recover low-order part of y */ \
		*(sum) = LOCAL(t); \
	} while (0)

#define SAMomentUsingMeanAndAccessor(p, n, reciprocal, moment, mean, result) \
	do \
	{ \
		assert(n); \
		assert(p); \
		assert(moment); \
		float LOCAL(sumPoweredDeviations) = 0; \
		float LOCAL(compensation) = 0; \
		for (size_t LOCAL(i) = 0; LOCAL(i) < (n); ++LOCAL(i)) \
		{ \
			const float LOCAL(deviation) = GetItemAtIndex((p), (n), LOCAL(i)) - (mean); \
			float LOCAL(poweredDeviation) = LOCAL(deviation); \
			for (size_t LOCAL(j) = 1; LOCAL(j) < (moment); ++LOCAL(j)) \
			{ \
				LOCAL(poweredDeviation) *= LOCAL(deviation); \
			} \
			LOCAL(poweredDeviation) -= LOCAL(compensation); \
			SAFast2Sum(&LOCAL(sumPoweredDeviations), &LOCAL(compensation), LOCAL(poweredDeviation)); \
		} \
		*(result) = (float)LOCAL(sumPoweredDeviations) * (reciprocal); \
	} while (0)

#define SAArithmeticMeanUsingAccessor(p, n, reciprocal, result) \
	SAMomentUsingMeanAndAccessor(p, n, reciprocal, 1u, 0.0f, result)

float SAArithmeticMeanF32(const float* const p, const size_t n, const float reciprocal)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	SAArithmeticMeanUsingAccessor(p, n, reciprocal, &result);
#undef GetItemAtIndex
	return result;
}

#define SAVarianceUsingMeanAndAccessor(p, n, reciprocal, mean, result) \
	SAMomentUsingMeanAndAccessor(p, n, reciprocal, 2u, mean, result)

float SAVarianceUsingMeanF32(const float* const p, const size_t n, const float reciprocal, const float mean)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	SAVarianceUsingMeanAndAccessor(p, n, reciprocal, mean, &result);
#undef GetItemAtIndex
	return result;
}

float SAMoment3UsingMeanF32(const float* const p, const size_t n, const float reciprocal, const float mean)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	SAMomentUsingMeanAndAccessor(p, n, reciprocal, 3u, mean, &result);
#undef GetItemAtIndex
	return result;
}

float SASkewnessUsingMeanAndVarianceF32(
	const float* const p,
	const size_t n,
	const float reciprocal,
	const float mean,
	const float variance)
{
	// Avoid division by zero.
	if (variance <= FLT_EPSILON * FLT_EPSILON)
		return 0;

	const float m3 = SAMoment3UsingMeanF32(p, n, reciprocal, mean);
	const float divisor = powf(variance, 1.5f);
	return m3 / divisor;
}

float SAMoment4UsingMeanF32(const float* const p, const size_t n, const float reciprocal, const float mean)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	SAMomentUsingMeanAndAccessor(p, n, reciprocal, 4u, mean, &result);
#undef GetItemAtIndex
	return result;
}

float SAKurtosisUsingMeanAndVarianceF32(
	const float* const p,
	const size_t n,
	const float reciprocal,
	const float mean,
	const float variance)
{
	// Avoid division by zero.
	if (variance <= FLT_EPSILON * FLT_EPSILON)
		return 0;

	const float m4 = SAMoment4UsingMeanF32(p, n, reciprocal, mean);
	const float divisor = variance * variance;
	return m4 / divisor - 3.0f;
}

float SARootMeanSquareF32(const float* const p, const size_t n, const float reciprocal)
{
	float result;
#define GetItemAtIndex(p, n, i) (p)[i]
	SAMomentUsingMeanAndAccessor(p, n, reciprocal, 2u, 0.0f, &result);
#undef GetItemAtIndex
	return sqrtf(result);
}

#define SACountSignChangesUsingAccessor(p, n, result) \
	do \
	{ \
		assert(n); \
		assert(p); \
		size_t LOCAL(numSignChanges) = 0; \
		float LOCAL(lastValue) = GetItemAtIndex((p), (n), 0); \
		for (size_t LOCAL(i) = 1; LOCAL(i) < (n); ++LOCAL(i)) \
		{ \
			const float LOCAL(nextValue) = GetItemAtIndex((p), (n), LOCAL(i)); \
			if (!signbit(LOCAL(lastValue)) != !signbit(LOCAL(nextValue))) \
				++LOCAL(numSignChanges); \
			LOCAL(lastValue) = LOCAL(nextValue); \
		} \
		*(result) = LOCAL(numSignChanges); \
	} while (0)

size_t SACountSignChangesF32(const float* const p, const size_t n)
{
	size_t result;
#define GetItemAtIndex(p, n, i) (p)[i]
	SACountSignChangesUsingAccessor(p, n, &result);
#undef GetItemAtIndex
	return result;
}

float SAPetrosianFractalDimensionF32(const float* const p, const size_t n)
{
	assert(n > 1);
	assert(p);

	size_t numSignChanges;
#define GetItemAtIndex DiffFromNextToCurrent
	SACountSignChangesUsingAccessor(p, n - 1, &numSignChanges);
#undef GetItemAtIndex
	const float size = n;
	const float logSize = logf(size);
	return logSize / (logSize + logf(size / (size + 0.4f * numSignChanges)));
}

#define SAHjorthMobilityUsingVarianceAndAccessor(p, n, reciprocal, variance, result) \
	do \
	{ \
		assert(n > 1); \
		assert(p); \
		float LOCAL(gradMean); \
		SAArithmeticMeanUsingAccessor((p), (n), (reciprocal), &LOCAL(gradMean)); \
		float LOCAL(gradVariance); \
		SAVarianceUsingMeanAndAccessor((p), (n), (reciprocal), LOCAL(gradMean), &LOCAL(gradVariance)); \
		*(result) = sqrtf(LOCAL(gradVariance) / (variance)); \
	} while (0)

float SAHjorthMobilityUsingVarianceF32(
	const float* const p,
	const size_t n,
	const float reciprocal,
	const float variance)
{
	float result;
#define GetItemAtIndex GradientAtIndex
	SAHjorthMobilityUsingVarianceAndAccessor(p, n, reciprocal, variance, &result);
#undef GetItemAtIndex
	return result;
}

float SAHjorthComplexityUsingMobilityF32(
	const float* const p,
	const size_t n,
	const float reciprocal,
	const float mobility)
{
	assert(n > 1);
	assert(p);

#define GetItemAtIndex GradientAtIndex
	float gradMean;
	SAArithmeticMeanUsingAccessor(p, n, reciprocal, &gradMean);
	float gradVariance;
	SAVarianceUsingMeanAndAccessor(p, n, reciprocal, gradMean, &gradVariance);
#undef GetItemAtIndex

#define GetItemAtIndex GradientOfGradientAtIndex
	float gradMobility;
	SAHjorthMobilityUsingVarianceAndAccessor(p, n, reciprocal, gradVariance, &gradMobility);
#undef GetItemAtIndex
	return gradMobility / mobility;
}

#undef LOCAL_CONCAT_
#undef LOCAL_CONCAT
#undef LOCAL
#undef DiffFromNextToCurrentUsingAccessor
#undef GradientAtIndexUsingAccessor
#undef SAFast2Sum
#undef SAMomentUsingMeanAndAccessor
#undef SAArithmeticMeanUsingAccessor
#undef SAVarianceUsingMeanAndAccessor
#undef SACountSignChangesUsingAccessor
#undef SAHjorthMobilityUsingVarianceAndAccessor
