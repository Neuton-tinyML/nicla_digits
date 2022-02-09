#ifndef COMPONENTS_STATALGS_TINY_STATFUNCTIONS_H_
#define COMPONENTS_STATALGS_TINY_STATFUNCTIONS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SAMinMaxResultF32
{
	float minValue;
	float maxValue;
} SAMinMaxResultF32;

/// Compute the minimum and the maximum values simultaneously.
SAMinMaxResultF32 SAMinMaxF32(const float* p, size_t n);

/// Compute the reciprocal of the specified number.  This operation is expected to occur as rarely as possible.
#define SAReciprocalF32(n) (1.0f / (float)(n))

/// Compute the arithmetic sum.
#define SASumF32(p, n) (SAArithmeticMeanF32((p), (n), 1.0f))

/// Compute the arithmetic mean.
float SAArithmeticMeanF32(const float* p, size_t n, float reciprocal);

/// Compute the variance using the precomputed mean.
float SAVarianceUsingMeanF32(const float* p, size_t n, float reciprocal, float mean);

/// Compute the variance.
float SAVarianceF32(const float* p, size_t n, float reciprocal);

/// Compute the 3rd central moment using the precomputed mean (for testing purposes only).
float SAMoment3UsingMeanF32(const float* p, size_t n, float reciprocal, float mean);

/// Compute the [skewness](https://en.wikipedia.org/wiki/Skewness) using the precomputed mean and variance.
float SASkewnessUsingMeanAndVarianceF32(const float* p, size_t n, float reciprocal, float mean, float variance);

/// Compute the 4th central moment using the precomputed mean (for testing purposes only).
float SAMoment4UsingMeanF32(const float* p, size_t n, float reciprocal, float mean);

/// Compute the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis) using the precomputed mean and variance.
float SAKurtosisUsingMeanAndVarianceF32(const float* p, size_t n, float reciprocal, float mean, float variance);

/// Compute the root mean square, [RMS](https://en.wikipedia.org/wiki/Root_mean_square).
float SARootMeanSquareF32(const float* p, size_t n, float reciprocal);

/// Count the number of sign changes.
size_t SACountSignChangesF32(const float* p, size_t n);

/// Compute the Petrosian fractal dimension, PFD.
float SAPetrosianFractalDimensionF32(const float* p, size_t n);

/// Compute the [Hjorth Mobility](https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Mobility)
/// using the precomputed variance.
float SAHjorthMobilityUsingVarianceF32(const float* p, size_t n, float reciprocal, float variance);

/// Compute the [Hjorth Complexity](https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Complexity)
/// using the precomputed Hjorth Mobility.
float SAHjorthComplexityUsingMobilityF32(const float* p, size_t n, float reciprocal, float mobility);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COMPONENTS_STATALGS_TINY_STATFUNCTIONS_H_
