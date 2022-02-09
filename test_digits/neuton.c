#include "neuton.h"
#include "StatFunctions.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>

#ifndef NEUTON_MODEL_FILE
#define NEUTON_MODEL_FILE "model/model.h"
#endif

#define INCLUDED_BY_NEUTON_C
#include NEUTON_MODEL_FILE

#undef NEUTON_MODEL_FILE
#undef INCLUDED_BY_NEUTON_C

#define ARR_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define MAX_INPUT_FLOAT	0.9999999f

/* IO buffers */
static float modelInputs[NEUTON_MODEL_INPUTS_COUNT] = { 0 };
static float modelOutput[NEUTON_MODEL_OUTPUTS_COUNT] = { 0 };

#if (NEUTON_PREPROCESSING_ENABLED == 1)
static uint16_t modelWindowFill = 0;
#endif

static uint8_t modelIsReadyForInference = 0;
static coeff_t modelAccumulators[NEUTON_MODEL_NEURONS_COUNT] = { 0 };


uint8_t neuton_model_quantization_level()
{
	return NEUTON_MODEL_QLEVEL;
}

uint8_t neuton_model_float_calculations()
{
	return NEUTON_MODEL_FLOAT_SUPPORT;
}

TaskType neuton_model_task_type()
{
	return (TaskType) NEUTON_MODEL_TASK_TYPE;
}

uint16_t neuton_model_inputs_count()
{
#if (NEUTON_PREPROCESSING_ENABLED == 0)
	return NEUTON_MODEL_INPUTS_COUNT;
#else
	return NEUTON_MODEL_INPUTS_COUNT_ORIGINAL;
#endif
}

uint16_t neuton_model_outputs_count()
{
	return NEUTON_MODEL_OUTPUTS_COUNT;
}

uint16_t neuton_model_neurons_count()
{
	return NEUTON_MODEL_NEURONS_COUNT;
}

uint32_t neuton_model_weights_count()
{
	return NEUTON_MODEL_WEIGHTS_COUNT;
}

uint16_t neuton_model_inputs_limits_count()
{
	return NEUTON_MODEL_INPUT_LIMITS_COUNT;
}

uint16_t neuton_model_window_size()
{
#if (NEUTON_PREPROCESSING_ENABLED == 1)
	return NEUTON_MODEL_WINDOW_SIZE;
#else
	return 1;
#endif
}

uint32_t neuton_model_ram_usage()
{
	return sizeof(modelInputs) + sizeof(modelOutput)
			+ sizeof(modelAccumulators)
			+ sizeof(modelIsReadyForInference)
#if (NEUTON_PREPROCESSING_ENABLED == 1)
			+ sizeof(modelWindowFill)
#endif
			;
}

uint32_t neuton_model_size()
{
	return sizeof(modelWeights)	+ sizeof(modelLinks) + sizeof(modelFuncCoeffs)
			+ sizeof(modelIntLinksBoundaries) + sizeof(modelExtLinksBoundaries)
			+ sizeof(modelOutputNeurons);
}

uint32_t neuton_model_size_with_meta()
{
	return neuton_model_size()
			+ sizeof(modelInputMin) + sizeof(modelInputMax)
			+ sizeof(modelOutputMin) + sizeof(modelOutputMax)
#if (NEUTON_MODEL_LOG_SCALE_OUTPUTS == 1)
			+ sizeof(modelOutputLogFlag) + sizeof(modelOutputLogScale)
#endif
#if (NEUTON_PREPROCESSING_ENABLED == 1)
			+ sizeof(modelInputScaleMin) + sizeof(modelInputScaleMax)
#endif
			;
}

static void neuton_model_denormalize_outputs()
{
#if (NEUTON_MODEL_TASK_TYPE == 0) || (NEUTON_MODEL_TASK_TYPE == 1)
	float sum = 0;

	for (uint16_t i = 0; i < NEUTON_MODEL_OUTPUTS_COUNT; ++i)
		sum += modelOutput[i];

	for (uint16_t i = 0; i < NEUTON_MODEL_OUTPUTS_COUNT; ++i)
		modelOutput[i] = (sum != 0) ? modelOutput[i] / sum: 0;
#endif

#if (NEUTON_MODEL_TASK_TYPE == 2)
	for (uint16_t i = 0; i < NEUTON_MODEL_OUTPUTS_COUNT; ++i)
	{
		modelOutput[i] = modelOutput[i] * (modelOutputMax[i] - modelOutputMin[i]) + modelOutputMin[i];
#if (NEUTON_MODEL_LOG_SCALE_OUTPUTS == 1)
		if (modelOutputLogFlag[i])
			modelOutput[i] = exp(modelOutput[i]) - modelOutputLogScale[i];
#endif
	}
#endif
}

#if (NEUTON_PREPROCESSING_ENABLED == 0)
static float neuton_model_normalize_value(uint16_t index, float value)
{
#if (NEUTON_MODEL_INPUT_LIMITS_COUNT == 1)
	float result = value - modelInputMin[0];
	float diff = modelInputMax[0] - modelInputMin[0];
#else
	float result = value - modelInputMin[index];
	float diff = modelInputMax[index] - modelInputMin[index];
#endif

	if (diff)
		result /= diff;

	if (result > 1.0)
		result = 1.0;
	if (result < 0.0)
		result = 0.0;

	return result;
}
#else // (NEUTON_PREPROCESSING_ENABLED == 0)
static float neuton_model_scale_input(uint16_t index, float value)
{
	float result = (value - modelInputScaleMin[index]);
	float diff = (modelInputScaleMax[index] - modelInputScaleMin[index]);
	if (diff)
		result /= diff;

	if (result > 1.0)
		result = 1.0;
	if (result < 0.0)
		result = 0.0;

	return result;
}
#endif // (NEUTON_PREPROCESSING_ENABLED == 0)

int8_t neuton_model_set_inputs(float *inputs)
{
	if (!inputs)
		return -1;

#if (NEUTON_PREPROCESSING_ENABLED == 0)
	for (uint16_t modelInputIndex = 0; modelInputIndex < NEUTON_MODEL_INPUTS_COUNT; ++modelInputIndex)
		modelInputs[modelInputIndex] = neuton_model_normalize_value(modelInputIndex, inputs[modelInputIndex]);

	modelIsReadyForInference = 1;

	return 0;
#else
	for (uint16_t originalIdx = 0; originalIdx < NEUTON_MODEL_INPUTS_COUNT_ORIGINAL; ++originalIdx)
		modelInputs[originalIdx * NEUTON_MODEL_WINDOW_SIZE + modelWindowFill] = inputs[originalIdx];

	if (++modelWindowFill >= NEUTON_MODEL_WINDOW_SIZE)
	{
		modelWindowFill = 0;
		modelIsReadyForInference = 1;

		return 0;
	}

	return 1;
#endif
}

#if (NEUTON_MODEL_QLEVEL == 8) || (NEUTON_MODEL_QLEVEL == 16)
#if (NEUTON_MODEL_FLOAT_SUPPORT == 0)
static coeff_t accurate_fast_sigmoid(acc_signed_t arg)
{
	coeff_t qResult = 0;
	coeff_t secondPointY = 0;
	coeff_t firstPointY = 0;

	const uint8_t QLVL = NEUTON_MODEL_QLEVEL;
	const acc_signed_t CT_MAX_VALUE = 1u << QLVL;
	const acc_signed_t intPart = abs(arg) / (2u << (QLVL - 1));
	const acc_signed_t realPart = abs(arg) - (intPart << QLVL);

	if (intPart == 0 && realPart == 0)
	{
		return ldexp(0.5, QLVL);
	}

	int s = arg > 0 ? 0 : 1;
	if (realPart == 0)
	{
		for (int i = 0; i < QLVL; i++)
		{
			const uint8_t bit = ((i / intPart + s) % 2);
			qResult = qResult | (bit << (QLVL - i - 1));
		}
		return qResult;
	}

	const acc_signed_t secondPointX = intPart + 1;
	if (intPart == 0)
	{
		firstPointY = ldexp(0.5, QLVL);
		for (int i = 0; i < QLVL; i++)
		{
			const uint8_t bit = ((i / secondPointX) % 2);
			secondPointY = secondPointY | (bit << (QLVL - i - 1));
		}
	}
	else
	{
		if (secondPointX == 0)
		{
			for (int i = 0; i < QLVL; i++)
			{
				const uint8_t bit = ((i / intPart) % 2);
				firstPointY = firstPointY | (bit << (QLVL - i - 1));
			}
			secondPointY = ldexp(0.5, QLVL);
		}
		else
		{
			for (int i = 0; i < QLVL; i++)
			{
				uint8_t bit = ((i / intPart) % 2);
				firstPointY = firstPointY | (bit << (QLVL - i - 1));
				bit = ((i / secondPointX) % 2);
				secondPointY = secondPointY | (bit << (QLVL - i - 1));
			}
		}
	}

	const acc_signed_t res = (CT_MAX_VALUE - realPart) * firstPointY + realPart * secondPointY;
	if (arg > 0)
	{
		return res >> QLVL;
	}
	else
	{
		qResult = res >> QLVL;
		return qResult == 0 ? CT_MAX_VALUE - 1 : CT_MAX_VALUE - qResult;
	}
}
#endif // (NEUTON_MODEL_FLOAT_SUPPORT == 0)

static float neuton_deqantize_value(coeff_t value)
{
	return (float) value / (float) (2u << (NEUTON_MODEL_QLEVEL - 1));
}

#if (NEUTON_MODEL_QLEVEL == 8)
#define KSHIFT 2
#endif

#if (NEUTON_MODEL_QLEVEL == 16)
#define KSHIFT 10
#endif

static coeff_t neuton_activation_fn(neurons_size_t neuronIndex, acc_signed_t summ)
{
#if (NEUTON_MODEL_FLOAT_SUPPORT == 1)

	const float qs = (float) (((acc_signed_t) modelFuncCoeffs[neuronIndex] * summ)
			>> (NEUTON_MODEL_QLEVEL + KSHIFT - 1)) / (float) (2u << (NEUTON_MODEL_QLEVEL - 1));
	const float tmpValue = 1.0f / (1.0f + expf(-qs));

	return ldexp(tmpValue > MAX_INPUT_FLOAT ? MAX_INPUT_FLOAT : tmpValue, NEUTON_MODEL_QLEVEL);

#else // (NEUTON_MODEL_FLOAT_SUPPORT == 1)

	return accurate_fast_sigmoid(
		-(((acc_signed_t) modelFuncCoeffs[neuronIndex] * summ) >> (NEUTON_MODEL_QLEVEL + KSHIFT - 1))
	);

#endif // (NEUTON_MODEL_FLOAT_SUPPORT == 1)
}
#endif // (NEUTON_MODEL_QLEVEL == 8) || (NEUTON_MODEL_QLEVEL == 16)

#if (NEUTON_MODEL_QLEVEL == 32)
static coeff_t neuton_activation_fn(neurons_size_t neuronIndex, acc_signed_t summ)
{
	return 1.0f / (1.0f + exp((acc_signed_t) ((acc_signed_t) -modelFuncCoeffs[neuronIndex]) * summ));
}
#endif

#if (NEUTON_PREPROCESSING_ENABLED == 1)
static void neuton_model_extract_features();
static void neuton_model_scale_features();
#endif

int8_t neuton_model_run_inference(uint16_t *index, float **outputs)
{
	if (!modelIsReadyForInference)
		return -1;

#if (NEUTON_PREPROCESSING_ENABLED == 1)
	neuton_model_extract_features();
	neuton_model_scale_features();
#endif

	for (neurons_size_t i = 0; i < ARR_SIZE(modelAccumulators); ++i)
		modelAccumulators[i] = 0;

	weights_size_t weightIndex = 0;
	for (neurons_size_t neuronIndex = 0; neuronIndex < NEUTON_MODEL_NEURONS_COUNT; ++neuronIndex)
	{
		acc_signed_t summ = 0;

		for (; weightIndex < modelIntLinksBoundaries[neuronIndex]; ++weightIndex)
		{
			const acc_signed_t firstValue  = (acc_signed_t) modelWeights[weightIndex];
			const acc_signed_t secondValue = (acc_signed_t) modelAccumulators[modelLinks[weightIndex]];
			summ += firstValue * secondValue;
		}

		for (; weightIndex < modelExtLinksBoundaries[neuronIndex]; ++weightIndex)
		{
			float input = (modelLinks[weightIndex] >= NEUTON_MODEL_INPUTS_COUNT) ? 1.0f : modelInputs[modelLinks[weightIndex]];

			const acc_signed_t firstValue  = (acc_signed_t) modelWeights[weightIndex];
#if (NEUTON_MODEL_QLEVEL == 32)
			const acc_signed_t secondValue = (acc_signed_t) input;
#else
			const acc_signed_t secondValue = (acc_signed_t) ldexp(input > MAX_INPUT_FLOAT ? MAX_INPUT_FLOAT : input, NEUTON_MODEL_QLEVEL);
#endif
			summ += firstValue * secondValue;
		}

		modelAccumulators[neuronIndex] = neuton_activation_fn(neuronIndex, summ);
	}

#if (NEUTON_MODEL_QLEVEL == 32)
	for (neurons_size_t i = 0; i < ARR_SIZE(modelOutputNeurons); ++i)
		modelOutput[i] = modelAccumulators[modelOutputNeurons[i]];
#else
	for (neurons_size_t i = 0; i < ARR_SIZE(modelOutputNeurons); ++i)
		modelOutput[i] = neuton_deqantize_value(modelAccumulators[modelOutputNeurons[i]]);
#endif

	neuton_model_denormalize_outputs();

#if (NEUTON_MODEL_OUTPUTS_COUNT == 1)
	if (index)
		*index = 0;
#else
	if (index)
	{
		uint16_t target = 0;
		float max = 0.0;

		for (uint16_t i = 0; i < ARR_SIZE(modelOutput); ++i)
			if (max < modelOutput[i])
			{
				max = modelOutput[i];
				target = i;
			}

		*index = target;
	}
#endif

	if (outputs)
		*outputs = modelOutput;

	return 0;
}

#if (NEUTON_PREPROCESSING_ENABLED == 1)

enum ExtractedFeatures
{
	EF_MIN = 0,
	EF_MAX,
	EF_MEAN,
	EF_RMS,
	EF_SIGN_CHANGES,
	EF_VARIANCE,
	EF_H_MOBILITY,
	EF_H_COMPLEXITY,
	EF_PFD,
	EF_SKEWNESS,
	EF_KURTOSIS,

	EF_COUNT
};

static void neuton_model_extract_features()
{
	static float reciprocal = SAReciprocalF32(NEUTON_MODEL_WINDOW_SIZE);
	SAMinMaxResultF32 r;

	for (uint16_t featureIdx = 0; featureIdx < NEUTON_MODEL_INPUTS_COUNT_ORIGINAL; ++featureIdx)
	{
		float* window = &modelInputs[featureIdx * NEUTON_MODEL_WINDOW_SIZE];
		float* ef = &modelInputs[featureIdx * EF_COUNT + NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE];

		r = SAMinMaxF32(window, NEUTON_MODEL_WINDOW_SIZE);
		ef[EF_MIN] = r.minValue;
		ef[EF_MAX] = r.maxValue;
		ef[EF_MEAN] = SAArithmeticMeanF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal);
		ef[EF_RMS] = SARootMeanSquareF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal);
		ef[EF_SIGN_CHANGES] = SACountSignChangesF32(window, NEUTON_MODEL_WINDOW_SIZE);
		ef[EF_VARIANCE] = SAVarianceUsingMeanF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal, ef[EF_MEAN]);
		ef[EF_H_MOBILITY] = SAHjorthMobilityUsingVarianceF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal, ef[EF_VARIANCE]);
		ef[EF_H_COMPLEXITY] = SAHjorthComplexityUsingMobilityF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal, ef[EF_H_MOBILITY]);
		ef[EF_PFD] = SAPetrosianFractalDimensionF32(window, NEUTON_MODEL_WINDOW_SIZE);
		ef[EF_SKEWNESS] = SASkewnessUsingMeanAndVarianceF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal, ef[EF_MEAN], ef[EF_VARIANCE]);
		ef[EF_KURTOSIS] = SAKurtosisUsingMeanAndVarianceF32(window, NEUTON_MODEL_WINDOW_SIZE, reciprocal, ef[EF_MEAN], ef[EF_VARIANCE]);
	}
}

static void neuton_model_scale_features()
{
	for (uint16_t originalFeatureIdx = 0; originalFeatureIdx < NEUTON_MODEL_INPUTS_COUNT_ORIGINAL; ++originalFeatureIdx)
	{
		float* window = &modelInputs[originalFeatureIdx * NEUTON_MODEL_WINDOW_SIZE];
		for (uint16_t i = 0; i < NEUTON_MODEL_WINDOW_SIZE; ++i)
			window[i] = neuton_model_scale_input(originalFeatureIdx, window[i]);

		float* extractedFeatures = &modelInputs[NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE + EF_COUNT * originalFeatureIdx];
		for (uint8_t i = 0; i < EF_COUNT; ++i)
			extractedFeatures[i] = neuton_model_scale_input(NEUTON_MODEL_INPUTS_COUNT_ORIGINAL + EF_COUNT * originalFeatureIdx + i, extractedFeatures[i]);
	}
}
#endif // (NEUTON_PREPROCESSING_ENABLED == 1)
