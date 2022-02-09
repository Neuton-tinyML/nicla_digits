#ifndef NEUTON_H
#define NEUTON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
	TASK_MULTICLASS_CLASSIFICATION = 0,
	TASK_BINARY_CLASSIFICATION     = 1,
	TASK_REGRESSION                = 2
} TaskType;

/* Model info */
uint8_t neuton_model_quantization_level();
uint8_t neuton_model_float_calculations();

TaskType neuton_model_task_type();
uint16_t neuton_model_inputs_count();
uint16_t neuton_model_outputs_count();

uint16_t neuton_model_neurons_count();
uint32_t neuton_model_weights_count();
uint16_t neuton_model_inputs_limits_count();
uint16_t neuton_model_window_size();

uint32_t neuton_model_ram_usage();
uint32_t neuton_model_size();
uint32_t neuton_model_size_with_meta();

/* Inference */
int8_t neuton_model_set_inputs(float* inputs);
int8_t neuton_model_run_inference(uint16_t* index, float** outputs);

#ifdef __cplusplus
}
#endif

#endif // NEUTON_H
