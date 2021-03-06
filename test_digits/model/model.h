#ifdef INCLUDED_BY_NEUTON_C

/* Model info */
#define NEUTON_MODEL_HEADER_VERSION 1
#define NEUTON_MODEL_QLEVEL 8
#define NEUTON_MODEL_FLOAT_SUPPORT 0
#define NEUTON_MODEL_TASK_TYPE 0  // multiclass classification
#define NEUTON_MODEL_NEURONS_COUNT 42
#define NEUTON_MODEL_WEIGHTS_COUNT 249
#define NEUTON_MODEL_INPUTS_COUNT 633
#define NEUTON_MODEL_INPUTS_COUNT_ORIGINAL 3
#define NEUTON_MODEL_INPUT_LIMITS_COUNT 1
#define NEUTON_MODEL_OUTPUTS_COUNT 10
#define NEUTON_MODEL_LOG_SCALE_OUTPUTS 0

/* Preprocessing */
#define NEUTON_PREPROCESSING_ENABLED 1
#define NEUTON_MODEL_WINDOW_SIZE 200
#define NEUTON_BITMASK_ENABLED 0

/* Scaling */
static const float modelInputScaleMin[] = {
	-2896, -5324, -926, -2896, 705, -389.57999, 583.23724, 0, 253159.72, 0.040567122,
	2.0779552, 1.0092942, -1.2507451, -1.82415, -5324, -240, -1107.735, 184.5219,
	0, 20506.645, 0.073239945, 1.2551718, 1.01112, -2.0263498, -1.5321034,
	-926, 4235, 3095.0449, 3267.3611, 0, 23924.35, 0.10083467, 1.2234855, 1.012935,
	-1.5571884, -1.3685987 };
static const float modelInputScaleMax[] = {
	5138, 7758, 9132, 331, 5138, 1821.9351, 2277.1987, 22, 2383364, 0.30892318,
	16.605913, 1.0298517, 1.4209546, 1.6369499, 561, 7758, 1831.265, 2628.1567,
	44, 5035907.5, 0.6038909, 9.933897, 1.0403278, 1.8804244, 6.912375, 3832,
	9132, 4437.8501, 4581.4375, 2, 3138837, 0.64389759, 6.2185793, 1.0393283,
	2.9110332, 16.944168 };

/* Limits */
static const float modelInputMin[] = { 0 };
static const float modelInputMax[] = { 1 };

static const float modelOutputMin[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static const float modelOutputMax[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

/* Types */
typedef uint8_t coeff_t;
typedef int8_t weight_t;
typedef int32_t acc_signed_t;
typedef uint32_t acc_unsigned_t;
typedef uint16_t sources_size_t;
typedef uint8_t weights_size_t;
typedef uint8_t neurons_size_t;

/* Structure */
static const weight_t modelWeights[] = {
	-53, 124, 74, -33, -28, -37, -51, 87, 39, 17, 125, -44, -67, -37, 121,
	52, 84, -38, -23, -128, 99, 74, 78, -22, 112, -78, 88, -72, 125, -54, -41,
	125, 42, 31, 103, -75, -52, -62, 36, 91, -128, 84, 34, 79, -75, -73, -66,
	33, 124, 71, -105, 122, -64, -128, 36, 68, 33, -64, -32, -67, 120, -96,
	115, 111, -128, -118, -124, -12, 113, -43, -34, -115, -18, 99, -49, 21,
	-112, 87, 32, 107, -63, -103, 74, -79, -112, -3, -128, -128, 47, -53, -128,
	62, 14, 70, -119, 61, -25, -128, -128, 27, 77, 38, 53, 126, -121, -83,
	37, 94, -28, -125, 75, 108, -111, -42, 41, 1, -3, -41, -81, 123, 25, -128,
	126, 21, -4, -90, 76, -122, 126, 74, 33, -128, -128, 70, -55, 27, 117,
	121, 126, -67, -68, -71, -111, -34, 123, -97, 101, -18, 110, -40, -87,
	-113, 60, -128, -128, -128, -128, 109, 91, -30, -125, 11, 79, -90, -6,
	-66, 125, -33, 105, 105, 28, -53, 69, 123, 28, -70, -128, -85, -87, 104,
	-111, 127, 118, -63, -79, -84, -128, 48, -23, -89, -86, 121, 18, -83, 107,
	-116, -128, -67, 103, -89, -66, -91, -75, 62, 116, 57, -128, -80, -15,
	125, 127, 79, -10, -44, -70, 99, -117, 65, -128, -128, -128, -65, 107,
	-36, -28, -31, 40, 116, -34, -125, -88, 81, -92, -128, -116, 86, -98, 73,
	-68, -93, 125, 89, -88, -128, -100, 56, -65, -23, 69 };

static const sources_size_t modelLinks[] = {
	26, 166, 368, 547, 607, 633, 11, 50, 204, 532, 616, 633, 208, 215, 388,
	603, 609, 633, 180, 205, 227, 308, 320, 633, 120, 199, 401, 623, 631, 633,
	175, 335, 348, 394, 458, 609, 633, 88, 180, 202, 370, 627, 633, 259, 383,
	444, 600, 610, 627, 633, 7, 194, 317, 388, 493, 600, 633, 0, 188, 248,
	497, 620, 633, 4, 1, 223, 298, 400, 445, 633, 71, 192, 251, 577, 616, 633,
	0, 11, 633, 244, 287, 333, 463, 610, 621, 633, 1, 13, 633, 318, 337, 393,
	439, 617, 628, 633, 51, 71, 85, 122, 147, 612, 633, 43, 250, 273, 374,
	620, 633, 251, 298, 471, 523, 609, 633, 113, 126, 277, 483, 632, 633, 7,
	19, 633, 40, 206, 268, 614, 627, 628, 633, 8, 21, 633, 169, 451, 493, 607,
	616, 629, 633, 132, 294, 299, 320, 458, 633, 68, 87, 160, 185, 631, 633,
	5, 243, 261, 273, 532, 633, 24, 243, 348, 403, 621, 633, 3, 10, 24, 26,
	27, 633, 0, 1, 2, 3, 3, 322, 633, 155, 190, 268, 488, 621, 633, 9, 23,
	30, 633, 7, 8, 6, 50, 155, 229, 633, 2, 15, 32, 633, 197, 214, 276, 413,
	542, 633, 0, 1, 2, 3, 10, 164, 633, 0, 1, 2, 3, 284, 633, 6, 18, 35, 36,
	633, 0, 1, 2, 3, 72, 186, 322, 527, 633, 4, 16, 38, 633, 0, 1, 3, 4, 482,
	633, 5, 17, 25, 29, 34, 40, 633 };

static const weights_size_t modelIntLinksBoundaries[] = {
	0, 6, 12, 18, 24, 30, 37, 43, 50, 58, 64, 70, 78, 79, 88, 89, 96, 103,
	109, 115, 123, 124, 133, 134, 141, 147, 154, 160, 170, 175, 178, 187, 190,
	198, 199, 210, 216, 222, 227, 235, 240, 248 };
static const weights_size_t modelExtLinksBoundaries[] = {
	6, 12, 18, 24, 30, 37, 43, 50, 57, 63, 70, 76, 79, 86, 89, 96, 103, 109,
	115, 121, 124, 131, 134, 141, 147, 153, 159, 165, 171, 178, 184, 188, 195,
	199, 205, 212, 218, 223, 232, 236, 242, 249 };

static const coeff_t modelFuncCoeffs[] = {
	154, 160, 138, 97, 84, 133, 135, 152, 103, 105, 160, 146, 82, 79, 160,
	144, 157, 144, 158, 109, 157, 157, 82, 160, 157, 119, 155, 120, 103, 127,
	157, 140, 139, 160, 144, 155, 154, 160, 37, 160, 144, 160 };

static const neurons_size_t modelOutputNeurons[] = {
	12, 14, 33, 28, 39, 41, 37, 20, 22, 31 };

#endif // INCLUDED_BY_NEUTON_C

