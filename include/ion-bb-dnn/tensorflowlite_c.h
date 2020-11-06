#ifndef ION_BB_DNN_TENSORFLOWLITE_C_H
#define ION_BB_DNN_TENSORFLOWLITE_C_H

#include <stdint.h>

#include "tensorflowlite_types.h"
#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

// Refer tensorflow 2.3.1 / tensorflow/lite/c/tensorflow_c.h
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;

using TfLiteModelCreate_t                        = TfLiteModel* (*)(const void* model_data, size_t model_size);
using TfLiteModelCreateFromFile_t                = TfLiteModel* (*)(const char* model_path);
using TfLiteModelDelete_t                        = void (*)(TfLiteModel* model);
using TfLiteInterpreterOptionsCreate_t           = TfLiteInterpreterOptions* (*)();
using TfLiteInterpreterOptionsDelete_t           = void (*)( TfLiteInterpreterOptions* options);
using TfLiteInterpreterOptionsSetNumThreads_t    = void (*)(TfLiteInterpreterOptions* options, int32_t num_threads);
using TfLiteInterpreterOptionsAddDelegate_t      = void (*)(TfLiteInterpreterOptions* options, TfLiteDelegate* delegate);
using TfLiteInterpreterOptionsSetErrorReporter_t = void (*)( TfLiteInterpreterOptions* options, void (*reporter)(void* user_data, const char* format, va_list args), void* user_data);
using TfLiteInterpreterCreate_t                  = TfLiteInterpreter* (*)(const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options);
using TfLiteInterpreterDelete_t                  = void (*)(TfLiteInterpreter* interpreter);
using TfLiteInterpreterGetInputTensorCount_t     = int32_t (*)(const TfLiteInterpreter* interpreter);
using TfLiteInterpreterGetInputTensor_t          = TfLiteTensor* (*)(const TfLiteInterpreter* interpreter, int32_t input_index);
using TfLiteInterpreterResizeInputTensor_t       = TfLiteStatus (*)(TfLiteInterpreter* interpreter, int32_t input_index, const int* input_dims, int32_t input_dims_size);
using TfLiteInterpreterAllocateTensors_t         = TfLiteStatus (*)( TfLiteInterpreter* interpreter);
using TfLiteInterpreterInvoke_t                  = TfLiteStatus (*)( TfLiteInterpreter* interpreter);
using TfLiteInterpreterGetOutputTensorCount_t    = int32_t (*)( const TfLiteInterpreter* interpreter);
using TfLiteInterpreterGetOutputTensor_t         = const TfLiteTensor* (*)( const TfLiteInterpreter* interpreter, int32_t output_index);
using TfLiteTensorType_t                         = TfLiteType (*)(const TfLiteTensor* tensor);
using TfLiteTensorNumDims_t                      = int32_t (*)(const TfLiteTensor* tensor);
using TfLiteTensorDim_t                          = int32_t (*)(const TfLiteTensor* tensor, int32_t dim_index);
using TfLiteTensorByteSize_t                     = size_t (*)(const TfLiteTensor* tensor);
using TfLiteTensorData_t                         = void* (*)(const TfLiteTensor* tensor);
using TfLiteTensorName_t                         = const char* (*)(const TfLiteTensor* tensor);
using TfLiteTensorQuantizationParams_t           = TfLiteQuantizationParams (*)(const TfLiteTensor* tensor);
using TfLiteTensorCopyFromBuffer_t               = TfLiteStatus (*)(TfLiteTensor* tensor, const void* input_data, size_t input_data_size);
using TfLiteTensorCopyToBuffer_t                 = TfLiteStatus (*)(const TfLiteTensor* output_tensor, void* output_data, size_t output_data_size);

TfLiteModelCreate_t                        TfLiteModelCreate;
TfLiteModelCreateFromFile_t                TfLiteModelCreateFromFile;
TfLiteModelDelete_t                        TfLiteModelDelete;
TfLiteInterpreterOptionsCreate_t           TfLiteInterpreterOptionsCreate;
TfLiteInterpreterOptionsDelete_t           TfLiteInterpreterOptionsDelete;
TfLiteInterpreterOptionsSetNumThreads_t    TfLiteInterpreterOptionsSetNumThreads;
TfLiteInterpreterOptionsAddDelegate_t      TfLiteInterpreterOptionsAddDelegate;
TfLiteInterpreterOptionsSetErrorReporter_t TfLiteInterpreterOptionsSetErrorReporter;
TfLiteInterpreterCreate_t                  TfLiteInterpreterCreate;
TfLiteInterpreterDelete_t                  TfLiteInterpreterDelete;
TfLiteInterpreterGetInputTensorCount_t     TfLiteInterpreterGetInputTensorCount;
TfLiteInterpreterGetInputTensor_t          TfLiteInterpreterGetInputTensor;
TfLiteInterpreterResizeInputTensor_t       TfLiteInterpreterResizeInputTensor;
TfLiteInterpreterAllocateTensors_t         TfLiteInterpreterAllocateTensors;
TfLiteInterpreterInvoke_t                  TfLiteInterpreterInvoke;
TfLiteInterpreterGetOutputTensorCount_t    TfLiteInterpreterGetOutputTensorCount;
TfLiteInterpreterGetOutputTensor_t         TfLiteInterpreterGetOutputTensor;
TfLiteTensorType_t                         TfLiteTensorType;
TfLiteTensorNumDims_t                      TfLiteTensorNumDims;
TfLiteTensorDim_t                          TfLiteTensorDim;
TfLiteTensorByteSize_t                     TfLiteTensorByteSize;
TfLiteTensorData_t                         TfLiteTensorData;
TfLiteTensorName_t                         TfLiteTensorName;
TfLiteTensorQuantizationParams_t           TfLiteTensorQuantizationParams;
TfLiteTensorCopyFromBuffer_t               TfLiteTensorCopyFromBuffer;
TfLiteTensorCopyToBuffer_t                 TfLiteTensorCopyToBuffer;

bool tensorflowlite_init() {
    static ion::bb::dnn::DynamicModule dm("tensorflowlite_c");
    if (!dm.is_available()) {
        return false;
    }

#define RESOLVE_SYMBOL(SYM_NAME)                              \
    SYM_NAME = dm.get_symbol<SYM_NAME ## _t>(#SYM_NAME);      \
    if (SYM_NAME == nullptr) {                                \
        throw std::runtime_error(                             \
            #SYM_NAME " is unavailable on your edgetpu DSO"); \
    }

    RESOLVE_SYMBOL(TfLiteModelCreate);
    RESOLVE_SYMBOL(TfLiteModelCreateFromFile);
    RESOLVE_SYMBOL(TfLiteModelDelete);
    RESOLVE_SYMBOL(TfLiteInterpreterOptionsCreate);
    RESOLVE_SYMBOL(TfLiteInterpreterOptionsDelete);
    RESOLVE_SYMBOL(TfLiteInterpreterOptionsSetNumThreads);
    RESOLVE_SYMBOL(TfLiteInterpreterOptionsAddDelegate);
    RESOLVE_SYMBOL(TfLiteInterpreterOptionsSetErrorReporter);
    RESOLVE_SYMBOL(TfLiteInterpreterCreate);
    RESOLVE_SYMBOL(TfLiteInterpreterDelete);
    RESOLVE_SYMBOL(TfLiteInterpreterGetInputTensorCount);
    RESOLVE_SYMBOL(TfLiteInterpreterGetInputTensor);
    RESOLVE_SYMBOL(TfLiteInterpreterResizeInputTensor);
    RESOLVE_SYMBOL(TfLiteInterpreterAllocateTensors);
    RESOLVE_SYMBOL(TfLiteInterpreterInvoke);
    RESOLVE_SYMBOL(TfLiteInterpreterGetOutputTensorCount);
    RESOLVE_SYMBOL(TfLiteInterpreterGetOutputTensor);
    RESOLVE_SYMBOL(TfLiteTensorType);
    RESOLVE_SYMBOL(TfLiteTensorNumDims);
    RESOLVE_SYMBOL(TfLiteTensorDim);
    RESOLVE_SYMBOL(TfLiteTensorByteSize);
    RESOLVE_SYMBOL(TfLiteTensorData);
    RESOLVE_SYMBOL(TfLiteTensorName);
    RESOLVE_SYMBOL(TfLiteTensorQuantizationParams);
    RESOLVE_SYMBOL(TfLiteTensorCopyFromBuffer);
    RESOLVE_SYMBOL(TfLiteTensorCopyToBuffer);

#undef RESOLVE_SYMBOL

    return true;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_H_
