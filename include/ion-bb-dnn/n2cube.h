// Refer dpu-pynq 1.2.0 / dnndk/n2cube.h

/*
 * Copyright 2020 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _ION_BB_DNN_N2CUBE_H_
#define _ION_BB_DNN_N2CUBE_H_

#include <stdexcept>

#include "rt_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define VITIS_AI_VERSION "1.2"
#define N2Cube_VERSION "4.2"

/* DPU Task runtime mode definitions */

/* Task in normal mode (defaul mode) */
#define T_MODE_NORMAL (0)

/* Task in profiling mode in order to collect performance stastics for each DPU Node */
#define T_MODE_PROFILE (1 << 0)

/* Task in debug mode in order to dump each Node's Code/Bias/Weights/Input/Output raw data for debugging */
#define T_MODE_DEBUG (1 << 1)

/* Exported data structures of DPU Kernel/Task/Tensor */
struct dpu_kernel;
struct dpu_task;
struct task_tensor;

typedef struct dpu_kernel DPUKernel;
typedef struct dpu_task DPUTask;
typedef struct task_tensor DPUTensor;

/* The exception handling mode */
#define N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT 0
#define N2CUBE_EXCEPTION_MODE_RET_ERR_CODE 1

using dpuOpen_t = int (*)();
using dpuClose_t = int (*)();
using dpuLoadKernel_t = DPUKernel *(*)(const char *netName);
using dpuDestroyKernel_t = int (*)(DPUKernel *kernel);
using dpuCreateTask_t = DPUTask *(*)(DPUKernel *kernel, int mode);
using dpuRunTask_t = int (*)(DPUTask *task);
using dpuDestroyTask_t = int (*)(DPUTask *task);
using dpuGetInputTensor_t = DPUTensor *(*)(DPUTask *task, const char *nodeName, int idx);
using dpuGetOutputTensor_t = DPUTensor *(*)(DPUTask *task, const char *nodeName, int idx);
using dpuGetTensorSize_t = int (*)(DPUTensor *tensor);
using dpuGetTensorAddress_t = int8_t *(*)(DPUTensor *tensor);
using dpuGetTensorScale_t = float (*)(DPUTensor *tensor);
using dpuGetTensorHeight_t = int (*)(DPUTensor *tensor);
using dpuGetTensorWidth_t = int (*)(DPUTensor *tensor);
using dpuGetTensorChannel_t = int (*)(DPUTensor *tensor);
using dpuRunSoftmax_t = int (*)(int8_t *input, float *output, int numClasses, int batchSize, float scale);

dpuOpen_t dpuOpen;
dpuClose_t dpuClose;
dpuLoadKernel_t dpuLoadKernel;
dpuDestroyKernel_t dpuDestroyKernel;
dpuCreateTask_t dpuCreateTask;
dpuRunTask_t dpuRunTask;
dpuDestroyTask_t dpuDestroyTask;
dpuGetInputTensor_t dpuGetInputTensor;
dpuGetOutputTensor_t dpuGetOutputTensor;
dpuGetTensorSize_t dpuGetTensorSize;
dpuGetTensorAddress_t dpuGetTensorAddress;
dpuGetTensorScale_t dpuGetTensorScale;
dpuGetTensorHeight_t dpuGetTensorHeight;
dpuGetTensorWidth_t dpuGetTensorWidth;
dpuGetTensorChannel_t dpuGetTensorChannel;
dpuRunSoftmax_t dpuRunSoftmax;

bool dnndk_init() {
    static ion::bb::dnn::DynamicModule dm("n2cube");
    if (!dm.is_available()) {
        std::cerr << "Can't load n2cube" << std::endl;
        return false;
    }

#define RESOLVE_SYMBOL(SYM_NAME, MANGLED_NAME)               \
    SYM_NAME = dm.get_symbol<SYM_NAME##_t>(#MANGLED_NAME);   \
    if (SYM_NAME == nullptr) {                               \
        throw std::runtime_error(                            \
            #SYM_NAME " is unavailable on your n2cube DSO"); \
    }

    RESOLVE_SYMBOL(dpuOpen, dpuOpen);
    RESOLVE_SYMBOL(dpuClose, dpuClose);
    RESOLVE_SYMBOL(dpuLoadKernel, _Z13dpuLoadKernelPKc);
    RESOLVE_SYMBOL(dpuDestroyKernel, _Z16dpuDestroyKernelP10dpu_kernel);
    RESOLVE_SYMBOL(dpuCreateTask, _Z13dpuCreateTaskP10dpu_kerneli);
    RESOLVE_SYMBOL(dpuDestroyTask, _Z14dpuDestroyTaskP8dpu_task);
    RESOLVE_SYMBOL(dpuRunTask, _Z10dpuRunTaskP8dpu_task);
    RESOLVE_SYMBOL(dpuGetInputTensor, _Z17dpuGetInputTensorP8dpu_taskPKci);
    RESOLVE_SYMBOL(dpuGetOutputTensor, _Z18dpuGetOutputTensorP8dpu_taskPKci);
    RESOLVE_SYMBOL(dpuGetTensorSize, _Z16dpuGetTensorSizeP11task_tensor);
    RESOLVE_SYMBOL(dpuGetTensorAddress, _Z19dpuGetTensorAddressP11task_tensor);
    RESOLVE_SYMBOL(dpuGetTensorScale, _Z17dpuGetTensorScaleP11task_tensor);
    RESOLVE_SYMBOL(dpuGetTensorHeight, _Z18dpuGetTensorHeightP11task_tensor);
    RESOLVE_SYMBOL(dpuGetTensorWidth, _Z17dpuGetTensorWidthP11task_tensor);
    RESOLVE_SYMBOL(dpuGetTensorChannel, _Z19dpuGetTensorChannelP11task_tensor);
    RESOLVE_SYMBOL(dpuRunSoftmax, _Z13dpuRunSoftmaxPaPfiif);

#undef RESOLVE_SYMBOL

    return true;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif
