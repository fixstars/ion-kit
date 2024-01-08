
#
# Examples
#

if(${ION_BB_BUILD_base})
    ion_compile(producerx3_compile SRCS producerx3_compile.cc PIPELINE_NAME producerx3)
    ion_run(producerx3 producerx3_compile SRCS producerx3_run.cc TARGET_STRING "host-profile")
    if(${CUDA_FOUND})
        ion_run(producerx3_gpu producerx3_compile SRCS producerx3_run.cc TARGET_STRING "host-cuda-cuda_capability_50-profile")
    endif()
endif()

if(${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_sgm})
    ion_compile(sgm_compile SRCS sgm_compile.cc PIPELINE_NAME sgm)
    ion_run(sgm sgm_compile SRCS sgm_run.cc TARGET_STRING "host-profile")
    if(${CUDA_FOUND})
        ion_run(sgm_gpu sgm_compile SRCS sgm_run.cc TARGET_STRING "host-cuda-cuda_capability_50-profile")
    endif()
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_dnn})
    ion_compile(dnn_compile SRCS dnn_compile.cc PIPELINE_NAME dnn)
    ion_run(dnn dnn_compile SRCS dnn_run.cc TARGET_STRING "host-profile")
    if(${CUDA_FOUND})
        ion_run(dnn_gpu dnn_compile SRCS dnn_run.cc TARGET_STRING "host-cuda-cuda_capability_50-profile")
    endif()

    ion_jit(dnn_jit SRCS dnn.cc)
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io})
    ion_compile(demo_compile SRCS demo_compile.cc PIPELINE_NAME demo)
    ion_run(demo demo_compile SRCS demo_run.cc TARGET_STRING "host-profile")
    if(${CUDA_FOUND})
        ion_run(demo_gpu demo_compile SRCS demo_run.cc TARGET_STRING "host-cuda-cuda_capability_50-profile")
    endif()

    ion_jit(demo_jit SRCS demo.cc)
endif()

if(${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io})
    ion_jit(isp_jit SRCS isp.cc)
    ion_register_test(isp isp_jit TARGET_STRING "host-profile")
    if(${CUDA_FOUND}})
        ion_register_test(isp_gpu isp_jit TARGET_STRING "host-cuda-cuda_capability_50-profile")
    endif()
endif()

if(${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_sgm})
    ion_jit(isp_and_sgm_jit SRCS isp_and_sgm.cc)
    ion_register_test(isp_and_sgm isp_and_sgm_jit TARGET_STRING "host-profile")
    if(${CUDA_FOUND}})
        # Temporaly disabled
        # ion_register_test(isp_and_sgm_gpu isp_and_sgm_jit RUNTIME_ARGS ${CMAKE_SOURCE_DIR}/data/image/OV5647x2-5184x1944-GB10.raw 5184 1944 10 6 TARGET_STRING "host-cuda-cuda_capability_50-profile")
    endif()
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_image-processing})
    ion_jit(imx219_isp_display_jit SRCS imx219_isp_display.cc)
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_dnn})
    ion_jit(gender_count SRCS gender_count.cc)
endif()

if(${ION_BB_BUILD_image-io})
    ion_jit(io SRCS io.cc)
endif()
# ion_jit(u3v_jit SRCS u3v.cc)

if(${ION_BB_BUILD_image-io} AND UNIX AND NOT APPLE)
    ion_jit(v4l2_jit SRCS v4l2.cc)
    ion_register_test(v4l2 v4l2_jit TARGET_STRING "host-profile")

    ion_jit(realsense_jit SRCS realsense.cc)
    ion_register_test(realsense realsense_jit TARGET_STRING "host-profile")
endif()

if(${ION_BB_BUILD_fpga})
    ion_compile(fpga_isp_compile SRCS fpga_isp_compile.cc PIPELINE_NAME fpga_isp)
    ion_run(fpga_isp_cpu fpga_isp_compile SRCS fpga_isp_run.cc TARGET_STRING "host-profile")

    ion_compile(dnn_fpga_compile SRCS dnn_fpga_compile.cc PIPELINE_NAME dnn_fpga)
    ion_run(dnn_fpga dnn_fpga_compile SRCS dnn_fpga_run.cc TARGET_STRING "arm-64-linux-vivado_hls-dpu")
endif()

if(MSVC)
    add_definitions(/bigobj)
    message(STATUS "Allow big object for example/")
endif()
