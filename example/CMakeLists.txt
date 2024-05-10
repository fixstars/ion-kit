
#
# Examples
#

find_package(OpenCV 4 QUIET)

if(${ION_BB_BUILD_base})
    if(${CUDA_FOUND})
        ion_aot_executable(producerx3 SRCS_COMPILE producerx3_compile.cc SRCS_RUN producerx3_run.cc TARGET_STRING "host-cuda-cuda_capability_50-profile")
    elseif()
        ion_aot_executable( SRCS_COMPILE producerx3_compile.cc SRCS_RUN producerx3_run.cc TARGET_STRING "host-profile")
    endif()
endif()

if(${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_sgm} AND OpenCV_FOUND)
    if(${CUDA_FOUND})
        ion_aot_executable(sgm SRCS_COMPILE sgm_compile.cc SRCS_RUN sgm_run.cc INCS ${OPenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES} TARGET_STRING "host-cuda-cuda_capability_50-profile")
    elseif()
        ion_aot_executable(sgm SRCS_COMPILE sgm_compile.cc SRCS_RUN sgm_run.cc INCS ${OPenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES} TARGET_STRING "host-profile")
    endif()
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_dnn} AND OpenCV_FOUND)
    if(${CUDA_FOUND})
        ion_aot_executable(dnn SRCS_COMPILE dnn_compile.cc SRCS_RUN dnn_run.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES} TARGET_STRING "host-cuda-cuda_capability_50-profile")
    elseif()
        ion_aot_executable(dnn SRCS_COMPILE dnn_compile.cc SRCS_RUN dnn_run.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES} TARGET_STRING "host-profile")
    endif()
    ion_jit_executable(dnn SRCS dnn.cc INCS ${OpenCV_INCLUDE_DIRS} LIBS ${OpenCV_LIBRARIES})
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io} AND OpenCV_FOUND AND UNIX AND NOT APPLE)
    if(${CUDA_FOUND})
        ion_aot_executable(demo SRCS_COMPILE demo_compile.cc SRCS_RUN demo_run.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES} TARGET_STRING "host-cuda-cuda_capability_50-profile")
    elseif()
        ion_aot_executable(demo SRCS_COMPILE demo_compile.cc SRCS_RUN demo_run.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES} TARGET_STRING "host-profile")
    endif()
    ion_jit_executable(demo SRCS demo.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ion-bb ${OpenCV_LIBRARIES})
endif()

if(${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io} AND OpenCV_FOUND)
    ion_jit_executable(isp SRCS isp.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ${OpenCV_LIBRARIES})
endif()

if(${ION_BB_BUILD_image-processing} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_sgm} AND OpenCV_FOUND)
    ion_jit_executable(isp_and_sgm SRCS isp_and_sgm.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ${OpenCV_LIBRARIES})
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_image-processing} AND OpenCV_FOUND)
    ion_jit_executable(imx219_isp_display_jit SRCS imx219_isp_display.cc)
endif()

if(${ION_BB_BUILD_base} AND ${ION_BB_BUILD_image-io} AND ${ION_BB_BUILD_dnn})
    ion_jit_executable(gender_count SRCS gender_count.cc)
endif()

#if(${ION_BB_BUILD_image-io})
#    ion_jit_executable(u3v_jit SRCS u3v.cc)
#endif()

if(${ION_BB_BUILD_image-io} AND OpenCV_FOUND AND UNIX AND NOT APPLE)
    ion_jit_executable(v4l2_jit SRCS v4l2.cc)
    ion_jit_executable(realsense_jit SRCS realsense.cc INCS ${OpenCV_INCLUDE_DIR} LIBS ${OpenCV_LIBRARIES})
endif()

if(${ION_BB_BUILD_fpga})
    ion_aot_executable(fpga_isp SRCS_COMPILE fpga_isp_compile.cc SRCS_RUN fpga_isp_run.cc TARGET_STRING "arm-64-linux-vivado_hls")
    ion_aot_executable(fpga_dnn SRCS_COMPILE fpga_dnn_compile.cc SRCS_RUN fpga_dnn_run.cc TARGET_STRING "arm-64-linux-vivado_hls-dpu")
endif()

if(MSVC)
    add_definitions(/bigobj)
    message(STATUS "Allow big object for example/")
endif()
