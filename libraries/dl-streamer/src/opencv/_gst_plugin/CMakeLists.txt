# ==============================================================================
# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set(TARGET_NAME "gstdlstreamer_opencv")

file(GLOB_RECURSE ELEMENTS_CPP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../*/elements.cpp")

add_library(${TARGET_NAME} SHARED register_elements.cpp ${ELEMENTS_CPP_FILE})
set_compile_flags(${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
PRIVATE
    dlstreamer_gst
    opencv_barcode_detector
    opencv_cropscale
    opencv_find_contours
    opencv_meta_overlay
    opencv_object_association
    opencv_remove_background
    opencv_tensor_normalize
    tensor_postproc_human_pose
)

# Only on Windows we have OpenCL so there is need to add opencv_warp_affine dependency
if(WIN32)
target_link_libraries(${TARGET_NAME}
PRIVATE
    opencv_warp_affine
)
endif()

install(TARGETS ${TARGET_NAME} DESTINATION ${DLSTREAMER_PLUGINS_INSTALL_PATH})
