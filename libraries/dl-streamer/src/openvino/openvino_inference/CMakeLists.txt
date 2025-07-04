# ==============================================================================
# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set(TARGET_NAME "openvino_inference")

find_package(OpenVINO REQUIRED)
pkg_search_module(VA va libva REQUIRED)

add_library(${TARGET_NAME} OBJECT openvino_inference.cpp)
set_compile_flags(${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
PRIVATE
    openvino::runtime
    dlstreamer_api
    dlstreamer_logger
)

target_include_directories(${TARGET_NAME}
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../monolithic/inference_backend/include/inference_backend/
    ${VA_INCLUDE_DIRS}
)

install(TARGETS ${TARGET_NAME} DESTINATION ${DLSTREAMER_PLUGINS_INSTALL_PATH})
