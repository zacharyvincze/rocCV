# ##############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.24)

# Enable enhanced message
if(NOT DEFINED ENHANCED_MESSAGE)
    string(ASCII 27 Esc)
    set(ColorReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(White       "${Esc}[37m")
    set(Cyan        "${Esc}[36m")
    set(BoldBlue    "${Esc}[1;34m")
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# Check if we are building using the TheRock install of ROCm.
set(USING_THE_ROCK OFF)
if (EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(USING_THE_ROCK ON)
endif()

# Set device library path for TheRock install of ROCm.
if (USING_THE_ROCK)
    if (NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
        set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
    endif()
endif()

# Set HIP_PLATFORM to amd if not set. This is required before finding HIP.
if (NOT DEFINED ENV{HIP_PLATFORM})
    set(ENV{HIP_PLATFORM} "amd")
endif()

# Ensure CMake can find the roccv library and headers.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

set(VERSION "1.0")
project(roccv-samples VERSION ${VERSION} LANGUAGES CXX)

message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${PROJECT_VERSION}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Compiler -- ${CMAKE_CXX_COMPILER}${ColorReset}")

# Define a sample executable. All samples link against roccv by default.
#   SOURCES         Source files; defaults to <NAME>.cpp in the current directory.
#   EXTRA_LIBS      Additional libraries to link.
#   EXTRA_INCLUDES  Additional include directories.
#   REQUIRES        Package names (e.g. rocjpeg) whose <NAME>_FOUND must be TRUE;
#                   samples with unmet requirements are silently skipped.
function(add_sample NAME)
    cmake_parse_arguments(SAMPLE "" "" "SOURCES;EXTRA_LIBS;EXTRA_INCLUDES;REQUIRES" ${ARGN})

    foreach(_req IN LISTS SAMPLE_REQUIRES)
        if (NOT ${_req}_FOUND)
            message("-- ${Yellow}roccv-samples -- sample ${NAME} requires ${_req} but it was not found, skipping${ColorReset}")
            return()
        endif()
    endforeach()

    if (NOT SAMPLE_SOURCES)
        set(SAMPLE_SOURCES "${NAME}.cpp")
    endif()

    message("-- ${Green}roccv-samples -- adding sample ${NAME}${ColorReset}")
    add_executable(${NAME} ${SAMPLE_SOURCES})
    target_link_libraries(${NAME} PRIVATE roccv::roccv ${SAMPLE_EXTRA_LIBS})
    target_include_directories(${NAME} PRIVATE ${SAMPLE_EXTRA_INCLUDES})
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/samples"
        BUILD_RPATH "$<TARGET_FILE_DIR:roccv::roccv>;${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib"
    )
endfunction()

if (NOT TARGET roccv::roccv)
    find_package(roccv QUIET)
else()
    set(roccv_FOUND TRUE)
endif()

find_package(OpenCV QUIET)
find_package(rocjpeg QUIET)

foreach(_pkg IN ITEMS roccv OpenCV rocjpeg)
    if (${_pkg}_FOUND)
        message("-- ${Green}roccv-samples -- found ${_pkg} version: ${${_pkg}_VERSION}${ColorReset}")
    endif()
endforeach()

set(_ocv EXTRA_LIBS ${OpenCV_LIBS} EXTRA_INCLUDES ${OpenCV_INCLUDE_DIRS} REQUIRES roccv OpenCV)

# ================================================
# rocCV sample executables
# ================================================
add_sample(bilateral_filter ${_ocv})
add_sample(custom_crop ${_ocv})
add_sample(center_crop ${_ocv})
add_sample(normalize ${_ocv})
add_sample(bnd_box ${_ocv})
add_sample(copy_make_border ${_ocv})
add_sample(warp_perspective ${_ocv})
add_sample(composite ${_ocv})
add_sample(gamma_contrast ${_ocv})

add_sample(roccv_cropandresize_app
    SOURCES cropandresize/cpp/main.cpp
    EXTRA_LIBS ${OpenCV_LIBS}
    EXTRA_INCLUDES ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
    REQUIRES roccv OpenCV
)

add_sample(rocjpeg_sample
    SOURCES rocjpeg_example/rocjpeg_sample.cpp
    EXTRA_LIBS rocjpeg::rocjpeg ${OpenCV_LIBS}
    EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/rocjpeg_example ${rocjpeg_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
    REQUIRES roccv OpenCV rocjpeg
)