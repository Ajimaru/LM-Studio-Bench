cmake_minimum_required(VERSION 3.24)
project(lm_studio_gpu_probe VERSION 1.0 LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "gpu_probe requires Windows 10/11 and an x64 compiler.")
endif()

# The SDKs are NOT covered by this repository's MIT license. Keep their
# sources in the build tree, never vendor them or redistribute driver DLLs.
# ADLX section 2(c)/3 permits only object-code distribution with an EULA.
# Intel's LICENSE restricts its SDK to Intel platforms and requires notices.
option(GPU_PROBE_ACCEPT_VENDOR_LICENSES
    "Accept the AMD ADLX and Intel IGCL SDK licenses for a local source build"
    OFF)
if(NOT GPU_PROBE_ACCEPT_VENDOR_LICENSES)
    message(FATAL_ERROR
        "Review the vendor licenses before building:\n"
        "https://github.com/GPUOpen-LibrariesAndSDKs/ADLX/blob/"
        "32b5a740d42295c5dfe9026b9f52683da0f3af91/"
        "ADLX%20SDK%20License%20Agreement.pdf\n"
        "https://github.com/intel/drivers.gpu.control-library/blob/"
        "b6c462933502e13d1537dd5024949a51be30e63d/LICENSE\n"
        "Then configure with -DGPU_PROBE_ACCEPT_VENDOR_LICENSES=ON.\n"
        "This does not authorize general binary/SDK redistribution.")
endif()

include(FetchContent)
# ADLX 2.0.0.125; IGCL API 1.1. Hashes pin the complete official archives.
set(adlx_revision "32b5a740d42295c5dfe9026b9f52683da0f3af91")
set(igcl_revision "b6c462933502e13d1537dd5024949a51be30e63d")
string(CONCAT adlx_url
    "https://codeload.github.com/GPUOpen-LibrariesAndSDKs/ADLX/"
    "tar.gz/${adlx_revision}")
string(CONCAT igcl_url
    "https://codeload.github.com/intel/drivers.gpu.control-library/"
    "tar.gz/${igcl_revision}")
FetchContent_Declare(adlx
    URL "${adlx_url}"
    URL_HASH
    "SHA256=64287574aa04623be989a1c6152f10e1b2aa43fab0e4f51601c22649669ad49b"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR gpu_probe_no_subproject)
FetchContent_Declare(igcl
    URL "${igcl_url}"
    URL_HASH
    "SHA256=88f4ebeacdea59a562e911cee1db2fa751b28460611ec71067ce1abe74047055"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR gpu_probe_no_subproject)
FetchContent_MakeAvailable(adlx igcl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()

add_library(adlx_sdk STATIC
    "${adlx_SOURCE_DIR}/SDK/ADLXHelper/Windows/Cpp/ADLXHelper.cpp"
    "${adlx_SOURCE_DIR}/SDK/Platform/Windows/WinAPIs.cpp")
target_include_directories(adlx_sdk SYSTEM PUBLIC
    "${adlx_SOURCE_DIR}/SDK/Include"
    "${adlx_SOURCE_DIR}/SDK/ADLXHelper/Windows/Cpp")
target_compile_definitions(adlx_sdk PUBLIC
    UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)

add_library(igcl_sdk INTERFACE)
target_include_directories(igcl_sdk SYSTEM INTERFACE
    "${igcl_SOURCE_DIR}/include")

add_library(probe_core STATIC
    probe_types.cpp vendor_status.cpp probe_windows.cpp
    amd_sensors.cpp intel_sensors.cpp intel_loader.cpp)
target_link_libraries(probe_core PUBLIC
    adlx_sdk igcl_sdk dxgi d3d12 gdi32 pdh)
target_include_directories(probe_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(gpu_probe main.cpp)
target_link_libraries(gpu_probe PRIVATE probe_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(gpu_probe_tests probe_tests.cpp)
    target_link_libraries(gpu_probe_tests PRIVATE probe_core)
    add_test(NAME gpu_probe_unit COMMAND gpu_probe_tests)
    add_test(NAME gpu_probe_energy_gap
        COMMAND gpu_probe_tests --energy-gap)
    add_test(NAME gpu_probe_loader
        COMMAND gpu_probe_tests --intel-loader)
    set_tests_properties(gpu_probe_unit gpu_probe_energy_gap gpu_probe_loader
        PROPERTIES TIMEOUT 10)
endif()

if(MSVC)
    target_compile_options(adlx_sdk PRIVATE /W0 /utf-8)
    foreach(target probe_core gpu_probe)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    endforeach()
    if(BUILD_TESTING)
        target_compile_options(gpu_probe_tests PRIVATE
            /W4 /permissive- /utf-8)
    endif()
endif()

# Preserve the licenses alongside LOCAL build output. There is deliberately
# no install/packaging rule: AMD binary redistribution needs a suitable EULA.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/licenses")
configure_file("${adlx_SOURCE_DIR}/ADLX SDK License Agreement.pdf"
    "${CMAKE_BINARY_DIR}/bin/licenses/ADLX SDK License Agreement.pdf"
    COPYONLY)
configure_file("${igcl_SOURCE_DIR}/LICENSE"
    "${CMAKE_BINARY_DIR}/bin/licenses/IGCL-LICENSE.txt" COPYONLY)
file(WRITE "${CMAKE_BINARY_DIR}/bin/licenses/NOTICE.txt"
    "Local source build; vendor SDKs are not MIT-licensed.\n"
    "AMD ADLX:\n"
    "Copyright Advanced Micro Devices, Inc. All rights reserved.\n"
    "SDK revision: ${adlx_revision}\n"
    "Intel IGCL:\n"
    "Copyright (C) 2025 Intel Corporation\n"
    "SDK revision: ${igcl_revision}\n"
    "No driver DLLs are distributed. Drivers must already be installed.\n"
    "Local-build SDK-license consent is not redistribution authorization.\n"
    "Review the attached licenses before any redistribution. ADLX requires\n"
    "object-code-only distribution and an appropriate end-user agreement.\n")
