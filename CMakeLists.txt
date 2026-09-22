cmake_minimum_required(VERSION 3.18)

project(decimate
    VERSION 1.3.0
    DESCRIPTION "High-performance multi-platform triangle-mesh decimation (CPU/OpenMP + optional CUDA)"
    LANGUAGES C)

# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
option(DECIMATE_ENABLE_CUDA "Build the CUDA GPU mirror (requires nvcc + CUDA toolkit)" OFF)
option(DECIMATE_BUILD_TESTS "Build unit tests" ON)
option(DECIMATE_BUILD_EXAMPLES "Build example programs" ON)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
#  Backend sources
# ---------------------------------------------------------------------------
set(DECIMATE_SOURCES src/decimate_cpu.c)

if(DECIMATE_ENABLE_CUDA)
    enable_language(CUDA)
    list(APPEND DECIMATE_SOURCES src/decimate_cuda.cu)
    set(DECIMATE_DEFS DECIMATE_HAS_CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;90")
endif()

# ---------------------------------------------------------------------------
#  Library target
# ---------------------------------------------------------------------------
add_library(decimate SHARED ${DECIMATE_SOURCES})
add_library(decimate::decimate ALIAS decimate)

# Name the shared artifact so it does not collide with other "decimate"
# libraries on the machine. Produces decimate_tri_mesh.dll /
# libdecimate_tri_mesh.so / libdecimate_tri_mesh.dylib.
set_target_properties(decimate PROPERTIES OUTPUT_NAME decimate_tri_mesh)

target_include_directories(decimate
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_definitions(decimate PRIVATE ${DECIMATE_DEFS})

# ---- OpenMP ---------------------------------------------------------------
if(NOT DECIMATE_ENABLE_CUDA)
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(decimate PUBLIC OpenMP::OpenMP_C)
        target_compile_definitions(decimate PRIVATE _OPENMP)
        message(STATUS "decimate: OpenMP enabled (CPU parallelism)")
    else()
        message(STATUS "decimate: OpenMP NOT found - building serial CPU path")
    endif()
else()
    # CUDA provides its own parallelism; CPU path still available as fallback.
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(decimate PUBLIC OpenMP::OpenMP_C)
    endif()
    target_link_libraries(decimate PRIVATE CUDA::cudart)
    message(STATUS "decimate: CUDA enabled (GPU mirror + CPU fallback)")
endif()

# ---- Platform-specific flags / linkage -----------------------------------
if(WIN32)
    # Public symbols are exported via __declspec(dllexport) in decimate.h,
    # enabled by DECIMATE_EXPORTS (see DECIMATE_API macro).
    target_compile_definitions(decimate PRIVATE DECIMATE_EXPORTS)
elseif(APPLE)
    target_link_options(decimate PRIVATE "-install_name @rpath/libdecimate_tri_mesh.dylib")
endif()

target_compile_options(decimate PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<C_COMPILER_ID:MSVC>:/W3>)

# ---------------------------------------------------------------------------
#  Install rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS decimate
    EXPORT decimate-targets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES include/decimate.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT decimate-targets
    FILE decimate-config.cmake
    NAMESPACE decimate::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/decimate)

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
if(DECIMATE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
#  Examples
# ---------------------------------------------------------------------------
if(DECIMATE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

message(STATUS "decimate: version ${PROJECT_VERSION}, CUDA=${DECIMATE_ENABLE_CUDA}")
