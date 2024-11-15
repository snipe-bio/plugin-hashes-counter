cmake_minimum_required(VERSION 3.15)
project(snipe_plugin_hashes_quant LANGUAGES CXX)

# Find Python and Nanobind packages
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Ensure we're running via scikit-build
if (NOT SKBUILD)
    message(WARNING "\
This CMake file is meant to be executed using 'scikit-build'. Running it directly will almost certainly not produce the desired result. If you are a user trying to install this package, please use the command below, which will install all necessary build dependencies, compile the package in an isolated environment, and then install it.
=====================================================================
$ pip install .
=====================================================================
If you are a software developer, and this is your own package, then it is usually much more efficient to install the build dependencies in your environment once and use the following command that avoids a costly creation of a new virtual environment at every compilation:
=====================================================================
$ pip install nanobind scikit-build-core[pyproject]
$ pip install --no-build-isolation -ve .
=====================================================================
You may optionally add -Ceditable.rebuild=true to auto-rebuild when the package is imported. Otherwise, you need to rerun the above after editing C++ files.")
endif()

# Set compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Ofast")

# Add OpenMP support if available (useful for multi-threading)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Check platform (macOS vs. Linux)
if(APPLE)
    # macOS specific settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
elseif(UNIX AND NOT APPLE)
    # Linux specific settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs -lrt")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/parallel-hashmap)

# Create and link the C++ module using nanobind
nanobind_add_module(
    _hashes_counter_impl
    STABLE_ABI
    NB_STATIC
    src/quant_sig.cpp
)

# Installation settings
install(TARGETS _hashes_counter_impl LIBRARY DESTINATION hashes_counter)
