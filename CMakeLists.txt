cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)

project(qc-simulator)

# ------
# Kokkos
# ------
# set(Kokkos_ENABLE_CUDA ON)
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP")
add_subdirectory(external/kokkos)

# --------------------------
# Format (advanced printing)
# --------------------------
if (NOT TARGET fmt)
    add_subdirectory(external/fmt)
endif()


add_subdirectory(external/argparse)

add_compile_options(-Wno-unused-local-typedefs -Wno-unused-parameter -static-libstdc++)

include_directories(src)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} kokkos fmt::fmt stdc++ argparse)
target_include_directories(${PROJECT_NAME} PUBLIC kokkos fmt::fmt)