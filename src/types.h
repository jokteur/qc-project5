/**
 * @file types.h
 *
 * Types contains the various kokkos views used throughout the code.
*/
#pragma once
#include "kokkos.h"

#include <utility>


// Macros to work around the fact that std::max/min is not available on GPUs
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)>0?(a):-(a))

// Memory traits
typedef Kokkos::MemoryTraits<Kokkos::RandomAccess> MemTrait_rnd;
typedef Kokkos::HostSpace Mem_HOST;

// Execution spaces
typedef Kokkos::DefaultExecutionSpace ExecSpace;
typedef Kokkos::RangePolicy<ExecSpace> RangePolicy;
typedef Kokkos::DefaultHostExecutionSpace HostExecSpace;
typedef Kokkos::RangePolicy<HostExecSpace> HostRangePolicy;


#define SLICE(start, end) std::pair<int, int>(start, end)

// For other system, add flags
#ifdef KOKKOS_ENABLE_CUDA
#define ENABLE_GPU
#endif


#define NOCLASS_LAMBDA(...) [ __VA_ARGS__ ] __host__ __device__

// SIMD operations
typedef Kokkos::Experimental::native_simd<double> simd_float;

typedef Kokkos::View<size_t*> Dimensions;


#define __1D_RANGE_POLICY(N, exec) \
    Kokkos::RangePolicy<exec>(0, N)
#define __2D_RANGE_POLICY(N1, N2, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec>({ (size_t)0, (size_t)0 }, { (size_t)N1, (size_t)N2 })
#define __3D_RANGE_POLICY(N1, N2, N3, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec>({ (size_t)0, (size_t)0, (size_t)0 }, { (size_t)N1, (size_t)N2, (size_t)N3 })
#define __4D_RANGE_POLICY(N1, N2, N3, N4, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, exec>({ (size_t)0, (size_t)0, (size_t)0, (size_t)0 }, { (size_t)N1, (size_t)N2, (size_t)N3, (size_t)N4 })
#define __1D_RANGE_POLICY_EXT(from, to, exec) \
    Kokkos::RangePolicy<exec>((size_t)from, (size_t)to)
#define __2D_RANGE_POLICY_EXT(from1, to1, from2, to2, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec>({ (size_t)from1, (size_t)from2 }, { (size_t)to1, (size_t)to2 })
#define __3D_RANGE_POLICY_EXT(from1, to1, from2, to2, from3, to3, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec>({ (size_t)from1, (size_t)from2, (size_t)from3 }, { (size_t)to1, (size_t)to2, (size_t)to3 })
#define __4D_RANGE_POLICY_EXT(from1, to1, from2, to2, from3, to3, from4, to4, exec) \
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, exec>({ (size_t)from1, (size_t)from2, (size_t)from3, (size_t)from4 }, { (size_t)to1, (size_t)to2, (size_t)to3, (size_t)to4 })
