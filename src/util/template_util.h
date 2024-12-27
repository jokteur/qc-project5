#pragma once

#ifdef HAS_MPI
#include <mpi.h>
#endif
#include "types.h"

// Is Complex
namespace detail {
    /// @private
    template<typename T>
    struct is_kokkos_complex : std::false_type {};

    /// @private
    template<>
    struct is_kokkos_complex<Kokkos::complex<double>> : std::true_type {};

    /// @private
    template<>
    struct is_kokkos_complex<Kokkos::complex<float>> : std::true_type {};
}

// Template function to check if a type is Kokkos::complex<double> or Kokkos::complex<float>
template<typename T>
constexpr bool is_complex() {
    return detail::is_kokkos_complex<T>::value;
}

namespace detail {
    // Primary template for getting value from tuple by index
    /// @private
    template <size_t N, typename Tuple>
    struct TupleValueGetter_impl {
        static constexpr auto getOrDefault(const Tuple& tuple, int defaultValue) {
            return defaultValue; // Default value when index is out of range
        }
    };

    // Specialization for when index is within range
    /// @private
    template <size_t N, typename... Types>
    struct TupleValueGetter_impl<N, std::tuple<Types...>> {
        static constexpr auto getOrDefault(const std::tuple<Types...>& tuple, int defaultValue) {
            if constexpr (N < sizeof...(Types)) {
                return std::get<N>(tuple); // Get value if index is within range
            }
            else {
                return defaultValue; // Default value when index is out of range
            }
        }
    };
}

// Helper function to get value from tuple by index with default value
template <size_t N, typename... Types>
KOKKOS_INLINE_FUNCTION constexpr auto get(const std::tuple<Types...>& tuple, int defaultValue) {
    return detail::TupleValueGetter_impl<N, std::tuple<Types...>>::getOrDefault(tuple, defaultValue);
}

template <int I, class... Ts>
KOKKOS_INLINE_FUNCTION decltype(auto) get(Ts&&... ts) {
    return std::get<I>(std::forward_as_tuple(ts...));
}

template <typename T>
struct TypeName {
    static const char* Get() {
        return typeid(T).name();
    }
};

/**
 * @brief Removes cv and reference qualifiers on a type
 * 
 * @tparam T Type to remove qualifiers
 */
template< class T >
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>; ///< Type after removal
};  

template <class T>
constexpr
std::string_view
type_name() {
    using namespace std;
#ifdef __clang__
    string_view p = __PRETTY_FUNCTION__;
    return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
    string_view p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return string_view(p.data() + 36, p.size() - 36 - 1);
#  else
    return string_view(p.data() + 49, p.find(';', 49) - 49);
#  endif
#elif defined(_MSC_VER)
    string_view p = __FUNCSIG__;
    return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

// Helper to remove parenthesis from macro arguments
#define __REMOVE_PARENTHESIS(A) __REMOVE_PARENTHESIS_HELPER A
#define __REMOVE_PARENTHESIS_HELPER(...) __VA_ARGS__

// Is Kokkos View
namespace detail {
    /// @private
    template <typename T, typename = void>
    struct is_kokkos_view : std::false_type {
    };

    /// @private
    template <typename T>
    struct is_kokkos_view<T, std::void_t<typename T::memory_space>> : std::true_type {
    };
}

/**
 * @brief Determine if a type is a Kokkos::View or Kokkos::DynRankView (by checking the existence of memory_space)
 *
 * @tparam T Type to test
 */
template <typename T> constexpr bool is_kokkos_view() {
    return detail::is_kokkos_view<typename remove_cvref<T>::type>::value;
}

// Is Kokkos static View
namespace detail {
    /// @private
    template<typename T, typename = void>
    struct has_static_rank : std::false_type {};

    // Partial specialization for when the static function exists.
    /// @private
    template<typename T>
    struct has_static_rank<T, std::void_t<decltype(T::rank())>> : std::true_type {};
}
template <typename T> constexpr bool has_static_rank() {
    return detail::has_static_rank<typename remove_cvref<T>::type>::value;
}