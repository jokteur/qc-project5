#pragma once
#include <string>
#include <fmt/core.h>
#include <fmt/format.h>
#include "types.h"
#include "util/template_util.h"

/// @private
template <typename T>
struct fmt::formatter<T, std::enable_if_t<is_complex<T>(), char>> : fmt::formatter<typename T::value_type> {
    mutable std::string fmt_opt;

    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        fmt_opt.clear();
        bool found_dot = false;
        while(it != end && *it != '}') {
            if (*it == '.') {
                fmt_opt.clear();
                found_dot = true;
            }
            if (*it == '}')
                continue;
            fmt_opt += *it;
            ++it;
        }
        if (!found_dot)
            fmt_opt = "";
        return it;
    }

    auto format(const T& value, format_context& ctx) const {
        std::string fmt_ = "{}";
        if (fmt_opt.size())
            fmt_ = "{:" + fmt_opt + "}";
        std::string out = fmt_ + " + " + fmt_ + "i";
        return fmt::format_to(ctx.out(), out, value.real(), value.imag());
    }
};