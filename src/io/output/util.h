#pragma once
#include <fmt/core.h>
inline std::string print_time(float seconds) {
    std::string out;
    if (seconds > 1) {
        out = fmt::format("{:.2f}s", seconds);
    }
    else if (seconds * 1000 > 1) {
        out = fmt::format("{:.2f}ms", seconds * 1000.f);
    }
    else if (seconds * 1e6 > 1) {
        out = fmt::format("{:.2f}us", seconds * 1e6);
    }
    else {
        out = fmt::format("{:.2f}ns", seconds * 1e9);
    }
    return out;
}