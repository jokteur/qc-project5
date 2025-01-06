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
inline std::string print_filesize(size_t size) {
    if (size < 1024UL) {
        return fmt::format("{}B", size);
    }
    else if (size < 1024UL * 1024UL) {
        return fmt::format("{:.1f}KB", (float)size / 1024.f);
    }
    else if (size < 1024UL * 1024UL * 1024UL) {
        return fmt::format("{:.1f}MB", (float)size / (1024.f * 1024.f));
    }
    else if (size < 1024UL * 1024UL * 1024UL * 1024UL) {
        return fmt::format("{:.1f}GB", (float)size / (1024.f * 1024.f * 1024.f));
    }
    else {
        return fmt::format("{:.1f}TB", (float)size / (1024.f * 1024.f * 1024.f * 1024.f));
    }
}