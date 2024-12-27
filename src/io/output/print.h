#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string>
#ifdef HAS_MPI
#include <mpi.h>
#endif
#include "print_complex.h"
#include "format_array.h"

inline std::string hline(int length = 70, char c = '=') {
    std::string out;
    for (int i = 0; i < length; i++) {
        out += c;
    }
    return out + "\n";
}

inline std::string header(const std::string& title, bool double_header = true, bool center = true, int min_length = 70) {
    size_t text_size = title.size();
    int left_size = 0;
    if (center) {
        left_size = (min_length - text_size) / 2;
    }
    int right_size = min_length - text_size - left_size;
    if (left_size < 0 || right_size < 0) {
        left_size = 0;
        right_size = 0;
        min_length = text_size;
    }
    std::string out;
    out = hline(min_length);
    out += fmt::format("{}{}{}", std::string(left_size, ' '), title, std::string(right_size, ' '));
    if (double_header)  {
        out += "\n";
        out += hline(min_length);  
    } 
    out += "\n";
    return out;
}

inline std::string warning(const std::string& message) {
    return fmt::format("\033[1;33mWarning: {}\033[0m", message);
}

inline std::string strong_warning(const std::string& message) {
    std::string out = "\033[1;31m";
    out += hline(70, '!');
    out += fmt::format("Warning: {}\n", message);
    out += hline(70, '!');
    out += "\033[0m";
    return out;
}