#pragma once
#include "argparse/argparse.hpp"

template<bool flag = false> void argument_match() {
    static_assert(flag, "argument: no matching type found");
}
template<typename T>
struct Argument {
    argparse::Argument arg;
    T& var;
    std::string name;
    std::function<void(const std::string&)> action;
};

bool stob(const std::string& str) {
    // To lower case
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return std::tolower(c); });
    if (lower == "false" || lower == "0" || lower == "no" || lower == "n")
        return false;
    else if (lower == "true" || lower == "1" || lower == "yes" || lower == "y")
        return true;
    else
        throw std::runtime_error("stob: unknown boolean value");
}

template<typename T>
std::vector<T> to_vector(const std::string& str) {
    std::vector<T> vec;
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        vec.push_back((T)std::stod(token));
    }
    return vec;
}

struct Parser {
    std::vector<Argument<int>> args_custom;
    std::vector<Argument<int>> args_i64;
    std::vector<Argument<size_t>> args_u64;
    std::vector<Argument<bool>> args_bool;
    std::vector<Argument<double>> args_f64;
    std::vector<Argument<std::string>> args_str;
    std::vector<Argument<std::vector<double>>> args_vec_f64;

    argparse::ArgumentParser parser;
    int null_ref;

    Parser(const std::string& name, const std::string& version) : parser(name, version) {}

    void parse_known_args(int argc, char** argv) {
        parser.parse_known_args(argc, argv);
        for (auto& arg : args_custom) {
            if (auto value = parser.present(arg.name)) {
                arg.action(parser.get<std::string>(arg.name));
            }
        }
        for (auto& arg : args_i64) {
            if (auto value = parser.present(arg.name)) {
                arg.var = (int)std::stod(*value);
            }
        }
        for (auto& arg : args_u64) {
            if (auto value = parser.present(arg.name)) {
                arg.var = (size_t)std::stod(*value);
            }
        }
        for (auto& arg : args_bool) {
            if (auto value = parser.present(arg.name)) {
                arg.var = stob(*value);
            }
        }
        for (auto& arg : args_f64) {
            if (auto value = parser.present(arg.name)) {
                arg.var = std::stod(*value);
            }
        }
        for (auto& arg : args_str) {
            if (auto value = parser.present(arg.name)) {
                arg.var = *value;
            }
        }
        for (auto& arg : args_vec_f64) {
            if(auto value = parser.present(arg.name)) {
                arg.var = to_vector<double>(parser.get<std::string>(arg.name));
            }
        }
    }

    std::pair<std::string, std::string> _split_name(const std::string& name) {
        if (name.find(",") != std::string::npos) {
            return { name.substr(0, name.find(",")), name.substr(name.find(",") + 1) };
        }
        else {
            return { name, "" };
        }
    }

    argparse::Argument& _get_basic_arg(const std::string& name, const std::string& help) {
        if (name.find(",") != std::string::npos) {
            auto short_name = name.substr(0, name.find(","));
            auto long_name = name.substr(name.find(",") + 1);
            return parser.add_argument(short_name, long_name).help(help);
        }
        else {
            return parser.add_argument(name).help(help);
        }
    }

    template<typename T>
    void _push_arg(argparse::Argument& arg, T& var, const std::string& name) {
        if constexpr (std::is_same_v<T, bool>) {
            args_bool.push_back({ arg, var, name, nullptr });
        }
        else if constexpr (std::is_same_v<T, int>) {
            args_i64.push_back({ arg, var, name, nullptr });
        }
        else if constexpr (std::is_same_v<T, size_t>) {
            args_u64.push_back({ arg, var, name, nullptr });
        }
        else if constexpr (std::is_same_v<T, double>) {
            args_f64.push_back({ arg, var, name, nullptr });
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            args_str.push_back({ arg, var, name, nullptr });
        }
        else if constexpr (std::is_same_v<T, std::vector<double>>) {
            args_vec_f64.push_back({ arg, var, name, nullptr });
        }
        else {
            argument_match();
        }
    }

    void add_argument(const std::string& name, const std::string& help, std::function<void(const std::string&)> action) {
        auto arg = _get_basic_arg(name, help);
        auto names = _split_name(name);
        args_custom.push_back({ arg, null_ref, names.first, action });
    }
    template<typename T>
    void add_argument(const std::string& name, const std::string& help, T& var) {
        auto arg = _get_basic_arg(name, help);
        auto names = _split_name(name);
        _push_arg(arg, var, names.first);
    }
    /**
     * @brief Add a flag argument to the parser
     * 
     * @note flags always store their value into var, use with caution
    */
    template<typename T>
    void add_flag(const std::string& name, const std::string& help, bool& var) {
        auto arg = _get_basic_arg(name, help).flag().store_into(var);
    }
};