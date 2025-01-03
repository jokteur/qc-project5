#pragma once

#include "complex.h"
#include "io/output/output.h"
#include "simulator.h"

#include <fstream>

Circuit read_circuit(const std::string& filename, bool verbose = true, bool skip_hadamard = true) {
    std::ifstream file(filename);
    fmt::println("Reading circuit from file: {}", filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    Circuit circuit;
    file >> circuit.num_qubits;
    while (!file.eof()) {
        Gate gate;

        file >> gate.cycle; 
        if (file.eof())
            break;

        std::string gate_type;
        file >> gate_type;
        gate.type = text_to_gate(gate_type);
        if (gate.type == GateType::CX || gate.type == GateType::CZ) {
            file >> gate.control >> gate.target;
        }
        else {
            file >> gate.target;
        }
        if (skip_hadamard && gate.cycle == 0) {
            continue;
        }
        circuit.gates.push_back(gate);
    }
    return circuit;
}