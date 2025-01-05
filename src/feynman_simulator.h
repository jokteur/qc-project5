#pragma once

#include "simulator.h"

#include <vector>
#include <map>

struct FeynmannSimulator {
    std::map<int, std::vector<Circuit>> feynman_circuits;
    Circuit circuit;
    int num_qubits;
    float fidelity;
    std::vector<int> qubit_divisions;

    FeynmannSimulator(const Circuit& circuit, int num_divisions, float fidelity = 1.0) : circuit(circuit), fidelity(fidelity) {
        num_qubits = circuit.num_qubits;
        qubit_divisions.resize(num_divisions);
        int qubits_per_division = num_qubits / num_divisions;
        int rest = num_qubits % num_divisions;
        for (int i = 0; i < num_divisions; ++i) {
            feynman_circuits[i];
            qubit_divisions[i] = i * qubits_per_division;
            if (i < rest)
                qubit_divisions[i]++;
        }
        for (int i = 1;i < num_divisions; ++i) {
            qubit_divisions[i] += qubit_divisions[i - 1];
        }

        std::vector<Circuit> circuits;
        circuits.resize(1);
        int num_cross_CZ = 0;

        for (const auto& gate : circuit.gates) {
            if (gate.control == -1) {
                for (int j = 0;j < circuits.size(); ++j) {
                    circuits[j].gates.push_back(gate);
                }
            }
            else {
                // Check if the gate is a cross CZ gate (i.e. between two subcircuits)
                bool is_cross_CZ = false;
                for (int i = 0;i < qubit_divisions;i++) {
                    bool is_target = gate.target >= qubit_divisions[i] && gate.target < qubit_divisions[i + 1];
                    bool is_control = gate.control >= qubit_divisions[i] && gate.control < qubit_divisions[i + 1];
                    if (!is_target && is_control || is_target && !is_control) {
                        is_cross_CZ = true;
                        break;
                    }
                }
                if (is_cross_CZ) {
                    
                }
            }
        }
    }
};