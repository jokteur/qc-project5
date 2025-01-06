#pragma once

#include "simulator.h"

#include <vector>
#include <map>
#include <random>


using Amplitude = Kokkos::View<cmplx*>;

KOKKOS_INLINE_FUNCTION cmplx get_amplitude(const Amplitude& wave_1, const Amplitude& wave_2, int num_qubits, int cut_idx, int idx) {
    size_t mask_1 = (1ull << cut_idx) - 1;
    mask_1 = mask_1 << (num_qubits - cut_idx);
    size_t mask_2 = (1ull << (num_qubits - cut_idx)) - 1;
    size_t idx_1 = (idx & mask_1) >> (num_qubits - cut_idx);
    size_t idx_2 = idx & mask_2;
    return wave_1(idx_1) * wave_2(idx_2);
}

struct FeynmanSimulator {
    Circuit global_circuit;
    int cut_idx;
    size_t num_paths;
    int num_xCZ;
    float fidelity;
    int num_qubits;
    size_t max_memory;
    size_t counter = 0;
    size_t N1;
    size_t N2;

    int count_number_of_cross_CZ(int cut) {
        int count = 0;
        for (const auto& gate : global_circuit.gates) {
            if (gate.type == GateType::CZ) {
                if (gate.control < cut && gate.target >= cut || gate.control >= cut && gate.target < cut) {
                    count++;
                }
            }
        }
        return count;
    }

    int find_optimal_cut() {
        int optimal_cut;
        int max_num_xCZ = 1e9;
        bool found = false;
        fmt::println("Finding optimal circuit cut that fits into memory");
        for (int i = 1;i < num_qubits;i++) {
            size_t memory_1 = wave_function_memory_size<precision>(i);
            size_t memory_2 = wave_function_memory_size<precision>(num_qubits - i);
            int num_xCZ = count_number_of_cross_CZ(i);
            if (memory_1 * 4 + memory_2 * 4 <= max_memory) {
                found = true;
                fmt::println("  Cut idx: {}, Number of cross CZ: {}, Memory left: {}, Memory right: {}",
                    i, num_xCZ, print_filesize(memory_1), print_filesize(memory_2));
                if (num_xCZ < max_num_xCZ) {
                    max_num_xCZ = num_xCZ;
                    optimal_cut = i;
                }
            }
        }
        if (!found) {
            throw std::runtime_error("Could not find a cut that fits into memory");
        }

        fmt::println("Optimal cut idx: {} with {} xCZ. Circuit size left: {}. Circuit size right: {}",
            optimal_cut, max_num_xCZ, optimal_cut, num_qubits - optimal_cut);
        num_paths = 1 << max_num_xCZ;
        num_xCZ = max_num_xCZ;
        fmt::println("Number of Feynman paths: {}", num_paths);
        return optimal_cut;
    }

    FeynmanSimulator(const Circuit& global_circuit, float fidelity, size_t max_memory, int cut_at) : global_circuit(global_circuit), fidelity(fidelity), max_memory(max_memory) {
        num_qubits = global_circuit.num_qubits;
        if (cut_at >= 0) {
            cut_idx = cut_at;
            num_xCZ = count_number_of_cross_CZ(cut_idx);
            num_paths = 1 << num_xCZ;
        }
        else {
            cut_idx = find_optimal_cut();
        }

        N1 = 1ull << cut_idx;
        N2 = 1ull << (num_qubits - cut_idx);

        Amplitude accumulator_wave_1("wave_1", N1);
        Amplitude accumulator_wave_2("wave_2", N2);
    }

    void recursive_path(
        std::mt19937& rng,
        float fidelity,
        SchrodingerSimulator& sim_1, SchrodingerSimulator& sim_2,
        int gate_idx, int level, int verbose
    ) {
        if (level == num_xCZ) { // Last level (leaf in tree of paths)
            counter++;
            std::uniform_real_distribution<precision> dist(0.0, 1.0);
            precision r = dist(rng);
            if (r > fidelity) { // Discard path with probability fidelity
                return;
            }
        }

        int diverging_idx = -1;
        bool is_control_in_1;
        bool is_target_in_1;

        for (int i = gate_idx;i < global_circuit.gates.size();i++) {
            auto& gate = global_circuit.gates[i];
            is_target_in_1 = gate.target < cut_idx;
            is_control_in_1 = gate.control < cut_idx;
            if (gate.control == -1) {
                if (is_target_in_1) {
                    sim_1.apply_gate(gate, false);
                }
                else {
                    gate.target -= cut_idx;
                    sim_2.apply_gate(gate, false);
                }
            }
            else {
                // CZ is in left cut
                if (is_target_in_1 && is_control_in_1) {
                    sim_1.apply_gate(gate, false);
                }
                // CZ is in right cut
                else if (!is_target_in_1 && !is_control_in_1) {
                    gate.control -= cut_idx;
                    gate.target -= cut_idx;
                    sim_2.apply_gate(gate, false);
                }
                // CZ is cross cut
                else {
                    if (verbose) {
                        // fmt::print("{:{}}", "", level);
                        // fmt::println("  Diverging path (level {}) at gate {} ctrl({:<2}) target({:<2})",
                        //     level, gate_to_text(gate.type), gate.control, gate.target);
                    }
                    diverging_idx = i;
                    break;
                }
            }
        }

        // We reached a leaf, end of recursion
        if (diverging_idx == -1) {
            fmt::println("  Finishing path {} ({:.1f}%)", counter, 100.0 * counter / num_paths);
            // Add the end of run, add the wave to the accumulator
            sim_1.normalise();
            sim_2.normalise();
            return;
        }
        // Otherwise, we have a diverging path
        auto sim_1_cpy = sim_1.copy();
        auto sim_2_cpy = sim_2.copy();
        auto gate_cpy = global_circuit.gates[diverging_idx];

        // Left path first (replace the ctrl with P0)
        if (is_control_in_1) {
            Gate gate;
            gate.type = GateType::P0;
            gate.target = gate_cpy.control;
            sim_1.apply_gate(gate, false);
        }
        else {
            Gate gate;
            gate.type = GateType::P0;
            gate.target = gate_cpy.control - cut_idx;
            sim_2.apply_gate(gate, false);
        }
        recursive_path(rng, fidelity, sim_1, sim_2, diverging_idx + 1, level + 1, verbose);
        // Right path (replace the ctrl with P1 and target with Z)
        if (is_control_in_1) {
            Gate gate;
            gate.type = GateType::P1;
            gate.target = gate_cpy.control;
            sim_1_cpy.apply_gate(gate, false);
            gate.type = GateType::Z;
            gate.target = gate_cpy.target - cut_idx;
            sim_2_cpy.apply_gate(gate, false);
        }
        else {
            Gate gate;
            gate.type = GateType::Z;
            gate.target = gate_cpy.target;
            sim_1_cpy.apply_gate(gate, false);
            gate.type = GateType::P1;
            gate.target = gate_cpy.control - cut_idx;
            sim_2_cpy.apply_gate(gate, false);
        }
        recursive_path(rng, fidelity, sim_1_cpy, sim_2_cpy, diverging_idx + 1, level + 1, verbose);
    }

    void run(float fidelity, int verbose = true) {
        std::random_device dev;
        std::mt19937 rng(dev());

        Kokkos::Timer timer;

        SchrodingerSimulator simulator_1;
        SchrodingerSimulator simulator_2;
        simulator_1.N = N1;
        simulator_2.N = N2;
        simulator_1.wave = Amplitude("wave_1", N1);
        simulator_2.wave = Amplitude("wave_2", N2);
        simulator_1.circuit.num_qubits = cut_idx;
        simulator_2.circuit.num_qubits = num_qubits - cut_idx;
        simulator_1.initialise_state(true);
        simulator_2.initialise_state(true);

        counter = 0;
        recursive_path(rng, fidelity, simulator_1, simulator_2, 0, 0, verbose);

        if (verbose) {
            Kokkos::fence();
            fmt::println("Total time: {}", print_time(timer.seconds()));
        }
    }

    Kokkos::View<cmplx*> run_flat(const Kokkos::View<size_t*>& bitstrings, float fidelity, int verbose = true) {
        std::random_device dev;
        std::mt19937 rng(dev());

        SchrodingerSimulator simulator_1;
        SchrodingerSimulator simulator_2;
        simulator_1.N = N1;
        simulator_2.N = N2;
        simulator_1.wave = Amplitude("wave_1", N1);
        simulator_2.wave = Amplitude("wave_2", N2);
        simulator_1.circuit.num_qubits = cut_idx;
        simulator_2.circuit.num_qubits = num_qubits - cut_idx;
        simulator_1.initialise_state(true);
        simulator_2.initialise_state(true);

        Kokkos::View<cmplx*> global_wave("global_wave", bitstrings.size());

        Kokkos::Timer timer;
        for (size_t p = 0;p < num_paths;p++) {
            std::uniform_real_distribution<precision> dist(0.0, 1.0);
            precision r = dist(rng);
            if (r > fidelity) { // Discard path with probability fidelity
                continue;
            }
            auto sim_1 = simulator_1.copy();
            auto sim_2 = simulator_2.copy();

            Kokkos::Timer path_timer;

            int xCZ_idx = 0;
            for (int i = 0;i < global_circuit.gates.size();i++) {
                auto gate = global_circuit.gates[i];
                bool is_target_in_1 = gate.target < cut_idx;
                bool is_control_in_1 = gate.control < cut_idx;
                if (gate.control == -1) {
                    if (is_target_in_1) {
                        sim_1.apply_gate(gate, false);
                    }
                    else {
                        gate.target -= cut_idx;
                        sim_2.apply_gate(gate, false);
                    }
                }
                else {
                    if (is_target_in_1 && is_control_in_1) {
                        sim_1.apply_gate(gate, false);
                    }
                    else if (!is_target_in_1 && !is_control_in_1) {
                        gate.control -= cut_idx;
                        gate.target -= cut_idx;
                        sim_2.apply_gate(gate, false);
                    }
                    else {
                        size_t xCZ_mask = 1ull << xCZ_idx;
                        if (xCZ_mask & p) {
                            if (is_control_in_1) {
                                Gate new_gate;
                                new_gate.type = GateType::P0;
                                new_gate.target = gate.control;
                                sim_1.apply_gate(new_gate, false);
                            }
                            else {
                                Gate new_gate;
                                new_gate.type = GateType::P0;
                                new_gate.target = gate.control - cut_idx;
                                sim_2.apply_gate(new_gate, false);
                            }
                        }
                        else {
                            if (is_control_in_1) {
                                Gate new_gate;
                                new_gate.type = GateType::P1;
                                new_gate.target = gate.control;
                                sim_1.apply_gate(new_gate, false);
                                new_gate.type = GateType::Z;
                                new_gate.target = gate.target - cut_idx;
                                sim_2.apply_gate(new_gate, false);
                            }
                            else {
                                Gate new_gate;
                                new_gate.type = GateType::Z;
                                new_gate.target = gate.target;
                                sim_1.apply_gate(new_gate, false);
                                new_gate.type = GateType::P1;
                                new_gate.target = gate.control - cut_idx;
                                sim_2.apply_gate(new_gate, false);
                            }
                        }
                        xCZ_idx++;
                    }
                }
            }

            double time = path_timer.seconds();
            if (verbose) {
                fmt::println("Path {} ({:.0f}%) , ETA {}", p, 100.0 * p / num_paths, print_time(time * (num_paths - p)));
            }
            sim_1.normalise();
            sim_2.normalise();
            Kokkos::parallel_for(bitstrings.extent(0), KOKKOS_CLASS_LAMBDA(size_t i) {
                size_t idx = bitstrings(i);
                auto ampl = get_amplitude(sim_1.wave, sim_2.wave, num_qubits, cut_idx, idx);
                global_wave(i) += ampl;
            });
            // fmt::println("sim_1: {}", print_statevector(sim_1.get_statevector(), 20));
            // fmt::println("sim_2: {}", print_statevector(sim_2.get_statevector(), 20));
            // fmt::println("global_wave: {}", print_statevector({num_qubits, global_wave}, 20));
        }
        fmt::println("Simulating all paths: {}", print_time(timer.seconds()));
        return global_wave;
    }
};