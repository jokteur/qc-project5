#pragma once
#include "complex.h"
#include "io/output/output.h"
#include "gates.h"

#include <vector>


struct Circuit {
    int num_qubits;
    std::vector<Gate> gates;
};

class SchrodingerSimulator {
    Kokkos::View<cmplx*> wave;
    size_t sqrt_counter = 0;
    size_t N;
    Circuit circuit;

    template<typename function>
    void apply_1Q_gate(int target, function& gate_func, int sqrt_add = 0) {
        int num_qubits = circuit.num_qubits;
        size_t w_size = 1ull << num_qubits;
        size_t gate_bitmask = (1ull << ((num_qubits - 1) - target));
        size_t offset_idx = 1ull << ((num_qubits - 1) - target);
        size_t idx[2] = { 0, 0 };
        size_t block_idx = 0;

        sqrt_counter += sqrt_add;

        while (block_idx < w_size) {
            if ((block_idx & gate_bitmask) == 0) {
                idx[0] = block_idx;
                idx[1] = offset_idx + block_idx;
                cmplx new_w[2];
                gate_func(wave(idx[0]), wave(idx[1]), new_w);
                wave(idx[0]) = new_w[0];
                wave(idx[1]) = new_w[1];
                ++block_idx;
            }
            else
                block_idx += (block_idx & gate_bitmask);
        }
    }

    void apply_CZ_gate(int ctrl, int target) {
        int num_qubits = circuit.num_qubits;
        size_t w_size = 1ull << num_qubits;
        size_t ctrl_bitmask = (1ull << ((num_qubits - 1) - ctrl));
        size_t target_bitmask = (1ull << ((num_qubits - 1) - target));

        size_t cz_bitmask = ctrl_bitmask | target_bitmask;
        size_t idx = cz_bitmask;
        while (idx < w_size) {
            wave(idx) *= -1.0;
            idx++;
            idx |= cz_bitmask;
        }
    }

    void apply_CX_gate(int ctrl, int target) {
        int num_qubits = circuit.num_qubits;
        size_t w_size = 1ull << num_qubits;
        size_t ctrl_bitmask = (1ull << ((num_qubits - 1) - ctrl));
        size_t target_bitmask = (1ull << ((num_qubits - 1) - target));

        size_t cx_bitmask = ctrl_bitmask | target_bitmask;
        size_t idx = cx_bitmask;
        while (idx < w_size) {
            size_t idx_swap = idx ^ target_bitmask;
            cmplx temp = wave(idx);
            wave(idx) = wave(idx_swap);
            wave(idx_swap) = temp;
            idx++;
            idx |= cx_bitmask;
        }
    }
public:
    SchrodingerSimulator(const Circuit& circuit) :
        circuit(circuit),
        wave("wave", 1 << circuit.num_qubits),
        N(1 << circuit.num_qubits) {
    }

    void initialise_state(bool hadamard = false) {
        if (hadamard) {
            Kokkos::parallel_for(wave.size(), KOKKOS_LAMBDA(size_t idx) {
                wave(idx) = 1.;
            });
            sqrt_counter = N;
        }
        else {
            // In case we are on GPU, we need to use parallel_for to access memory
            Kokkos::parallel_for(1, KOKKOS_LAMBDA(size_t) {
                wave(0) = 1.;
            });
        }
    }

    void run(bool verbose = true) {
        Kokkos::Timer timer;
        for (const auto& gate : circuit.gates) {
            Kokkos::Timer gate_timer;
            switch (gate.type) {
            case GateType::X:
                apply_1Q_gate(gate.target, x_gate);
                break;
            case GateType::Y:
                apply_1Q_gate(gate.target, y_gate);
                break;
            case GateType::Z:
                apply_1Q_gate(gate.target, z_gate);
                break;
            case GateType::H:
                apply_1Q_gate(gate.target, h_gate, 1);
                break;
            case GateType::T:
                apply_1Q_gate(gate.target, t_gate, 1);
                break;
            case GateType::SqrtX:
                apply_1Q_gate(gate.target, sqrt_x_gate, 2);
                break;
            case GateType::SqrtY:
                apply_1Q_gate(gate.target, sqrt_y_gate, 2);
                break;
            case GateType::CX:
                apply_CX_gate(gate.control, gate.target);
                break;
            case GateType::CZ:
                apply_CZ_gate(gate.control, gate.target);
                break;
            }
            if (verbose) {
                Kokkos::fence();
                fmt::print("Cycle: {:>3}, time: {:>10},", gate.cycle, print_time(gate_timer.seconds()));
                if (gate.control == -1)
                    fmt::println(" {} on qubit {:<2}", gate_to_text(gate.type), gate.target);
                else
                    fmt::println(" {} ctrl({:<2}) target({:<2})", gate_to_text(gate.type), gate.control, gate.target);
                
            }
        }

        if (verbose) {
            Kokkos::fence();
            fmt::println("Total time: {}", print_time(timer.seconds()));
        }
    }
};