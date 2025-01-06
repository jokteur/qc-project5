#pragma once
#include "complex.h"
#include "io/output/output.h"
#include "gates.h"

#include <vector>

template<typename T>
size_t wave_function_memory_size(int circuit_size) {
    return (1ull << circuit_size) * sizeof(T) * 2;
}

struct Circuit {
    int num_qubits;
    int depth;
    std::vector<Gate> gates;
};

struct StateVector {
    int num_qubits;
    Kokkos::View<cmplx*> wave;
};

struct SampleVector {
    int num_qubits;
    Kokkos::View<size_t*> bitstrings;
    Kokkos::View<cmplx*> wave;
};

std::string print_statevector(const StateVector& vector, int first_N = -1) {
    Kokkos::fence();
    std::string out;
    Kokkos::View<cmplx*, Kokkos::HostSpace> wave_host = Kokkos::create_mirror_view(vector.wave);
    size_t N = vector.wave.extent(0);
    Kokkos::deep_copy(wave_host, vector.wave);
    for (size_t i = 0; i < N; ++i) {
        out += fmt::format("{:0{}b}: {}\n", i, vector.num_qubits, wave_host(i));
        if (first_N > 0 && i >= first_N) {
            out += fmt::format("...\n");
            break;
        }
    }
    return out;
}

std::string print_samplevector(const SampleVector& vector, int first_N = -1) {
    Kokkos::fence();
    std::string out;
    Kokkos::View<cmplx*, Kokkos::HostSpace> wave_host = Kokkos::create_mirror_view(vector.wave);
    Kokkos::View<size_t*, Kokkos::HostSpace> bitstring_host = Kokkos::create_mirror_view(vector.bitstrings);
    size_t N = vector.wave.extent(0);
    Kokkos::deep_copy(wave_host, vector.wave);
    Kokkos::deep_copy(bitstring_host, vector.bitstrings);
    for (size_t i = 0; i < N; ++i) {
        out += fmt::format("{:0{}b}: {}\n", bitstring_host(i), vector.num_qubits, wave_host(i));
        if (first_N > 0 && i >= first_N) {
            out += fmt::format("...\n");
            break;
        }
    }
    return out;
}

std::string print_probabilities(const StateVector& vector, int first_N = -1) {
    Kokkos::fence();
    std::string out;
    Kokkos::View<precision*> probs("probs", vector.wave.extent(0));
    Kokkos::parallel_for(vector.wave.extent(0), KOKKOS_LAMBDA(size_t idx) {
        probs(idx) = Kokkos::abs(vector.wave(idx) * vector.wave(idx));
    });
    Kokkos::View<precision*, Kokkos::HostSpace> probs_host = Kokkos::create_mirror_view(probs);
    size_t N = vector.wave.extent(0);
    Kokkos::deep_copy(probs_host, probs);
    for (size_t i = 0; i < N; ++i) {
        out += fmt::format("{:0{}b}: {}\n", i, vector.num_qubits, probs(i));
        if (first_N > 0 && i >= first_N) {
            out += fmt::format("...\n");
            break;
        }
    }
    return out;
}

struct SchrodingerSimulator {
    Kokkos::View<cmplx*> wave;
    size_t sqrt_counter = 0;
    size_t N;
    Circuit circuit;

    /**
     * Apply a 1-qubit gate to the wavefunction
     *
     * @param target The target qubit
     * @param gate_func The function that applies the gate
     * @param sqrt_add The number of square roots to add to the counter
     *
     * We want to multiply the wavefunction by the gate matrix. Let's
     * say that we have the matrix G:
     *    a b
     *    c d
     *
     * When we apply the gate to a target k qubit, the following matrix is
     * applied:
     *
     *      k - 1       k    n - k - 1
     *  /           \   |   /         \
     *  I x I x ... I x G x I x ... x I
     *
     * Let us take n = 3. Here are the three possible cases:
     *
     * I x I x G : target qubit 2, offset = 1
     * ======================================
     * a b . . . . . .  block 1
     * c d . . . . . .  |
     * . . a b . . . .  block 2
     * . . c d . . . .  |
     * . . . . a b . .  block 3
     * . . . . c d . .  |
     * . . . . . . a b  block 4
     * . . . . . . c d  |
     *
     * thread_idx: 0, 1, 2, 3
     * block_idx:  0, 2, 4, 6
     *
     *
     * I x G x I : target qubit 1, offset = 2
     * ======================================
     * a . b . . . . .  block 1
     * . a . b . . . .  |  block 2
     * c . d . . . . .  |  |
     * . c . d . . . .     |
     * . . . . a . b .  block 3
     * . . . . . a . b  |  block 4
     * . . . . c . d .  |  |
     * . . . . . c . d     |
     *
     * thread_idx: 0, 1, 2, 3
     * block_idx:  0, 1, 4, 5
     *
     * G x I x I : target qubit 0, offset = 4
     * ======================================
     * a . . . b . . .  block 1
     * . a . . . b . .  |  block 2
     * . . a . . . b .  |  |  block 3
     * . . . a . . . b  |  |  |  block 4
     * c . . . d . . .  |  |  |  |
     * . c . . . d . .     |  |  |
     * . . c . . . d .        |  |
     * . . . c . . . d           |
     *
     * thread_idx: 0, 1, 2, 3
     * block_idx:  0, 1, 2, 3
     *
     * We just need to map the thread_idx to the block_idx and apply the gate
     * to the block (a, b) and (c, d)
     *
     * The formula for block_idx is:
     * block_idx = idx + offset * floor(idx / 2)
     *           = 2 * thread_idx - (thread_idx % offset)
     */
     // Because function pointers are difficult in CUDA, we need to use macros
#define apply_1Q_gate(target, gate_func, sqrt_add) {         \
        int num_qubits = circuit.num_qubits;                 \
        size_t nblocks = 1ull << num_qubits - 1;             \
        size_t offset = 1ull << ((num_qubits - 1) - target); \
        sqrt_counter += sqrt_add;                            \
        Kokkos::parallel_for(nblocks, KOKKOS_CLASS_LAMBDA(size_t i) { \
            size_t block_idx = 2 * i - (i % offset);                  \
            size_t idx[2] = { block_idx, block_idx + offset };        \
            cmplx new_w[2];                                           \
            gate_func(wave(idx[0]), wave(idx[1]), new_w);             \
            wave(idx[0]) = new_w[0];                                  \
            wave(idx[1]) = new_w[1];                                  \
        });                                                           \
    } 

    /** Optimised T gate */
    void apply_T_gate(int target, int sqrt_add = 0) {
        int num_qubits = circuit.num_qubits;
        size_t nblocks = 1ull << num_qubits - 1;            // 2^(num_qubits - 1)
        size_t offset = 1ull << ((num_qubits - 1) - target); // 2^(num_qubits - 1 - target)

        cmplx j = cmplx(0, 1);
        Kokkos::parallel_for(nblocks, KOKKOS_CLASS_LAMBDA(size_t i) {
            size_t block_idx = 2 * i - (i % offset);
            size_t idx[2] = { block_idx, block_idx + offset };
            wave(idx[1]) = wave(idx[1]) * (1 + j) / Kokkos::sqrt(2);
        });
    }

    void apply_CZ_gate(int ctrl, int target) {
        int num_qubits = circuit.num_qubits;
        size_t nthreads = 1ull << (num_qubits - 2);
        size_t left = ctrl;
        size_t right = target;

        // Convert from left-most significant to right-most significant
        left = num_qubits - 1 - left;
        right = num_qubits - 1 - right;
        if (left < right) {
            std::swap(left, right);
        }
        size_t gap_size = left - right - 1;

        size_t right_mask = (1ull << right) - 1;
        size_t middle_mask = (1ull << gap_size) - 1;

        Kokkos::parallel_for(nthreads, KOKKOS_CLASS_LAMBDA(size_t i) {
            size_t right_bits = i & right_mask;

            // Discard the rightmost bit
            size_t leftover = (i >> right);

            size_t middle_bits = (leftover & middle_mask);

            size_t left_bits = leftover >> gap_size;

            size_t idx = (left_bits << 1) | 1;
            idx = (idx << (gap_size)) | middle_bits;
            idx = (idx << 1) | 1;
            idx = (idx << right) | right_bits;

            wave(idx) *= -1.0;
        });
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

    SchrodingerSimulator(const Circuit& circuit) : circuit(circuit),
        wave("wave", 1 << circuit.num_qubits),
        N(1 << circuit.num_qubits) {
    }

    SchrodingerSimulator() = default;

    SchrodingerSimulator copy() {
        SchrodingerSimulator copy(*this);
        copy.wave = Kokkos::View<cmplx*>("wave", N);
        Kokkos::deep_copy(copy.wave, wave);
        return copy;
    }

    void initialise_state(bool hadamard = false) {
        if (hadamard) {
            precision factor = 1. / Kokkos::pow(Kokkos::sqrt(2), circuit.num_qubits);
            Kokkos::parallel_for(N, KOKKOS_CLASS_LAMBDA(size_t idx) { wave(idx) = 1.; });
            sqrt_counter = circuit.num_qubits;
        }
        else {
            // In case we are on GPU, we need to use parallel_for to access memory
            Kokkos::parallel_for(1, KOKKOS_CLASS_LAMBDA(size_t) { wave(0) = 1.; });
        }
    }

    void apply_gate(const Gate& gate, bool verbose) {
        Kokkos::Timer gate_timer;
        switch (gate.type) {
        case GateType::X:
            apply_1Q_gate(gate.target, x_gate, 0);
            break;
        case GateType::Y:
            apply_1Q_gate(gate.target, y_gate, 0);
            break;
        case GateType::Z:
            apply_1Q_gate(gate.target, z_gate, 0);
            break;
        case GateType::H:
            apply_1Q_gate(gate.target, h_gate, 1);
            break;
        case GateType::T:
            apply_T_gate(gate.target, 1);
            break;
        case GateType::P0:
            apply_1Q_gate(gate.target, p0_gate, 0);
            break;
        case GateType::P1:
            apply_1Q_gate(gate.target, p1_gate, 0);
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

    void normalise() {
        Kokkos::parallel_for(N, KOKKOS_CLASS_LAMBDA(size_t idx) { wave(idx) /= Kokkos::pow(Kokkos::sqrt(2), sqrt_counter); });
    }

    void run(bool verbose = true) {
        Kokkos::Timer timer;
        for (const auto& gate : circuit.gates) {
            apply_gate(gate, verbose);
        }

        normalise();

        if (verbose) {
            Kokkos::fence();
            fmt::println("Total time: {}", print_time(timer.seconds()));
        }
    }

    StateVector get_statevector() {
        StateVector vector;
        vector.num_qubits = circuit.num_qubits;
        vector.wave = wave;
        return vector;
    }

    Kokkos::View<cmplx*> get_probabilities() {
        Kokkos::View<cmplx*> probs("probs", N);
        Kokkos::parallel_for(N, KOKKOS_CLASS_LAMBDA(size_t idx) { probs(idx) = Kokkos::abs(wave(idx) * wave(idx)); });
        return probs;
    }

    std::string print_probabilities(int first_N = -1) {
        Kokkos::fence();
        std::string out;
        Kokkos::View<cmplx*> probs = get_probabilities();
        Kokkos::View<cmplx*, Kokkos::HostSpace> h_probs = Kokkos::create_mirror_view(probs);
        Kokkos::deep_copy(h_probs, probs);
        for (size_t i = 0; i < N; ++i) {
            out += fmt::format("{:0{}b}: {}\n", i, circuit.num_qubits, probs(i));
            if (first_N > 0 && i >= first_N) {
                out += fmt::format("...\n");
                break;
            }
        }
        return out;
    }
};