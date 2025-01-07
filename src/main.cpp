#include "kokkos.h"
#include "io/output/output.h"
#include "util/arg_parser.h"
#include <fstream>
#include <Kokkos_UnorderedMap.hpp>

#include "reader.h"
#include "simulator.h"
#include "feynman_simulator.h"

struct Arguments {
    std::string circuit_file;
    bool verbose = true;
    std::string output_statevector;
    std::string output_probabilities;
    int nbitstrings = -1;
    double epsilon = 5e-4;
    int use_feynman = 0;
    bool use_rejection = true;
    int cut_at = -1;
    double fidelity = 1.0;
    int recursive = 0;
    size_t max_memory = 16; // in GB
};

int main(int argc, char* argv[]) {
#ifdef KOKKOS_ENABLE_CUDA
    fmt::println("Using CUDA");
#else
    fmt::println("Using OpenMP");
#endif

    Arguments args;

    Parser arg_parser("Quantum Simulator", "0.1");
    arg_parser.add_argument("-c,--circuit", "Path to the circuit file", args.circuit_file);
    arg_parser.add_argument("-v,--verbose", "Print verbose output", args.verbose);
    arg_parser.add_argument("--output_statevector", "Output the whole statevector to file", args.output_statevector);
    arg_parser.add_argument("--output_probabilities", "Output the probabilities to file", args.output_statevector);
    arg_parser.add_argument("--use_feynman", "Use the Feynman simulator (divide the circuit into n circuits)", args.use_feynman);
    arg_parser.add_argument("--cut_at", "Cut the circuit at a specific qubit (if not specified, automatic)", args.cut_at);
    arg_parser.add_argument("--fidelity", "Fidelity of the Feynman simulator", args.fidelity);
    arg_parser.add_argument("--nbitstrings", "Number of bitstrings (-1 for full vector)", args.nbitstrings);
    arg_parser.add_argument("--use_rejection", "Use rejection sampling", args.use_rejection);
    arg_parser.add_argument("--epsilon", "Epsilon for fidelity of sampling", args.epsilon);
    arg_parser.add_argument("--max_memory", "Maximum memory in GB", args.max_memory);
    arg_parser.add_argument("--recursive", "Recursive Feynman", args.recursive);
    arg_parser.parse_known_args(argc, argv);

    if (args.circuit_file.empty()) {
        fmt::println("Please provide a circuit file");
        arg_parser.print_help();
        return 1;
    }


    Kokkos::initialize(argc, argv);
    {
        Circuit circuit = read_circuit(args.circuit_file, args.verbose, true);

        // Schrodinger simulator
        if (args.use_feynman == 0) {
            SchrodingerSimulator simulator(circuit);
            simulator.initialise_state(true);
            simulator.run(args.verbose);

            fmt::println("Statevector:\n{}", print_statevector(simulator.get_statevector(), 20));

            if (!args.output_statevector.empty()) {
                std::ofstream out(args.output_statevector);
                out << print_statevector(simulator.get_statevector());
            }
            if (!args.output_probabilities.empty()) {
                std::ofstream out(args.output_probabilities);
                out << simulator.print_probabilities();
            }
        }
        // Feynman + Schrödinger simulator
        else {
            fmt::print("Feynman simulator");
            if (args.recursive == 1)
                fmt::println(" (recursive)");
            else 
                fmt::println(" (flat)");
                
            std::random_device dev;
            std::mt19937 rng(dev());
            int seed = rng();

            size_t memory_size = args.max_memory * 1024 * 1024 * 1024;
            FeynmanSimulator simulator(circuit, args.fidelity, memory_size, args.cut_at);
            if (args.nbitstrings < 0 || args.nbitstrings >= (1ull << circuit.num_qubits)) {
                if (memory_size < wave_function_memory_size<precision>(circuit.num_qubits)) {
                    fmt::println("Not enough memory to run the full statevector simulation");
                    return 1;
                }

                Kokkos::View<size_t*> bitstrings("bitstrings", 1ull << circuit.num_qubits);
                Kokkos::parallel_for(bitstrings.extent(0), KOKKOS_LAMBDA(size_t i) { bitstrings(i) = i; });

                Kokkos::View<cmplx*> wave;
                if (args.recursive == 1)
                    wave = simulator.run(bitstrings, args.fidelity, args.verbose);
                else
                    wave = simulator.run_flat(bitstrings, args.fidelity, args.verbose);

                StateVector vector;
                vector.num_qubits = circuit.num_qubits;
                vector.wave = wave;
                fmt::println("Statevector (full):\n{}", print_statevector(vector, 20));
                if (!args.output_statevector.empty()) {
                    std::ofstream out(args.output_statevector);
                    out << print_statevector(vector);
                }
                if (!args.output_probabilities.empty()) {
                    std::ofstream out(args.output_probabilities);
                    out << print_probabilities(vector);
                }
            }
            else if (args.use_rejection) {
                // Implement frugal rejection sampling, from Google's article arXiv:1807.10749v3
                /**
                 * Find M' such that 2exp(-M'/(1-exp(-M'))) < epsilon
                 */
                int M = 1;
                while (2 * std::exp(-M / (1 - std::exp(-M))) >= args.epsilon) {
                    M++;
                }
                fmt::println("For {} bitstrings and epsilon {:.1e}, we have M': {}", args.nbitstrings, args.epsilon, M);
                if (args.nbitstrings * M >= (1ull << circuit.num_qubits)) {
                    fmt::println("Too many samples for the given epsilon. Do you want to run the full simulation?");
                    return 1;
                }

                fmt::println("Seed: {}", seed);

                Kokkos::View<size_t*> accepted_bitstrings("accepted_bitstrings", args.nbitstrings * 2);
                Kokkos::View<cmplx*> accepted_amplitude("accepted_amplitude", args.nbitstrings * 2);
                auto idx_counter = Kokkos::View<size_t*>("incr", 1); // Atomic counter

                long int num_bitstring_left = args.nbitstrings;

                Kokkos::Random_XorShift64_Pool<> random_pool((size_t)seed);

                size_t N = 1ull << circuit.num_qubits;

                Kokkos::UnorderedMap<size_t, int> bitset_map(args.nbitstrings * 2);

                Kokkos::Timer timer;
                size_t total_accepted = 0;
                while (num_bitstring_left > 0) {
                    Kokkos::View<size_t*> bitstrings("bitstrings", args.nbitstrings * M);

                    // Generate m*l distinct bitstrings
                    Kokkos::UnorderedMap<size_t, int> bitset_map_tmp(args.nbitstrings * 2);
                    Kokkos::parallel_for("generate_bitstrings", bitstrings.extent(0), KOKKOS_LAMBDA(size_t i) {
                        auto generator = random_pool.get_state();
                        size_t bit = generator.rand64() % N;
                        while (bitset_map.exists(bit) || bitset_map_tmp.exists(bit)) {
                            bit = generator.rand64() % N;
                        }
                        bitset_map_tmp.insert(bit, 1);
                        random_pool.free_state(generator);
                        bitstrings(i) = bit;
                    });

                    // Running the actual simulation on Feynman paths
                    Kokkos::View<cmplx*> wave;
                    if (args.recursive)
                        wave = simulator.run(bitstrings, args.fidelity, args.verbose);
                    else
                        wave = simulator.run_flat(bitstrings, args.fidelity, args.verbose);

                    auto accepted_counter = Kokkos::View<size_t*>("incr", 1); // Accepted counter
                    // Accept or reject bitstrings with probability min(1, |psi|^2 N / M)
                    Kokkos::parallel_for("accept_reject", bitstrings.extent(0), KOKKOS_LAMBDA(size_t i) {
                        size_t bit = bitstrings(i);
                        cmplx amplitude = wave(i);
                        float probability = Kokkos::abs(amplitude * amplitude);
                        float accept_probability = Kokkos::min(1.f, probability * N / M);
                        auto generator = random_pool.get_state();
                        bool accept = generator.drand() < accept_probability;
                        random_pool.free_state(generator);

                        // This may accept a bit more than the number of bitstrings left
                        // because of parallel execution
                        if (accept && accepted_counter(0) < num_bitstring_left) {
                            size_t idx = Kokkos::atomic_fetch_inc(&idx_counter(0));
                            Kokkos::atomic_inc(&accepted_counter(0));
                            bitset_map.insert(bit, 1);
                            accepted_amplitude(idx) = amplitude;
                            accepted_bitstrings(idx) = bit;
                        }
                    });
                    auto accepted_counter_host = Kokkos::create_mirror_view(accepted_counter);
                    Kokkos::deep_copy(accepted_counter_host, accepted_counter);
                    num_bitstring_left -= accepted_counter_host(0);
                    total_accepted += accepted_counter_host(0);
                    fmt::println("Accepted: {} / {}", args.nbitstrings - num_bitstring_left, args.nbitstrings);
                }

                fmt::println("Total time: {}", print_time(timer.seconds()));

                // Extract 
                Kokkos::View<cmplx*> amplitudes("amplitudes", total_accepted);
                Kokkos::View<size_t*> bitstrings("bitstrings", total_accepted);
                fmt::println("Total accepted: {}", total_accepted);
                Kokkos::parallel_for("extract_amplitudes", amplitudes.extent(0), KOKKOS_LAMBDA(size_t i) {
                    amplitudes(i) = accepted_amplitude(i);
                    bitstrings(i) = accepted_bitstrings(i);
                });

                SampleVector vector{ circuit.num_qubits, bitstrings, amplitudes };

                if (!args.output_statevector.empty()) {
                    std::ofstream out(args.output_statevector);
                    out << print_samplevector(vector);
                }
            }
            else {
                Kokkos::Timer timer;
                fmt::println("Seed: {}", seed);

                Kokkos::View<size_t*> bitstrings("bitstrings", args.nbitstrings);
                Kokkos::View<cmplx*> amplitudes("amplitudes", args.nbitstrings);

                size_t N = 1ull << circuit.num_qubits;
                Kokkos::Random_XorShift64_Pool<> random_pool((size_t)seed);

                // Generate m*l distinct bitstrings
                Kokkos::UnorderedMap<size_t, int> bitset_map_tmp(args.nbitstrings * 2);
                Kokkos::parallel_for("generate_bitstrings", bitstrings.extent(0), KOKKOS_LAMBDA(size_t i) {
                    auto generator = random_pool.get_state();
                    size_t bit = generator.rand64() % N;
                    while (bitset_map_tmp.exists(bit)) {
                        bit = generator.rand64() % N;
                    }
                    bitset_map_tmp.insert(bit, 1);
                    random_pool.free_state(generator);
                    bitstrings(i) = bit;
                });

                // Running the actual simulation on Feynman paths
                    Kokkos::View<cmplx*> wave;
                    if (args.recursive)
                        wave = simulator.run(bitstrings, args.fidelity, args.verbose);
                    else
                        wave = simulator.run_flat(bitstrings, args.fidelity, args.verbose);
                fmt::println("Total time: {}", print_time(timer.seconds()));

                SampleVector vector{ circuit.num_qubits, bitstrings, amplitudes };

                if (!args.output_statevector.empty()) {
                    std::ofstream out(args.output_statevector);
                    out << print_samplevector(vector);
                }
            }
        }
    }
    Kokkos::finalize();
    return 0;
}