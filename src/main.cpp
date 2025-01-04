#include "kokkos.h"
#include "io/output/output.h"
#include "util/arg_parser.h"
#include <fstream>

#include "reader.h"
#include "simulator.h"

struct Arguments {
    std::string circuit_file;
    bool verbose = true;
    std::string output_statevector;
    std::string output_probabilities;
    int nshots = 1000000;
};

int main(int argc, char* argv[]) {
    Arguments args;

    Parser arg_parser("Quantum Simulator", "0.1");
    arg_parser.add_argument("-c,--circuit", "Path to the circuit file", args.circuit_file);
    arg_parser.add_argument("-v,--verbose", "Print verbose output", args.verbose);
    arg_parser.add_argument("--output_statevector", "Output the whole statevector to file", args.output_statevector);
    arg_parser.add_argument("--nshots", "Number of shots", args.nshots);
    arg_parser.add_argument("--output_probabilities", "Output the probabilities to file", args.output_statevector);
    arg_parser.parse_known_args(argc, argv);

    if (args.circuit_file.empty()) {
        fmt::println("Please provide a circuit file");
        arg_parser.print_help();
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        Circuit circuit = read_circuit(args.circuit_file, args.verbose, true);

        SchrodingerSimulator simulator(circuit);
        simulator.initialise_state(true);
        simulator.run(args.verbose);

        fmt::println("Statevector:\n{}", simulator.print_statevector(20));

        if (!args.output_statevector.empty()) {
            std::ofstream out(args.output_statevector);
            out << simulator.print_statevector();
        }
        if (!args.output_probabilities.empty()) {
            std::ofstream out(args.output_probabilities);
            out << simulator.print_probabilities();
        }
    }
    Kokkos::finalize();
    return 0;
}