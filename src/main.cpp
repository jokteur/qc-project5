#include "kokkos.h"
#include "io/output/output.h"
#include "util/arg_parser.h"

#include "reader.h"
#include "simulator.h"

struct Arguments {
    std::string circuit_file;
    bool verbose = true;
};

int main(int argc, char* argv[]) {
    Arguments args;

    Parser arg_parser("Quantum Simulator", "0.1");
    arg_parser.add_argument("-c,--circuit", "Path to the circuit file", args.circuit_file);
    arg_parser.add_argument("-v,--verbose", "Print verbose output", args.verbose);
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
    }
    Kokkos::finalize();
    return 0;
}