import os
import subprocess
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import YGate
from qiskit.quantum_info import Statevector
import numpy as np
import time

GRCS_folder = "GRCS/inst/rectangular/"


def try_mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def print_header(text: str):
    print(f"\n{'='*len(text)}\n{text}\n{'='*len(text)}")


def generate_circuit_name(circuit_size: str, circuit_ncycles: int, circuit_idx: int):
    return f"cz_v2/{circuit_size}/inst_{circuit_size}_{circuit_ncycles}_{circuit_idx}.txt"


def generate_circuit_qiskit(circuit_size: str, circuit_ncycles: int, circuit_idx: int) -> QuantumCircuit:
    path = GRCS_folder + generate_circuit_name(circuit_size, circuit_ncycles, circuit_idx)

    with open(path, "r") as file:
        nqubits = int(file.readline())

        qc = QuantumCircuit(nqubits)
        for line in file:
            cycle, gate, *qubits = line.split()
            target = nqubits - 1 - int(qubits[0])
            if len(qubits) > 1:
                control = nqubits - 1 - int(qubits[1])
            if gate == "h":
                qc.h(target)
            elif gate == "x":
                qc.x(target)
            elif gate == "y":
                qc.y(target)
            elif gate == "z":
                qc.z(target)
            elif gate == "t":
                qc.t(target)
            elif gate == "x_1_2":
                qc.sx(target)
            elif gate == "y_1_2":
                qc.append(YGate().power(1 / 2), [target])
            elif gate == "cz":
                qc.cz(control, target)
            elif gate == "cx":
                qc.cx(control, target)
    return qc


def run_qcsimulator(
    circuit_size: str,
    circuit_ncycles: int,
    circuit_idx: int,
    feynman: bool,
    fidelity: float = 1.0,
    nbitstrings=-1,
    epsilon=5e-4,
    max_memory=12,
    verbose: False = bool,
    output_file: str = None,
    use_rejection: bool = False,
    cut_at: int = -1,
):
    """
    Run the QCSimulator on the GRCS folder
    """

    qc_path = f"build/qc-simulator"
    if not os.path.exists(qc_path):
        print("Error: qc-simulator not found. Please build the project first.")
        sys.exit(1)

    circuit_name = generate_circuit_name(circuit_size, circuit_ncycles, circuit_idx)

    if not os.path.exists(GRCS_folder + circuit_name):
        print(f"Error: circuit {circuit_name} not found.")
        sys.exit(1)

    try_mkdir("tmp")
    os.chdir("tmp")

    logname = f"log_{circuit_size}_{circuit_ncycles}_{circuit_idx}_feynman{feynman}_out"

    with open(logname, "w") as outfile:
        subprocess.run(
            [
                f"../{qc_path}",
                "-c",
                f"../{GRCS_folder}{circuit_name}",
                "--fidelity",
                str(fidelity),
                "--epsilon",
                str(epsilon),
                "--max_memory",
                str(max_memory),
                "--nbitstrings",
                str(nbitstrings),
                "--verbose=1" if verbose else "--verbose=0",
                f"--output_probabilities={output_file}" if output_file else "",
                "--use_feynman=1" if feynman else "",
                f"--cut_at={cut_at}",
                f"--use_rejection={int(use_rejection)}",
            ],
            stdout=outfile,
        )

    execute_time = 0
    with open(logname, "r") as file:
        for line in file:
            if "Total time" in line or "Simulating all paths" in line:
                execute_time = line.split()[-1]
                break

    if execute_time == 0:
        print("Error: Program hasn't run. Check log.")
        sys.exit(1)

    if "ms" in execute_time:
        execute_time = float(execute_time[:-2]) / 1000
    elif "us" in execute_time:
        execute_time = float(execute_time[:-2]) / 1000000
    elif "ns" in execute_time:
        execute_time = float(execute_time[:-2]) / 1000000000
    else:
        execute_time = float(execute_time[:-1])

    os.chdir("..")
    return execute_time


def read_amplitudes_from_file(filename: str):
    amplitudes = []
    bitstrings = []
    with open(filename, "r") as file:
        for line in file:
            if not line:
                continue
            bitstring, amplitude = line.split(":")
            real, imag = amplitude.split("+")
            bitstrings.append(int(bitstring, base=2))
            amplitudes.append(complex(real=float(real.strip()), imag=float(imag.strip()[:-1])))

    return np.array(bitstrings), np.array(amplitudes)


def porter_thomas_distribution(nqubits: int, bitstrings: np.array, amplitudes: np.array, fidelity):
    probabilities = np.abs(amplitudes) ** 2 

    N = 2**nqubits

    bins = np.linspace(0, 10, 100)

    hist, bin_edges = np.histogram(probabilities * N, bins=bins, density=True)
    x = np.linspace(0, np.max(bin_edges), 1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.semilogy(x, np.exp(-x), "-k", label="Porter-Thomas ideal")
    plt.semilogy(bin_centers, hist, "-", linewidth=3)
    plt.xlim([0, 10])

    plt.show()

    return bitstrings, probabilities


def qiskit_vs_schrodinger(circuit_size, circuit_ncycles, circuit_idx):
    qc = generate_circuit_qiskit(circuit_size, circuit_ncycles, circuit_idx)
    print_header("Qiskit vs Schrodinger")
    print("Circuit:", generate_circuit_name(circuit_size, circuit_ncycles, circuit_idx))

    t = time.time()
    sv = Statevector.from_instruction(qc)
    t_qiskit = time.time() - t
    print(f"Qiskit: {t_qiskit}s")

    t_schr = run_qcsimulator(circuit_size, circuit_ncycles, circuit_idx, False, output_file="vector.txt")
    print(f"Schrodinger: {t_schr}s")
    print(f"Speedup: {t_qiskit/t_schr:.1f}")

    _, amplitudes = read_amplitudes_from_file("tmp/vector.txt")

    qiskit_amplitudes = sv.data
    print("Amplitudes are equal:", np.allclose(qiskit_amplitudes, amplitudes))


def test_qiskit_vs_schrodinger():
    for c in range(10, 27, 5):
        for i in range(0, 2):
            qiskit_vs_schrodinger("4x4", c, i)


def test_qiskit_vs_feynman():
    print_header("Qiskit vs Feynman")
    nsamples = 100000

    qc = generate_circuit_qiskit("4x5", 10, 0)
    t = time.time()
    sv = Statevector.from_instruction(qc)
    t_qiskit = time.time() - t
    print(f"Qiskit: {t_qiskit}s")
    prob = np.abs(sv.data**2)
    sorted_qiskit = np.argsort(prob)[::-1]

    print(f"Number of samples: {nsamples} (out of {2**20})")
    t_feynman = run_qcsimulator(
        "4x5", 10, 0, True, nbitstrings=nsamples, output_file="vector.txt", cut_at=10, use_rejection=True
    )
    print(f"Feynman: {t_feynman}s")
    print(f"Speedup: {t_qiskit/t_feynman:.1f}")

    bitstrings, amplitudes = read_amplitudes_from_file("tmp/vector.txt")
    probabilities = np.abs(amplitudes**2)
    sorted_feynman = np.argsort(probabilities)[::-1]

    for i, j in zip(sorted_feynman[:20], sorted_qiskit[:20]):
        print(f"Qiskit: {j:b}: {prob[j]:.4e} / Feynman: {bitstrings[i]:b}: {probabilities[i]:.4e}")


# Please uncomment to run the different tests and results
# test_qiskit_vs_schrodinger()
# test_qiskit_vs_feynman()


f = 0.1
run_qcsimulator("6x6", 27, 0, True, nbitstrings=1000000, use_rejection=False, fidelity=f, output_file="vector.txt")
bitstrings, amplitudes = read_amplitudes_from_file("tmp/vector.txt")
porter_thomas_distribution(25, bitstrings, amplitudes, f)
