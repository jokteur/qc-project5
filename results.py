import os
import subprocess
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import YGate
from qiskit.quantum_info import Statevector
from qiskit_aer import QasmSimulator, AerSimulator, StatevectorSimulator
import numpy as np
import time

plt.rcParams.update({'font.size': 20})

GRCS_folder = "GRCS/inst/rectangular/"

qc_path = f"build/qc-simulator"
if not os.path.exists(qc_path):
    print("Error: qc-simulator not found. Please build the project first.")
    sys.exit(1)

out = subprocess.run(["./" + qc_path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if "CUDA" in out.stdout.decode("utf-8"):
    GPU = True
else:
    GPU = False

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
    circuit_name = generate_circuit_name(circuit_size, circuit_ncycles, circuit_idx)

    if not os.path.exists(GRCS_folder + circuit_name):
        print(f"Error: circuit {circuit_name} not found.")
        sys.exit(1)

    try_mkdir("tmp")
    os.chdir("tmp")

    logname = f"log_{circuit_size}_{circuit_ncycles}_{circuit_idx}_feynman{feynman}_gpu{GPU}_out"

    with open(logname, "w") as outfile:
        program_args = " ".join([f"../{qc_path}",
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
                f"--use_rejection={int(use_rejection)}"])
        
        os.system(program_args + " > " + logname)

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


def porter_thomas_distribution(label:str, nqubits: int, bitstrings: np.array, amplitudes: np.array, fidelity):
    probabilities = np.abs(amplitudes) ** 2 / fidelity

    N = 2**nqubits

    bins = np.linspace(0, 10, 100)

    hist, bin_edges = np.histogram(probabilities * N, bins=bins, density=True)
    x = np.linspace(0, np.max(bin_edges), 1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure()
    plt.semilogy(bin_centers, hist, "-", linewidth=3, label=label)
    plt.semilogy(x, np.exp(-x), "--r", label="Porter-Thomas ideal")
    plt.xlim([0, 10])
    if fidelity < 1:
        plt.xlabel("Np/f")
        plt.ylabel("Probability density")
    else:
        plt.xlabel("Np")
        plt.ylabel("Probability density")
    plt.legend(loc="upper right")
    plt.tight_layout()
    import re
    label = re.sub(r'\W+', '', label)
    plt.savefig(f"tmp/porter_thomas_{label}.png")
    # plt.show()

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


def performance_tests():
    print_header("Performance tests")

    circuit_size = ["4x5", "5x5"]
    if not GPU:
        circuit_size.append("5x6")
    circuit_ncycles = 27
    circuit_idx = 5

    for size in circuit_size:
        print(f"Circuit size: {size}")
        qc = generate_circuit_qiskit(size, 10, 0)
        qc.measure_all()
        simulator = StatevectorSimulator(device="CPU")
        circ = transpile(qc, simulator)
        t = time.time()
        simulator.run(circ).result()
        t_qiskit = time.time() - t
        print(f"  Qiskit aer Statevector: {t_qiskit}s")

        t_schr = run_qcsimulator(size, circuit_ncycles, circuit_idx, False)
        print(f"  Schrodinger: {t_schr}s")

def test_thomas_portman():
    print_header("Thomas Portman distribution")
    f = 1
    depth = 34
    size = "4x5"
    nqubits = np.cumprod([int(s) for s in size.split("x")])[-1]
    run_qcsimulator(size, 34, 0, False, output_file="vector.txt", max_memory=12)
    bitstrings, amplitudes = read_amplitudes_from_file("tmp/vector.txt")
    porter_thomas_distribution(f"{size}, depth {depth}", nqubits, bitstrings, amplitudes, f)

    # Now run a bigger circuit, with fidelity
    oneOverF = 64
    f = 1/oneOverF
    size = "6x7"
    depth = 27
    nqubits = np.cumprod([int(s) for s in size.split("x")])[-1]
    # run_qcsimulator(size, depth, 0, True, nbitstrings=1000000, use_rejection=False, fidelity=f, output_file="vector_6x7.txt", max_memory=12)
    bitstrings, amplitudes = read_amplitudes_from_file("tmp/vector_6x7.txt")

    porter_thomas_distribution(f"{size}, depth {depth}, f=1/{oneOverF}", nqubits, bitstrings, amplitudes, f)

# Please uncomment to run the different tests and results
# test_qiskit_vs_schrodinger()
# test_qiskit_vs_feynman()
performance_tests()
if GPU:
    test_thomas_portman()
