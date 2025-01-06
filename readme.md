# Installation

```bash
git clone https://github.com/jokteur/qc-project5.git
cd qc-project5
git submodule update --init --recursive
```

# Compilation

```bash
mkdir build
cd build
cmake ..
```

With GPU support (Nvidia, CUDA):
```bash
mkdir build
cd build
cmake .. -DKokkos_ENABLE_CUDA=ON
```

# Results

In root of git, simply:
```
python results.py
```

Need these python libraries installed:
- qiskit
- numpy
- matplotlib