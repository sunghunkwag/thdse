# THDSE — Core Infrastructure Modules

Two deterministic, mathematical core modules extracted from the original THDSE repository. All probabilistic, swarm-based, and LLM-dependent modules have been permanently purged.

## Module 1: High-Speed Topological Computing Engine (`src/hdc_core/`)

A pure Rust library (with PyO3 bindings) implementing an FHRR (Fourier Holographic Reduced Representation) memory arena for Vector Symbolic Architecture operations:

- **Allocate** complex hypervectors in a zero-GC contiguous arena
- **Inject** phase vectors to initialize hypervector slots
- **Bind** two hypervectors via O(1) complex element-wise multiplication (AVX2 SIMD)
- **Bundle** multiple hypervectors via superposition + unit-magnitude normalization (AVX2 SIMD)
- **Correlate** two hypervectors via cosine similarity in the complex domain

This module is a standalone VSA engine suitable for use in any external meta-architecture requiring high-throughput hyperdimensional computing.

## Module 2: Multi-Layer Structure Dissector (`src/topology/`)

A pure Python preprocessing tool that decomposes source code into a multi-layer directed graph:

- **`TopologicalASTGraphCA`** — Parses source code into an AST-based directed graph with complex-valued node states evolved via a cellular automaton message-passing scheme.
- **`MultiLayerGraphBuilder`** — Extends the AST graph with two additional edge layers:
  - **CFG** (Control Flow Graph): sequential flow, branch targets, loop back-edges
  - **Data-Dep** (Data Dependency): lightweight def-use chains within scope

No encoding logic, no swarm coupling. Pure deterministic graph construction.

## Project Structure

```
thdse/
├── src/
│   ├── hdc_core/              # Rust FHRR Arena (AVX2 SIMD, PyO3 bindings)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── topology/              # Multi-layer graph builder + CA
│       ├── ast_graph_ca.py
│       └── multi_layer_builder.py
├── run_tests.py
├── requirements.txt
└── Cargo.toml
```

## Dependencies

- **Rust**: `pyo3 0.18` (PyO3 extension module), `std::arch::x86_64` (AVX2/FMA intrinsics)
- **Python**: `networkx >= 3.1`, `numpy >= 1.24.0`
- **Build**: `maturin` (for compiling `hdc_core` as a Python extension)

## Usage

```bash
# Build the Rust VSA engine
cd src/hdc_core && maturin develop --release && cd ../..

# Run smoke tests
python run_tests.py
```
