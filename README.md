# Topological-Holographic Dynamic Swarm Engine (THDSE)

A deterministic, non-statistical framework for autonomous software engineering through Recursive Self-Improvement (RSI), grounded in Vector Symbolic Architectures (VSA) and algebraic topology.

## Architecture

The engine operates as a strict mathematical pipeline comprising five stages:

1. **Topological AST Graph Construction** — Source code is parsed into an Abstract Syntax Tree via Python's `ast` module and projected onto a directed graph with three edge layers: structural (AST parent-child), control-flow (CFG sequential/branch edges), and data-dependency (def-use chains).

2. **Holographic Hypervector Encoding** — Each graph node is mapped to a 1024-dimensional complex FHRR (Fourier Holographic Reduced Representation) hypervector via element-wise phase binding and bundling, executed through a custom Rust AVX2 SIMD arena with zero garbage-collection overhead.

3. **Autophagic Swarm Mutation** — A pool of 20–100 concurrent typists generates candidate code mutations. In the V2 pipeline, a two-phase strategy is employed: Phase 1 performs undirected mutation; Phase 2 injects resonance feedback (per-axiom similarity scores projected into natural-language constraints) to steer subsequent mutations away from known bug signatures.

4. **Axiom Vector Database & Auto-Axiomatization** — Bug signatures are stored as phase-difference vectors. The V2 engine supports unsupervised axiom discovery: given `(buggy, fixed)` code pairs, novel divergence vectors are automatically mined and registered when their cosine distance from existing axioms exceeds a novelty threshold.

5. **Holographic Verification & Collapse** — All surviving mutations are superimposed in the FHRR arena. The candidate with the lowest maximum resonance against all known axioms (i.e., the vector most distant from every known bug signature) is selected as the output.

## Experimental Results

### Protocol

Three canonical Python bug patterns were tested using the offline rule-based AST mutation backend (no external LLM API required). Each test instantiates an independent FHRR arena (`dim=1024, capacity=50,000`) and executes the full V2 Resonance-Guided Autophagy pipeline.

### Results

| Bug Pattern | Input | Output | Detection Mechanism |
|---|---|---|---|
| Off-by-one indexing | `lst[len(lst)]` | `lst[len(lst) - 1]` | `ast.Subscript` analysis: index expression matches `len(collection)` on same variable |
| Null dereference | `user.name.upper()` | `if user is not None: return user.name.upper()` | `ast.Return` → `ast.Attribute` chain traversal to identify unguarded root variables |
| Missing recursion base case | `return n * factorial(n-1)` | `if n <= 1: return 1` inserted | `ast.FunctionDef` self-call detection without preceding `ast.If` guard |

All three bugs were detected and patched autonomously (Exit code 0) through deterministic AST analysis combined with HDC vector-space filtering.

### V1 → V2 Improvements

| Metric | V1 | V2 |
|---|---|---|
| Graph edge layers | 1 (AST only) | 3 (AST + CFG + Data-Dep) |
| Axiom registration | Manual | Unsupervised mining from `(buggy, fixed)` pairs |
| Mutation strategy | Blind brute-force (100 typists) | Two-phase resonance-guided (20 typists) |
| Axioms auto-mined from 3 seed pairs | 0 | 3 |

## Limitations

The current system operates on **syntactic topology** and cannot detect bugs that require runtime value reasoning (e.g., type-dependent invariants, concurrency hazards). The rule-based AST fallback is pattern-matched; generalization to unseen bug morphologies requires either LLM integration or expansion of the `NodeTransformer` visitor set.

## Project Structure

```
thdse/
├── src/
│   ├── hdc_core/          # Rust FHRR Arena (AVX2 SIMD, PyO3 bindings)
│   ├── topology/           # AST graph builder, V1/V2 encoders
│   ├── axioms/             # Axiom vector DB, V1/V2 engines
│   ├── prophet/            # GNN structural anomaly detector
│   ├── scribe/             # LLM typist + offline AST patcher
│   └── swarm/              # V1/V2 orchestrators
├── scripts/                # Entry-point pipeline scripts
└── Cargo.toml
```

## Dependencies

- **Rust**: `pyo3`, `std::arch::x86_64` (AVX2)
- **Python**: `ast`, `networkx`, `numpy`, `torch`, `torch-geometric`, `httpx`, `asyncio`
- **Build**: `maturin`
