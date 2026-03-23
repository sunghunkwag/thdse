# THDSE — Topological Holographic Design Space Engine

A strictly deterministic code synthesis engine combining Vector Symbolic Architecture (VSA) with SMT constraint solving. Zero randomness. Zero LLMs. Every operation is algebraically provable.

## Quick Start

```bash
# Build Rust VSA engine
cd src/hdc_core && maturin develop --release && cd ../..

# Install Python dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run discovery pipeline
python src/discover.py --stdlib --output ./report/
```

## Architecture

```
  Python Source Code
         │
         ▼
  ┌─────────────────┐    ┌────────────────────┐    ┌──────────────┐
  │ 1. TOPOLOGY      │───▶│ 2. FHRR PROJECTION  │───▶│ 3. SYNTHESIS  │
  │ Parse → 3-layer  │    │ Graph → Hypervector │    │ Clique → Bind │
  │ graph (AST/CFG/  │    │ via Rust Arena      │    │ → Novel Vector│
  │ Data-Dep)        │    │ (AVX2 w/ fallback)  │    │               │
  └─────────────────┘    └────────────────────┘    └──────┬───────┘
                                                          │
  ┌─────────────────┐    ┌────────────────────┐    ┌──────▼───────┐
  │ 6. ANALYSIS      │◀───│ 5. SERL LOOP       │◀───│ 4. DECODING   │
  │ Resonance matrix,│    │ Execute → Evaluate  │    │ Z3 SMT solve  │
  │ clustering, diff,│    │ → Expand vocab      │    │ → Python AST  │
  │ refactoring      │    │ → Next cycle        │    │ UNSAT → Phase │
  └─────────────────┘    └────────────────────┘    │  Transition   │
                                                    └──────────────┘
```

**UNSAT Resolution Cascade**: Quotient Space Folding (project out V_error) → Operator Fusion (ast ⊗ (cfg ⊕ data)) → Dimension Expansion (d → 2d via conjugate reflection).

**SERL Loop**: Synthesize → Decode → Execute in sandbox → If fitness ≥ 0.4, expand sub-tree vocabulary → Repeat. Halts when F_eff = 0 for K consecutive cycles (space closed).

**Swarm**: N agents with heterogeneous corpora communicate via raw FHRR phase arrays. Orchestrator computes algebraic consensus via correlate_matrix + Bron-Kerbosch. UNSAT walls target only participating agents (Localized Quotient Isolation) with drift reconciliation for long-term coordinate alignment. Agent synthesis runs in parallel via `ProcessPoolExecutor` with per-process agent instantiation.

## Modules

| Module | Path | Purpose |
|--------|------|---------|
| **HDC Core** | `src/hdc_core/` | Rust/PyO3 FHRR arena — bind(⊗), bundle(⊕), correlate, quotient projection. Runtime AVX2/FMA detection with safe scalar fallback for non-x86 hardware. |
| **Topology** | `src/topology/` | Multi-layer graph builder (AST + CFG + Data-Dep edges). Cellular automaton state evolution with tolerance-based zero checks (`np.isclose`). |
| **Projection** | `src/projection/` | Isomorphic graph→hypervector mapping. Per-layer decomposition preserved via `LayeredProjection`. |
| **Synthesis** | `src/synthesis/` | Axiomatic synthesizer (resonance matrix → Bron-Kerbosch cliques → chain-bind). Wall archive for contradiction vectors. SERL loop orchestrator. |
| **Decoder** | `src/decoder/` | Z3 SMT-based constraint decoder. Sub-tree vocabulary (depth 2–4 AST fragments, SHA-256 dedup). Variable threading (union-find). Fitness-gated vocab expansion. |
| **Execution** | `src/execution/` | Subprocess-isolated sandbox. 10 deterministic test vectors. Monotonic fitness scoring (0.0–1.0). |
| **Corpus** | `src/corpus/` | Batch ingestion engine + stdlib auto-discovery. |
| **Analysis** | `src/analysis/` | Resonance analysis, structural diff (per-layer), refactoring detection, temporal change classification, boundary cartography. |
| **Swarm** | `src/swarm/` | Multi-agent system — protocol (PhaseMessage + uint16 quantization), consensus (resonance cliques), agent (independent THDSE pipeline), orchestrator (parallel synthesis + LQID + drift reconciliation). |
| **Utils** | `src/utils/` | Pure-Python FHRR phase algebra (bind, bundle, conjugate, entropy). |
| **CLI** | `src/discover.py` | Entry point — modes: `analysis`, `diff`, `refactor`, `synthesis`, `self-diagnostic`, `cartography`, `serl`, `swarm`, `empirical`. |

## HDC Core API (Rust)

| Method | Description |
|--------|-------------|
| `allocate()` | Allocate hypervector handle |
| `inject_phases(h, phases)` | Initialize slot from phase vector |
| `bind(h1, h2, out)` | Complex multiply (⊗) — runtime SIMD dispatch |
| `bundle(handles, out)` | Superposition + normalize (⊕) — runtime SIMD dispatch |
| `compute_correlation(h1, h2)` | Cosine similarity |
| `correlate_matrix(handles)` | Batch upper-triangle pairwise correlation (single FFI call) |
| `correlate_matrix_subset(q, t)` | Asymmetric Q×T correlation matrix |
| `expand_dimension(new_dim)` | Meta-grammar emergence (conjugate reflection) |
| `bind_bundle_fusion(h, handles, out)` | Fused A ⊗ (B₁ ⊕ … ⊕ Bₖ) |
| `project_to_quotient_space(v)` | Orthogonal projection H → H/⟨V_error⟩ |
| `project_to_multi_quotient_space(vs)` | Gram-Schmidt + multi-vector projection |
| `compute_entropy(h)` | Thermodynamic cost S(h) = n_bind·ln2 + n_bundle·ln2 |
| `extract_phases(h)` | Read back phase angles |

All SIMD operations (bind, bundle, normalize, quotient projection) use **runtime CPU feature detection** (`is_x86_feature_detected!`) with safe scalar fallbacks. No hardcoded `#[target_feature]` — runs correctly on any x86_64 or non-x86 hardware.

## Project Structure

```
thdse/
├── src/
│   ├── hdc_core/                 # Rust FHRR Arena (PyO3, AVX2 w/ scalar fallback)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── topology/                 # Multi-layer graph builder + cellular automaton
│   ├── projection/               # Graph → VSA isomorphic mapping
│   ├── synthesis/                # Axiomatic synthesizer, SERL loop, wall archive
│   ├── decoder/                  # Z3 SMT decoder, sub-tree vocab, variable threading
│   ├── execution/                # Subprocess-isolated sandbox
│   ├── corpus/                   # Batch ingestion + stdlib discovery
│   ├── analysis/                 # Resonance, structural diff, refactoring, cartography
│   ├── swarm/                    # Multi-agent orchestration via phase arrays
│   ├── utils/                    # Phase algebra utilities
│   └── discover.py               # CLI entry point
├── tests/                        # 94 pytest tests (89 pass, 3 skip, 2 known issues)
├── run_tests.py                  # Integration test orchestrator
├── requirements.txt
└── Cargo.toml                    # Workspace root
```

## CLI Usage

```bash
python src/discover.py --stdlib --output ./report/                      # Resonance analysis
python src/discover.py --mode diff FILE_A FILE_B                        # Structural diff
python src/discover.py --mode refactor --corpus ./projects/             # Refactoring detection
python src/discover.py --mode synthesis --corpus ./projects/            # Synthesis
python src/discover.py --mode serl --corpus ./projects/                 # SERL closed-loop
python src/discover.py --mode swarm --stdlib --n-agents 3               # Multi-agent swarm
python src/discover.py --mode cartography --corpus ./projects/          # Boundary mapping
python src/discover.py --mode self-diagnostic                           # Ouroboros self-analysis
python src/discover.py --mode empirical --output ./report/              # Reproducible validation
```

**Parameters**: `--dimension` (default: 256), `--capacity` (default: 2M), `--threshold` (default: 0.15), `--n-agents` (default: 3), `--consensus-threshold` (default: 0.85).

## Dependencies

- **Rust**: `pyo3 0.18` (PyO3 bindings)
- **Python**: `networkx >= 3.1`, `numpy >= 1.24.0`, `z3-solver >= 4.12.0`, `scipy >= 1.10.0`
- **Build**: `maturin` (for Rust extension)

## Testing

```bash
python -m pytest tests/ -v          # Full suite (94 tests)
python run_tests.py                 # Integration tests
```

| Test Module | Coverage |
|-------------|----------|
| `test_swarm.py` (22) | Agent lifecycle, consensus, LQID, drift reconciliation, weighted bundling |
| `test_boundary_cartography.py` (13) | Wall archive, multi-quotient projection, residual dimension, serialization |
| `test_decoder_subtree.py` (11) | Sub-tree vocab, variable threading, round-trip fidelity |
| `test_batch_correlation.py` (9) | Batch correlation correctness, equivalence, edge cases |
| `test_execution_sandbox.py` (9) | Full fitness gradient (compile fail → nontrivial computation) |
| `test_vocab_expander.py` (6) | Fitness gating, dedup, projection, activation |
| `test_serl.py` (5) | Cycle history, vocabulary expansion, stagnation detection |
| `test_nesting_decoder.py` (4) | Hierarchical nesting constraints |
| `test_structural_diff.py` (7) | Per-layer similarity, structural delta, serialization |
| `test_empirical_report.py` (2) | Stdlib analysis pipeline |

## Key Design Decisions

- **Determinism**: All hashing uses `zlib.crc32` (session-independent). Z3 DPLL(T) is deterministic for fixed formulas. No `random`, no `PYTHONHASHSEED` dependency.
- **Hardware portability**: Rust SIMD kernels use runtime `is_x86_feature_detected!()` with `#[cfg(target_arch)]` gates. Safe scalar fallbacks run on any platform.
- **Floating-point safety**: Zero comparisons use `np.isclose(x, 0, atol=1e-8)` to prevent NaN propagation from precision drift.
- **Swarm parallelism**: `ProcessPoolExecutor` with a top-level worker function that instantiates fresh agents per process. No pickling of Rust FFI or Z3 solver objects — only primitive arguments cross the IPC boundary.
- **Memory safety**: SIMD loop bounds use `end.sub(floats % 8)` for correct tail handling. All scalar fallbacks use bounds-checked indexing.
- **Subprocess isolation**: Synthesized code executes in separate OS processes (no shared memory, no capability inheritance).

## Empirical Validation (Python 3.11 stdlib)

| Metric | Value |
|--------|-------|
| Modules ingested | 519 / 544 |
| Mean resonance | 0.0187 (σ = 0.1228) |
| Cliques (size ≥ 3) | 109 |
| Notable | `encodings/cpXXX.py` → single clique (ρ ≈ 1.0); `importlib/readers.py` ↔ `simple.py` (ρ = 1.000) |
