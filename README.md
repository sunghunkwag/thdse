# THDSE — Multi-Layer Topological Phase-Transition Engine

A strictly deterministic architecture that mathematically synthesizes new code structures by combining Vector Symbolic Architecture (VSA) operations with SMT-based constraint solving. Zero randomness. Zero LLMs. Every step is algebraically provable.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Source Code    │      │  Axiom Corpus    │      │  Novel Code     │
│   (Python)       │      │  {c_1, ..., c_n} │      │  (Python AST)   │
└────────┬────────┘      └────────┬─────────┘      └────────▲────────┘
         │                        │                          │
         ▼                        ▼                          │
┌─────────────────┐      ┌──────────────────┐      ┌────────┴────────┐
│   TOPOLOGY       │      │   PROJECTION      │      │   DECODER        │
│   AST + CFG +    │─────▶│   Graph → FHRR    │─┐   │   VSA → SMT → AST│
│   Data-Dep Graph │      │   Hypervector     │ │   │   (Z3 Solver)    │
└─────────────────┘      └──────────────────┘ │   └────────▲────────┘
                                               │            │
                                               ▼            │
                                      ┌──────────────────┐ │
                                      │   SYNTHESIS       │─┘
                                      │   Phase Transition│
                                      │   bind(A_i ⊗ A_j) │
                                      └──────────────────┘
                                               │
                                      ┌────────▼─────────┐
                                      │   HDC_CORE (Rust) │
                                      │   FHRR Arena      │
                                      │   AVX2 SIMD       │
                                      └──────────────────┘

                         ┌──────────────────────────────────┐
                         │   PHASE TRANSITION MECHANISMS     │
                         │                                   │
                         │ 1. Quotient Space Folding         │
                         │    H → H / ⟨V_error⟩             │
                         │ 2. Operator Fusion                │
                         │    A ⊗ (B₁ ⊕ … ⊕ Bₖ)           │
                         │ 3. Dimension Expansion            │
                         │    d → 2d (conjugate reflection)  │
                         └──────────────────────────────────┘

                         ┌──────────────────┐
                         │   DISCOVERY       │
                         │   Corpus Ingest → │
                         │   Resonance →     │
                         │   Clustering →    │
                         │   Report          │
                         └──────────────────┘
```

## Modules

### 1. Topology (`src/topology/`)
Pure preprocessing: decomposes Python source into a 3-layer directed graph.
- **AST layer**: structural parent-child edges
- **CFG layer**: sequential flow, branch targets, loop back-edges
- **Data-Dep layer**: lightweight def-use chains within scope

### 2. HDC Core (`src/hdc_core/`)
Rust library (PyO3 bindings) implementing an FHRR memory arena:
- `allocate()` → handle
- `inject_phases(handle, phases)` → initialize slot from phase vector
- `bind(h1, h2, out)` → complex element-wise multiplication (⊗), AVX2 SIMD
- `bundle(handles, out)` → superposition + normalize (⊕), AVX2 SIMD
- `compute_correlation(h1, h2)` → cosine similarity in complex domain
- `expand_dimension(new_dim)` → meta-grammar emergence via conjugate reflection
- `bind_bundle_fusion(h, handles, out)` → fused algebraic operator A ⊗ (B₁ ⊕ … ⊕ Bₖ)
- `compute_entropy(h)` → thermodynamic cost S(h) = n_bind·ln2 + n_bundle·ln2
- `project_to_quotient_space(v_error_id)` → orthogonal projection H → H/⟨V_error⟩, AVX2 SIMD
- `extract_phases(handle)` → read back phase angles from arena slot

### 3. Projection (`src/projection/`)
**Isomorphic Projector**: maps the 3-layer graph into a single holographic hypervector.
- CFG sequences → chain-bind (⊗) preserving order
- Data-dep sets → bundle (⊕) preserving membership
- AST identity → bind(type ⊗ position ⊗ CA-state)
- Final: bind all three layer vectors into one hologram
- **Contradiction Vector Synthesis**: given a set of conflicting atom handles from the UNSAT core, algebraically combines them via bind/bundle to produce a single V_error encoding the exact axis of topological contradiction

### 4. Synthesis (`src/synthesis/`)
**Axiomatic Synthesizer**: deterministic phase-transition engine.
1. Ingest stable codebases → project each to an axiom vector
2. Compute pairwise resonance matrix via `correlate`
3. Extract maximal cliques where all pairs exceed threshold τ (Bron-Kerbosch)
4. Chain-bind each clique → novel synthesized hypervector

The synthesized vector is quasi-orthogonal to each source axiom (concentration of measure) yet encodes their shared structural invariants.

Additional capabilities:
- **Autopoietic self-reference** (Ouroboros loop): the engine ingests, projects, and synthesizes its own source code
- **Topological thermodynamics**: entropy-constrained synthesis with thermodynamic ranking by ascending entropy

### 5. Decoder (`src/decoder/`)
**Constraint Decoder**: translates synthesized hypervectors back to code via Z3.
1. **Probe**: correlate against a vocabulary of structural atoms (node types, CFG patterns, data-dep signatures), with layer-aware unbinding for algebraically correct signal recovery
2. **Encode**: translate activated atoms into Z3 formulas (reachability, ordering, well-formedness), with optional thermodynamic entropy ceiling and UNSAT core tracking via `assert_and_track`
3. **Solve**: if SAT → compile model to Python AST; if UNSAT → trigger Phase Transition cascade:

#### Phase Transition Cascade (UNSAT Resolution)

When the Z3 solver determines that the probed topological constraints form an unsatisfiable system, the decoder initiates a strictly ordered cascade of deterministic resolution mechanisms:

**Phase 1 — Dynamic Null Space Projection (Quotient Space Folding)**

The algebraically cheapest resolution: no dimension change, no re-projection required.

1. **UNSAT Core Extraction**: Z3's `unsat_core()` isolates the minimal subset of conflicting constraints. Each constraint is tracked via `assert_and_track` with labels corresponding to structural atom vocabulary keys (e.g., `atom:cfg_seq:If→Return`).
2. **Reverse Lookup**: the extracted core labels are mapped back to their corresponding hypervector handles in the hdc_core arena, recovering the exact set of conflicting structural atoms.
3. **Contradiction Vector Synthesis**: the conflicting atom handles are algebraically combined into a single Contradiction Vector V_error:
   - For |core| = 1: V_error = atom₁
   - For |core| = 2: V_error = atom₁ ⊗ atom₂
   - For |core| ≥ 3: V_error = atom₁ ⊗ (atom₂ ⊕ … ⊕ atomₖ) via the fusion operator
4. **Orthogonal Projection**: the entire arena is projected to the quotient space H/⟨V_error⟩ by annihilating the component parallel to V_error from every live hypervector:

$$
X' = X - V_{\text{error}} \cdot \frac{V_{\text{error}}^H \cdot X}{V_{\text{error}}^H \cdot V_{\text{error}}}
$$

   This is implemented in Rust with AVX2 SIMD acceleration (two-pass: scalar inner product, SIMD subtraction + re-normalization). The projection is performed in-place with O(head × dimension) complexity and zero auxiliary allocation. Each component is re-normalized to unit magnitude to preserve the FHRR invariant |z_j| = 1.

5. **Re-probe and Re-solve**: the folded projection is re-probed against the (now algebraically modified) atom vocabulary, and the resulting constraints are submitted to a fresh Z3 solver instance.

**Phase 2 — Operator Fusion**: synthesize fuse(ast, [cfg, data]) = ast ⊗ (cfg ⊕ data), collapsing cross-layer interference via a novel algebraic grammar rule.

**Phase 3 — Dimension Expansion**: expand the representational capacity d → 2d via deterministic conjugate reflection, then rebuild the vocabulary and re-probe at the higher dimension.

### 6. Corpus Ingestion (`src/corpus/`)
**Batch Ingester**: scalable pipeline for processing directories of Python projects into VSA space.
- Walks project trees with configurable filtering (skip tests, `__init__.py`, generated code, files exceeding 64 KB)
- Graceful error handling: syntax errors, projection failures, and arena capacity exhaustion are logged and skipped without crashing
- Serialization support via pickle for persistent axiom stores, enabling corpus reuse without re-processing
- Empirically validated against 500+ Python standard library modules in under 65 seconds on commodity hardware

**Stdlib Corpus**: auto-discovers the current Python installation's standard library as a guaranteed-available corpus requiring no external downloads.

### 7. Structural Analysis (`src/analysis/`)
**Resonance Analyzer**: computes the full pairwise resonance matrix and extracts structural patterns invisible to manual code review.

- **Hierarchical clustering**: agglomerative Ward-linkage clustering on the resonance-derived distance matrix (1 − ρ), yielding dendrograms that reveal which files are structurally isomorphic despite differing in purpose
- **Outlier detection**: identifies axioms whose mean resonance falls below 2σ of the corpus-wide distribution — structurally unique files that may represent novel patterns or architectural defects
- **Clique interpretation**: for each maximal clique extracted by Bron-Kerbosch, bundles all member vectors into a centroid, probes it against the decoder's structural atom vocabulary, and emits a human-readable fingerprint (e.g., "heavy branching + tight def-use chains + single loop")
- **Cross-project bridging**: discovers pairs (file_A ∈ project_X, file_B ∈ project_Y) with resonance ρ > 0.8 across entirely different domains — structural twins that constitute the most empirically interesting findings

**Report Generator**: produces self-contained JSON and Markdown reports including:
- Corpus summary (file count, dimension d, resonance μ/σ/min/max)
- Top-k most resonant pairs with structural fingerprints
- Top-k lowest-resonance outliers
- All cliques of size ≥ 3 with decoded structural fingerprints
- Cross-project structural twins
- Linkage matrix for external dendrogram visualization

### 8. CLI Entry Point (`src/discover.py`)
Single-command orchestration of the full empirical discovery pipeline:

```bash
# Analyze the Python standard library
python src/discover.py --stdlib --output ./report/

# Analyze one or more project directories
python src/discover.py --corpus ./project_a/ ./project_b/ --output ./report/

# Save ingested corpus for later reuse
python src/discover.py --corpus ./projects/ --save corpus.pkl --output ./report/

# Reload a saved corpus without re-ingesting
python src/discover.py --load corpus.pkl --output ./report/
```

Configurable parameters: `--dimension` (default: 256), `--capacity` (default: 2M handles), `--threshold` (resonance τ, default: 0.15), `--max-files`.

## Project Structure

```
thdse/
├── src/
│   ├── hdc_core/              # Rust FHRR Arena (AVX2 SIMD, PyO3)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── topology/              # Multi-layer graph builder + CA
│   │   ├── ast_graph_ca.py
│   │   └── multi_layer_builder.py
│   ├── projection/            # Graph → VSA isomorphic mapping
│   │   └── isomorphic_projector.py
│   ├── synthesis/             # Axiomatic phase-transition engine
│   │   └── axiomatic_synthesizer.py
│   ├── decoder/               # SMT-based constraint decoder
│   │   └── constraint_decoder.py
│   ├── corpus/                # Batch ingestion engine
│   │   ├── ingester.py
│   │   └── stdlib_corpus.py
│   ├── analysis/              # Resonance analysis + reporting
│   │   ├── resonance_analyzer.py
│   │   └── report.py
│   ├── utils/                 # Phase algebra utilities
│   │   └── arena_ops.py
│   └── discover.py            # CLI entry point
├── run_tests.py
├── requirements.txt
└── Cargo.toml
```

## Dependencies

- **Rust**: `pyo3 0.18`, `std::arch::x86_64` (AVX2/FMA)
- **Python**: `networkx >= 3.1`, `numpy >= 1.24.0`, `z3-solver >= 4.12.0`, `scipy >= 1.10.0`
- **Build**: `maturin`

## Usage

```bash
# Build the Rust VSA engine
cd src/hdc_core && maturin develop --release && cd ../..

# Install Python dependencies
pip install -r requirements.txt

# Run full test suite (12 tests including quotient space folding and stdlib discovery)
python run_tests.py

# Run the empirical discovery pipeline against stdlib
python src/discover.py --stdlib --output ./report/
```

## Empirical Validation

The discovery pipeline has been validated against the Python 3.11 standard library:

| Metric | Value |
|--------|-------|
| Modules ingested | 519 / 544 eligible |
| Ingestion time | ~61 s |
| Mean resonance | 0.0187 |
| Std resonance | 0.1228 |
| Cliques (size ≥ 3) | 109 |

Notable findings:
- **Encoding family isomorphism**: all `encodings/cpXXX.py` modules form a single large clique (resonance ≈ 1.0), correctly identifying them as structurally identical codec implementations
- **Structural outliers**: `lib2to3/pgen2/token.py` exhibits the lowest mean resonance (−0.022), consistent with its role as a flat constant-definition file with no control flow or data dependencies
- **Importlib twins**: `importlib/readers.py` and `importlib/simple.py` achieve the highest pairwise resonance (1.000), reflecting their parallel resource-loading architectures

## Mathematical Guarantees

- **Determinism**: every operation is a pure function of its inputs. Z3's DPLL(T) is deterministic for a fixed formula.
- **No hallucination**: only structures satisfiable under the SMT constraints are emitted.
- **Invertibility** (up to VSA capacity): decoded ASTs, when re-projected, correlate with the input synthesis vector.
- **Compositionality**: bind (⊗) preserves ordering, bundle (⊕) preserves membership — both are algebraically exact in FHRR.
- **Thermodynamic ordering**: synthesized structures are ranked by entropy S(h) = n_bind·ln 2 + n_bundle·ln 2, enabling principled selection of minimum-complexity solutions.
- **Meta-grammar emergence**: on UNSAT, the decoder deterministically synthesizes new algebraic operators (fusion) or expands the representational dimension (d → 2d via conjugate reflection) to resolve contradictions.
- **Quotient space correctness**: the orthogonal projection X' = X − V·(V^H·X)/(V^H·V) annihilates exactly the one-dimensional subspace spanned by V_error. The resulting quotient space H/⟨V_error⟩ preserves all pairwise correlations that are orthogonal to the contradiction axis, ensuring that structurally valid relationships remain undistorted after folding. Post-projection re-normalization to unit magnitude per component maintains the FHRR representational invariant.
