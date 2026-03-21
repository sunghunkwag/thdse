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
- **Self-diagnostic** (`run_self_diagnostic`): performs per-layer resonance analysis of the engine's own modules, measuring internal AST, CFG, and data-dependency coherence across the projector, decoder, and HDC core. Outputs a structured `self_diagnostic.json` report with per-module and per-layer-pair similarity scores
- **Synthesis quality assessment** (`run_synthesis_quality_test`): closed-loop fidelity measurement that executes the full synthesize → decode → re-project → cosine similarity pipeline, yielding a quantitative round-trip correlation score for each synthesis product

### 5. Decoder (`src/decoder/`)
**Constraint Decoder**: translates synthesized hypervectors back to code via Z3.
1. **Probe**: correlate against a vocabulary of structural atoms (node types, CFG patterns, data-dep signatures), with layer-aware unbinding for algebraically correct signal recovery
2. **Encode**: translate activated atoms into Z3 formulas (reachability, ordering, well-formedness), with optional thermodynamic entropy ceiling and UNSAT core tracking via `assert_and_track`
3. **Solve**: if SAT → compile model to Python AST; if UNSAT → trigger Phase Transition cascade:

#### Sub-Tree Vocabulary Synthesis (`src/decoder/subtree_vocab.py`)

The original decoder operates over an abstract atom vocabulary (e.g., "If", "Return", "FunctionDef") and reconstructs minimal placeholder ASTs. The sub-tree vocabulary module replaces this abstraction layer with concrete AST fragments harvested from the ingested corpus, enabling the decoder to assemble structurally faithful code rather than synthetic templates.

The construction proceeds as follows:

1. **Extraction**: for each ingested source file, all contiguous AST sub-trees of depth 2 through 4 are enumerated via bounded recursive descent. This depth range captures meaningful structural units — conditional blocks, loop bodies, assignment chains — while remaining tractable for corpus-scale enumeration.

2. **Canonicalization**: each extracted sub-tree undergoes a deterministic normalization pass:
   - Variable names are replaced with positional placeholders (`x0`, `x1`, …) assigned in order of first occurrence during a pre-order traversal
   - Literal values are replaced with type markers (`INT`, `STR`, `FLOAT`, `BOOL`, `NONE`)
   - The resulting canonical form is invariant under alpha-renaming and value substitution

3. **Deduplication**: a SHA-256 digest of the canonical AST's serialized form serves as a unique key. Duplicate sub-trees across different source files are collapsed into a single vocabulary entry, with a provenance set tracking all originating source locations.

4. **FHRR Projection**: each deduplicated canonical sub-tree is projected through the standard topology → FHRR pipeline, yielding a hypervector handle that participates in the same algebraic space as full-file projections. The decoder's probing and SMT encoding stages can therefore operate uniformly over both atom-level and sub-tree-level vocabulary entries.

The sub-tree vocabulary is constructed once per corpus ingestion and reused for all subsequent decoding operations.

#### Variable Threading Engine (`src/decoder/variable_threading.py`)

When the decoder assembles multiple canonical sub-trees into a single program, the positional placeholder variables (`x0`, `x1`, …) from distinct sub-trees must be unified into a consistent concrete namespace. The variable threading engine resolves this via a union-find (disjoint set) algorithm operating over def-use relationships:

1. For each ordered sub-tree in the assembly, definition and use sites are identified
2. Placeholder slots that participate in a shared data-flow edge (as determined by the data-dependency layer) are merged into the same equivalence class
3. Each equivalence class is assigned a deterministic concrete name from a fixed pool (`x`, `y`, `z`, `n`, …)
4. Unconnected placeholders receive fresh names

The overall complexity is O(n · α(n)), where α denotes the inverse Ackermann function — effectively linear for all practical input sizes.

#### Sub-Tree SMT Encoding

The extended decoder provides `encode_smt_subtrees()`, which formulates the sub-tree selection problem as an SMT instance: Z3 determines which corpus-derived sub-trees to include and in what order, subject to CFG ordering constraints, well-formedness invariants, and an optional thermodynamic entropy ceiling. The corresponding `compile_model_subtrees()` assembles the selected sub-trees into a valid Python AST with variable threading applied, and verifies the result via `compile()`.

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

### 6. Structural Diff Engine (`src/analysis/structural_diff.py`)

The standard resonance metric collapses the three topology layers (AST, CFG, data-dependency) into a single scalar similarity ρ ∈ ℝ. While sufficient for clustering and clique extraction, this conflation discards diagnostically valuable information about which structural dimension drives the observed similarity or divergence.

The structural diff engine decomposes pairwise comparison into per-layer cosine similarities:

- **sim_ast**: structural congruence of the abstract syntax tree
- **sim_cfg**: control-flow topology similarity (branching, sequencing, loop structure)
- **sim_data**: data-dependency isomorphism (def-use chain alignment)

Each comparison yields a `LayerSimilarity` record accompanied by a diagnostic explanation synthesized from the layer-wise decomposition (e.g., "identical control flow with divergent data wiring" or "structurally isomorphic with minor CFG variation").

#### Refactoring Detector (`src/analysis/refactoring_detector.py`)

Leveraging the layer-decomposed similarity, the refactoring detector identifies two classes of actionable findings:

1. **Structural duplicates**: file pairs satisfying sim_ast > 0.7 ∧ sim_cfg > 0.7 with distinct directory provenance. These represent duplicated implementations amenable to extraction into shared modules.

2. **Interface unification candidates**: file pairs satisfying sim_data > 0.8 ∧ sim_ast < 0.3. These encode the same computation through different syntactic expressions — candidates for interface normalization or adapter pattern introduction.

#### Temporal Structural Analysis (`src/analysis/temporal_diff.py`)

Given two versions of the same source file, the temporal analysis module projects both through the FHRR pipeline and classifies the change by its topological signature:

| Classification | Criterion |
|----------------|-----------|
| **Structural refactor** | AST changed, data flow preserved |
| **Logic change** | CFG changed, AST preserved |
| **Variable rewiring** | Data-dep changed, others preserved |
| **Complete rewrite** | All layers changed |
| **Cosmetic change** | All layers preserved (whitespace/comments only) |

This provides semantic classification of code evolution that textual diff cannot capture: git observes character-level deltas, whereas THDSE observes topological transformations.

### 7. Corpus Ingestion (`src/corpus/`)
**Batch Ingester**: scalable pipeline for processing directories of Python projects into VSA space.
- Walks project trees with configurable filtering (skip tests, `__init__.py`, generated code, files exceeding 64 KB)
- Graceful error handling: syntax errors, projection failures, and arena capacity exhaustion are logged and skipped without crashing
- Serialization support via pickle for persistent axiom stores, enabling corpus reuse without re-processing
- Empirically validated against 500+ Python standard library modules in under 65 seconds on commodity hardware

**Stdlib Corpus**: auto-discovers the current Python installation's standard library as a guaranteed-available corpus requiring no external downloads.

### 8. Resonance Analysis (`src/analysis/`)
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
- Structural diff reports (per-layer decomposition)
- Refactoring recommendations (structural duplicates, interface unification candidates)
- Temporal change classification for version-to-version comparisons
- Linkage matrix for external dendrogram visualization

### 9. CLI Entry Point (`src/discover.py`)
Multi-mode orchestration of the full pipeline, selectable via `--mode`:

```bash
# Standard resonance analysis (default mode)
python src/discover.py --stdlib --output ./report/

# Structural diff between two files
python src/discover.py --mode diff FILE_A FILE_B

# Refactoring detection across a corpus
python src/discover.py --mode refactor --corpus ./projects/ --output ./report/

# Synthesis: ingest corpus, extract cliques, synthesize novel structures
python src/discover.py --mode synthesis --corpus ./projects/ --output ./report/

# Ouroboros self-diagnostic: the engine analyzes its own source
python src/discover.py --mode self-diagnostic --output ./report/

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
│   │   ├── constraint_decoder.py
│   │   ├── subtree_vocab.py   # Corpus-derived sub-tree vocabulary
│   │   └── variable_threading.py  # Union-find variable unification
│   ├── corpus/                # Batch ingestion engine
│   │   ├── ingester.py
│   │   └── stdlib_corpus.py
│   ├── analysis/              # Resonance analysis + reporting
│   │   ├── resonance_analyzer.py
│   │   ├── report.py
│   │   ├── structural_diff.py     # Layer-decomposed similarity
│   │   ├── refactoring_detector.py # Duplicate & unification detection
│   │   └── temporal_diff.py       # Version-to-version change classification
│   ├── utils/                 # Phase algebra utilities
│   │   └── arena_ops.py
│   └── discover.py            # CLI entry point (multi-mode)
├── tests/
│   ├── test_decoder_subtree.py    # Sub-tree vocabulary & threading tests
│   └── test_structural_diff.py    # Structural diff engine tests
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

# Run full test suite (14 tests including sub-tree decoding and structural diff)
python run_tests.py

# Run the empirical discovery pipeline against stdlib
python src/discover.py --stdlib --output ./report/

# Run Ouroboros self-diagnostic
python src/discover.py --mode self-diagnostic --output ./report/
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
- **Sub-tree compositionality**: the sub-tree vocabulary preserves the algebraic guarantees of the atom vocabulary while operating over concrete AST fragments. Canonicalization is deterministic (pre-order traversal with positional placeholder assignment), deduplication is collision-resistant (SHA-256), and variable threading is near-linear (union-find with path compression and union by rank). The assembled output is verified via `compile()` to ensure syntactic validity.

## Changelog

### v0.3.0 — Architectural Upgrade (4-Leap Expansion)

This release introduces four interconnected subsystems that collectively advance the engine from prototype-stage resonance analysis toward a structurally grounded code synthesis and evolution analysis platform.

**Leap 1 — Sub-Tree Vocabulary Synthesis.** The decoder's abstract atom vocabulary (individual AST node types) is supplanted by a corpus-derived sub-tree vocabulary. Concrete AST fragments of depth 2–4 are extracted, canonicalized via alpha-renaming and type-marker substitution, deduplicated by SHA-256, and projected into the FHRR space. A union-find–based variable threading engine ensures consistent naming when composing selected sub-trees into a single program. The SMT encoding is extended accordingly: Z3 selects which sub-trees to include and in what order, and the compiler assembles real AST fragments verified via `compile()`.

**Leap 2 — Structural Diff Engine.** A layer-decomposed similarity analysis module decomposes the scalar resonance metric into per-layer components (sim_ast, sim_cfg, sim_data), enabling diagnostic explanations of structural similarity and divergence. Two downstream detectors leverage this decomposition: a refactoring detector that identifies structural duplicates and interface unification candidates, and a temporal analysis module that classifies version-to-version changes by their topological signature (structural refactor, logic change, variable rewiring, complete rewrite, or cosmetic change).

**Leap 3 — Ouroboros Self-Diagnostic.** The autopoietic self-reference capability is extended with a quantitative self-diagnostic: the engine ingests its own source modules, computes per-layer resonance among them, and outputs a structured report. A complementary synthesis quality test performs the full closed-loop pipeline (synthesize → decode → re-project → cosine similarity) to measure round-trip fidelity.

**Leap 4 — CLI and Pipeline Integration.** The CLI entry point is restructured around a `--mode` selector supporting five operational modes: `analysis` (default resonance pipeline), `diff` (pairwise structural comparison), `refactor` (corpus-wide duplicate and unification detection), `synthesis` (clique extraction and code generation), and `self-diagnostic` (Ouroboros self-analysis). The report generator is extended with sections for structural diffs, refactoring recommendations, and temporal change classifications.
