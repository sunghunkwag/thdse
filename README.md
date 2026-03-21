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

                         ┌──────────────────────────────────┐
                         │   BOUNDARY CARTOGRAPHY            │
                         │                                   │
                         │ 1. Collect V_error walls           │
                         │ 2. Map wall resonance → cliques   │
                         │ 3. Estimate residual dimension    │
                         │ 4. Extract open directions        │
                         │ 5. Synthesize along open dirs     │
                         │ 6. Iterate until space closes     │
                         └──────────────────────────────────┘

                         ┌──────────────────┐
                         │   DISCOVERY       │
                         │   Corpus Ingest → │
                         │   Resonance →     │
                         │   Clustering →    │
                         │   Report          │
                         └──────────────────┘

                         ┌──────────────────────────────────┐
                         │   SERL (Closed-Loop Expansion)    │
                         │                                   │
                         │ corpus → synthesize → decode →    │
                         │ EXECUTE → EVALUATE →              │
                         │   [fitness > θ] → REINGEST →      │
                         │   vocab_expand → next cycle        │
                         │                                   │
                         │ F_eff = new_atoms / syntheses      │
                         │ > 0 → space is open                │
                         │ = 0 for K cycles → space closed    │
                         └──────────────────────────────────┘

                         ┌──────────────────────────────────┐
                         │   SWARM (Collective Intelligence)  │
                         │                                   │
                         │ N agents × independent pipelines  │
                         │ Communication: raw phase arrays    │
                         │ Consensus: correlate_matrix +      │
                         │   Bron-Kerbosch clique extraction  │
                         │ UNSAT → Global Quotient Collapse   │
                         │   (broadcast V_error to all agents)│
                         │                                   │
                         │ Heterogeneous corpora → diverse    │
                         │   resonance landscapes →           │
                         │   collective > any individual      │
                         └──────────────────────────────────┘
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
- `correlate_matrix(handles)` → full upper-triangle pairwise correlation matrix in a single FFI call; returns flat `Vec<f32>` of length N*(N-1)/2 in row-major order
- `correlate_matrix_subset(queries, targets)` → asymmetric correlation matrix between query and target handle sets; returns flat `Vec<f32>` of length |Q|×|T| in row-major order
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
- **Ouroboros SERL** (`run_serl_self`): runs the full SERL closed-loop on the engine's own 9 source files, measuring whether the engine can expand its own representational vocabulary through self-synthesis

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

### 6. Boundary Cartography (`src/analysis/boundary_cartography.py`)

The synthesis pipeline produces novel code structures by composing axiom vectors via algebraic operations. However, not all regions of the hypervector space yield satisfiable constraint systems — the Z3 solver may return UNSAT, indicating a topological contradiction. Each such event produces a Contradiction Vector V_error that encodes the precise axis of failure. The Boundary Cartography subsystem systematically collects, analyses, and exploits these contradiction signals to map the geometry of the design space boundary.

#### Wall Archive (`src/synthesis/wall_archive.py`)

The Wall Archive provides persistent storage for Contradiction Vectors accumulated across synthesis attempts. Each `WallRecord` captures:

- The arena handle and phase array of V_error (the latter enabling re-injection after arena reset)
- The UNSAT core labels and corresponding atom handles from Z3's minimal conflict extraction
- Provenance metadata (synthesis context, temporal index, status)

Walls are never deleted; they transition between three statuses: **active** (confirmed UNSAT boundary), **resolved** (previously UNSAT, now SAT after a design space modification such as dimension expansion or vocabulary update), and **orphaned** (constituent atoms no longer present in the current vocabulary). Revalidation is triggered deterministically by design space changes and proceeds by re-encoding the original UNSAT core constraints and re-solving via Z3.

The archive supports Bron-Kerbosch clique extraction over the wall resonance matrix, identifying **boundary regions** — clusters of structurally similar contradictions that delineate coherent forbidden zones in the design space.

Serialization and deserialization via pickle enable persistence of the boundary map across sessions, with automatic re-injection of phase arrays into a fresh arena upon loading.

#### Residual Dimension Estimation

After accumulating k wall vectors, the cartography module estimates the effective dimension of the residual subspace H/⟨V₁, …, Vₖ⟩ — the portion of the original d-dimensional space that remains accessible for synthesis. The procedure is as follows:

1. **Deterministic probe generation**: N = min(2d, 50) probe vectors are generated via a hash-seeded linear congruential generator, ensuring full reproducibility
2. **Multi-quotient projection**: all probes are projected through the quotient space spanned by the wall vectors, removing the forbidden subspace
3. **Correlation matrix construction**: the N × N pairwise correlation matrix of the projected probes is computed
4. **Spectral analysis**: approximate singular values are derived via Gershgorin disc estimates on the correlation matrix; the effective dimension d_eff equals the count of singular values exceeding a noise threshold (ε = 0.01)

The compression ratio d_eff / d quantifies how much of the design space has been consumed by contradictions. When d_eff < 0.1 · d, the space is classified as **closed** — further synthesis is unlikely to produce novel satisfiable structures.

#### Open Direction Extraction

The cartography module identifies the top-k directions of maximum variance in the residual space — the unexplored regions where new synthesis has the highest probability of producing novel, non-contradictory results. For each direction, the procedure selects the corpus axioms most aligned with that direction (by correlation) and synthesizes from this direction-biased subset rather than from standard resonance cliques.

#### Multi-Quotient Projection (`src/hdc_core/src/lib.rs`)

The HDC core is extended with `project_to_multi_quotient_space`, which removes an entire subspace spanned by multiple wall vectors simultaneously. Unlike sequential single-vector projection (which introduces order-dependent numerical drift), this operation first orthogonalizes the wall vectors via Gram-Schmidt with complex inner products, then projects each orthogonalized basis vector out sequentially. The resulting projection is mathematically equivalent to simultaneous subspace removal, with complexity O(k² · d + k · head · d) where k denotes the number of walls.

#### Cartography Loop

The `run_boundary_cartography` function executes the complete iterative loop:

1. **Map**: compute wall resonance matrix and extract boundary region cliques
2. **Measure**: estimate the residual subspace dimension
3. **Navigate**: extract principal open directions from the residual space
4. **Synthesize**: attempt directed synthesis along each open direction
5. **Record**: if UNSAT, record the new wall and re-map; if SAT, record the successful synthesis
6. **Terminate**: halt when no open directions remain, the space is classified as closed, or the maximum iteration count is reached

The output is a structured `boundary_map.json` containing the wall archive summary, boundary region cliques, residual dimension trajectory, and all successful directed syntheses.

### 7. SERL — Synthesis-Execution-Reingestion Loop (`src/synthesis/serl.py`)

The preceding pipeline is open-loop: corpus → project → synthesize → decode → output disappears. The decoded output is never executed, never evaluated, and never fed back into the system. This means the system cannot distinguish meaningful synthesis from garbage, the sub-tree vocabulary is frozen after initial ingestion, and no mechanism exists by which the design space can expand across synthesis cycles.

SERL closes this loop by introducing three new components:

#### Execution Sandbox (`src/execution/sandbox.py`)

A subprocess-isolated execution environment that evaluates synthesized Python code against a type-diverse set of 10 deterministic test vectors covering integers, floats, strings, lists, dicts, None, and booleans. Each synthesized function is called with every test vector inside a child process with strict timeout (2 seconds total). The sandbox computes a monotonic fitness score:

| Fitness | Condition |
|---------|-----------|
| 0.00 | `compile()` fails |
| 0.10 | Compiles but `exec()` raises on module load |
| 0.15 | No function defined, or subprocess timeout |
| 0.20 | Function rejects all input types, or all accepted inputs crash |
| 0.30 | Runs but returns `None` for all inputs |
| 0.40+ | `0.4 + 0.3 × type_coverage + 0.3 × value_diversity`, capped at 1.0 |

`TypeError` on a specific input is expected (e.g., an integer function receiving a string) and is not penalized — only `n_accepted` inputs contribute to the type coverage score. This prevents the design space from collapsing to functions that trivially accept all types.

Subprocess isolation provides OS-level security: the synthesized code runs in a separate process with no capability inheritance, preventing `__subclasses__()` escapes, filesystem access, or import hijacking.

#### Vocabulary Self-Expansion (`src/decoder/vocab_expander.py`)

When a synthesized+decoded source passes the fitness threshold (0.4 by default):

1. Parse the source into an AST
2. Extract sub-trees (depth 2–4, same range as `SubTreeVocabulary`)
3. Canonicalize each sub-tree (same alpha-renaming and type-marker substitution)
4. Check if the canonical form already exists in the vocabulary (SHA-256 deduplication)
5. If new: register as a `SubTreeAtom` and project through the FHRR pipeline
6. The next synthesis cycle can now use this atom in decoding

This is the mechanism by which F_eff (effective reachable expressibility) grows across cycles. The expansion is strictly conditional on fitness: garbage code never pollutes the vocabulary. The `expand()` method is idempotent — calling it twice with the same source adds zero duplicate atoms.

The fitness threshold of 0.4 requires the code to compile, execute without error, accept at least one input type, succeed on at least one input, AND return at least one non-None value. This is the minimum bar for "nontrivial computation."

#### SERL Loop Orchestrator (`src/synthesis/serl.py`)

Each cycle:
1. Synthesize from resonance cliques (existing pipeline)
2. Decode each synthesis to Python source (existing pipeline)
3. Execute each decoded source in the sandbox
4. For sources with fitness > threshold: expand the sub-tree vocabulary
5. Measure: did vocabulary grow? (F_eff expansion check)
6. If vocabulary grew → next cycle with expanded design space
7. If no growth for K consecutive cycles → space is closed, halt

The loop terminates when: `max_cycles` reached, K consecutive cycles with zero new vocabulary atoms (stagnation detection), or no cliques found.

The F_eff measurement is the scientific output:
- `f_eff_expansion_rate = total_new_atoms / total_syntheses_attempted`
- If > 0 after multiple cycles: the system is demonstrably open (at least partially)
- If = 0 after `stagnation_limit` cycles: the system is empirically closed — a valid scientific result

`SERLResult` records the full cycle history including per-cycle metrics: syntheses attempted/decoded/executed/above-fitness, new vocabulary atoms, vocabulary size before/after, best fitness, and best source.

#### Empirical Results

On the 4-function trivial corpus (identity, double, negate, square):

| Metric | Value |
|--------|-------|
| Cycles completed | 6 |
| Vocabulary | 8 → 9 atoms (+1 new, 12.5% expansion) |
| Best fitness | 0.97 |
| F_eff expansion | Cycle 0 (1/6 = 17%) |
| Diagnosis | Partial expansion — grew but converged |

On the engine's own 9 source files (Ouroboros SERL):

| Metric | Value |
|--------|-------|
| Self-axioms | 9 |
| Vocabulary | 1899 atoms |
| F_eff expansion rate | 0.0 |
| Diagnosis | Space is closed for self-application |

The Ouroboros result (f_eff = 0) is not a failure — it is an empirical measurement confirming that the engine's current synthesis pipeline does not produce code that passes the fitness threshold when operating on its own source. This quantitative finding corresponds to the CAGE-RSI negative diagnosis.

### 8. Structural Diff Engine (`src/analysis/structural_diff.py`)

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

### 9. Corpus Ingestion (`src/corpus/`)
**Batch Ingester**: scalable pipeline for processing directories of Python projects into VSA space.
- Walks project trees with configurable filtering (skip tests, `__init__.py`, generated code, files exceeding 64 KB)
- Graceful error handling: syntax errors, projection failures, and arena capacity exhaustion are logged and skipped without crashing
- Serialization support via pickle for persistent axiom stores, enabling corpus reuse without re-processing
- Empirically validated against 500+ Python standard library modules in under 65 seconds on commodity hardware

**Stdlib Corpus**: auto-discovers the current Python installation's standard library as a guaranteed-available corpus requiring no external downloads.

### 10. Resonance Analysis (`src/analysis/`)
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

### 11. Swarm — Multi-Agent Collective Intelligence (`src/swarm/`)

The THDSE-Swarm extends the single-agent synthesis pipeline into a deterministic multi-agent collective intelligence system. N agents communicate exclusively via raw FHRR phase arrays — no text, no LLMs, no natural language. Each agent is an independent THDSE instance with its own arena, projector, synthesizer, decoder, wall archive, sandbox, and SERL loop. A central orchestrator aggregates candidate vectors via algebraic consensus.

#### Architecture

```
┌──────────────────────────────────────────────────────┐
│                  SwarmOrchestrator                     │
│                                                       │
│  Collects V_cand from all agents                      │
│  → correlate_matrix (single FFI call)                 │
│  → Bron-Kerbosch clique extraction (ρ > 0.85)        │
│  → bundle clique members → consensus centroid         │
│  → ConstraintDecoder → Z3 → Python AST               │
│  → If UNSAT: extract V_error → broadcast to all       │
│  → If SAT: ExecutionSandbox → fitness → expand vocab  │
└──────┬──────┬──────┬──────┬───────────────────────────┘
       │      │      │      │
  ┌────▼───┐┌─▼────┐┌▼─────┐┌▼────────┐
  │Agent 0 ││Agent 1││Agent 2││Agent N-1│
  │ Arena  ││ Arena ││ Arena ││ Arena   │
  │ Synth  ││ Synth ││ Synth ││ Synth   │
  │ Decoder││Decoder││Decoder││ Decoder │
  │ SERL   ││ SERL  ││ SERL  ││ SERL    │
  │Corpus: ││Corpus:││Corpus:││Corpus:  │
  │ math/* ││ sys/* ││ net/* ││custom/* │
  └────────┘└──────┘└──────┘└─────────┘
```

**Critical design constraint — Agent Heterogeneity**: Each agent ingests a DIFFERENT corpus. Identical agents produce identical resonance landscapes and provide no collective intelligence gain. Heterogeneous corpora produce complementary wall archives and vocabularies. An UNSAT in Agent A's space may be SAT in Agent B's space.

#### VSA2VSA Communication Protocol (`src/swarm/protocol.py`)

Agents communicate via `PhaseMessage` — a minimal header (JSON, ≤200 bytes: sender_id, message_type, metadata, timestamp) followed by a raw float32 array (phases packed via `struct.pack`, not JSON-encoded strings). Message types: `"candidate"` (synthesis output), `"wall"` (V_error broadcast), `"target"` (goal injection), `"ack"`. For d=256, total message size is ~1.2 KB.

`SwarmConfig` parameterizes a swarm run: n_agents, dimension, arena_capacity, consensus_threshold (default: 0.85), fitness_threshold (default: 0.4), max_rounds, stagnation_limit, and per-agent corpus_paths.

#### ThdseAgent (`src/swarm/agent.py`)

A self-contained THDSE instance with its own `FhrrArena`, `IsomorphicProjector`, `AxiomaticSynthesizer`, `ConstraintDecoder`, `SubTreeVocabulary`, `WallArchive`, `ExecutionSandbox`, and `VocabularyExpander`. Each agent:

- **Ingests** its assigned corpus directories (or inline dict for testing)
- **Runs local synthesis**: compute resonance → extract cliques → synthesize → decode → execute → SERL gating (only candidates with fitness ≥ 0.4 are emitted as `PhaseMessage`)
- **Receives wall broadcasts**: injects V_error into local arena → `project_to_quotient_space` → records wall
- **Receives consensus**: attempts to decode and expand local vocabulary from the consensus vector

#### Resonance Clique Consensus (`src/swarm/consensus.py`)

Given N candidate vectors from M agents:
1. Inject all candidates into a temporary arena (fresh each round, never corrupts agent arenas)
2. `correlate_matrix()` — single FFI call for the N×N correlation matrix
3. Build adjacency graph where edge exists iff ρ > consensus_threshold
4. Bron-Kerbosch maximal clique extraction
5. Select largest clique (ties broken by mean resonance)
6. Bundle clique members into centroid vector via `arena.bundle()`
7. Return `ConsensusResult` with centroid phases, clique membership, and contributing agent IDs

#### SwarmOrchestrator (`src/swarm/orchestrator.py`)

Each round:
1. Each agent runs local synthesis → emits candidate `PhaseMessage`s
2. Orchestrator collects all candidates
3. Compute consensus via `compute_swarm_consensus`
4. Decode consensus centroid using the orchestrator's **merged vocabulary** (union of all agents' `SubTreeVocabulary` atoms — can decode structures no single agent could)
5. If SAT and fitness ≥ threshold: broadcast consensus to all agents for local vocabulary expansion
6. If UNSAT: broadcast V_error to ALL agents → **Global Quotient Collapse** (every agent projects out the contradiction axis from its entire memory space)
7. Track metrics: rounds, consensus rate, walls, vocabulary growth, best fitness
8. Halt when: max_rounds reached OR stagnation_limit consecutive zero-progress rounds

`SwarmResult` records the full round history, per-agent stats, convergence diagnosis, and f_eff_expansion_rate.

#### Swarm SERL (`src/swarm/swarm_serl.py`)

Top-level entry point `run_swarm_serl(config)` wraps the orchestrator in a SERL-compatible measurement framework: creates the orchestrator, runs the loop, returns `SwarmResult`.

### 12. CLI Entry Point (`src/discover.py`)
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

# Boundary cartography: map design space walls and navigate open directions
python src/discover.py --mode cartography --corpus ./projects/ --output ./report/

# SERL: closed-loop synthesis-execution-reingestion with vocabulary expansion
python src/discover.py --mode serl --corpus ./projects/ --output ./report/

# Swarm: multi-agent collective intelligence via FHRR phase arrays
python src/discover.py --mode swarm --stdlib --n-agents 3 --output ./report/

# Swarm with explicit per-agent corpora
python src/discover.py --mode swarm --corpus ./math/ ./sys/ ./net/ --n-agents 3 --output ./report/

# Analyze one or more project directories
python src/discover.py --corpus ./project_a/ ./project_b/ --output ./report/

# Save ingested corpus for later reuse
python src/discover.py --corpus ./projects/ --save corpus.pkl --output ./report/

# Reload a saved corpus without re-ingesting
python src/discover.py --load corpus.pkl --output ./report/
```

Configurable parameters: `--dimension` (default: 256), `--capacity` (default: 2M handles), `--threshold` (resonance τ, default: 0.15), `--max-files`, `--n-agents` (swarm mode, default: 3), `--consensus-threshold` (swarm mode, default: 0.85).

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
│   │   ├── axiomatic_synthesizer.py
│   │   ├── serl.py                # SERL loop orchestrator
│   │   └── wall_archive.py        # Contradiction vector persistence
│   ├── decoder/               # SMT-based constraint decoder
│   │   ├── constraint_decoder.py
│   │   ├── subtree_vocab.py   # Corpus-derived sub-tree vocabulary
│   │   ├── vocab_expander.py  # Fitness-gated vocabulary expansion
│   │   └── variable_threading.py  # Union-find variable unification
│   ├── execution/             # Subprocess-isolated code sandbox
│   │   └── sandbox.py
│   ├── corpus/                # Batch ingestion engine
│   │   ├── ingester.py
│   │   └── stdlib_corpus.py
│   ├── analysis/              # Resonance analysis + reporting
│   │   ├── resonance_analyzer.py
│   │   ├── report.py
│   │   ├── structural_diff.py     # Layer-decomposed similarity
│   │   ├── refactoring_detector.py # Duplicate & unification detection
│   │   ├── temporal_diff.py       # Version-to-version change classification
│   │   └── boundary_cartography.py # Design space boundary mapping
│   ├── swarm/                 # Multi-agent collective intelligence
│   │   ├── protocol.py            # PhaseMessage, SwarmConfig, serialization
│   │   ├── consensus.py           # Resonance clique consensus (Bron-Kerbosch)
│   │   ├── agent.py               # ThdseAgent (independent THDSE pipeline)
│   │   ├── orchestrator.py        # SwarmOrchestrator (Global Quotient Collapse)
│   │   └── swarm_serl.py          # Swarm-level SERL entry point
│   ├── utils/                 # Phase algebra utilities
│   │   └── arena_ops.py
│   └── discover.py            # CLI entry point (multi-mode)
├── tests/
│   ├── test_decoder_subtree.py    # Sub-tree vocabulary & threading tests
│   ├── test_structural_diff.py    # Structural diff engine tests
│   ├── test_boundary_cartography.py # Boundary cartography & wall archive tests
│   ├── test_batch_correlation.py  # Batch correlation correctness & equivalence tests
│   ├── test_execution_sandbox.py  # Execution sandbox fitness tests (9 tests)
│   ├── test_vocab_expander.py     # Vocabulary expansion tests (6 tests)
│   ├── test_serl.py               # SERL loop tests (5 tests)
│   └── test_swarm.py              # Swarm integration tests (22 tests)
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

# Run full test suite (including sub-tree decoding, structural diff, and boundary cartography)
python run_tests.py

# Run the empirical discovery pipeline against stdlib
python src/discover.py --stdlib --output ./report/

# Run Ouroboros self-diagnostic
python src/discover.py --mode self-diagnostic --output ./report/

# Run multi-agent swarm (3 agents, stdlib partitioned automatically)
python src/discover.py --mode swarm --stdlib --n-agents 3 --output ./report/
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
- **Multi-quotient correctness**: the multi-vector quotient projection H/⟨V₁, …, Vₖ⟩ first orthogonalizes the wall vectors via Gram-Schmidt with complex inner products, then applies sequential single-vector projection. The Gram-Schmidt step ensures that sequential projection is mathematically equivalent to simultaneous subspace removal, eliminating order-dependent numerical drift inherent in naïve sequential application. Linearly dependent wall vectors are detected (‖Vᵢ‖² < ε) and excluded, preventing rank-deficient projections.
- **SERL monotonicity**: the vocabulary size is monotonically non-decreasing across SERL cycles — no atoms are ever removed. The fitness gate (θ = 0.4) ensures only code that compiles, executes, and produces nontrivial computation can expand the vocabulary. Idempotency is enforced by SHA-256 deduplication: identical canonical sub-trees are never counted twice. The F_eff expansion rate (new_atoms / total_syntheses) is a well-defined, deterministic scalar that quantifies whether design space self-expansion has occurred.
- **Subprocess isolation**: synthesized code executes in a separate OS process via `subprocess.run()` with timeout, providing hard security boundaries. No shared memory, no capability inheritance, no `__subclasses__()` escape vectors. If the child process crashes, hangs, or produces malformed output, it is killed and scored at fitness = 0.15.
- **Boundary convergence**: the cartography loop is monotonically non-expansive — each iteration either discovers a new wall (reducing the residual dimension) or produces a successful synthesis (demonstrating reachability), guaranteeing termination within max(d, max_iterations) steps. The residual dimension estimate provides a quantitative convergence certificate: when d_eff < 0.1 · d, the design space is classified as closed.
- **Swarm consensus correctness**: the consensus centroid is computed by weighted bundling (⊕) of clique members, where weights derive from pairwise resonance within the clique. The temporary consensus arena is created fresh each round and discarded after, ensuring zero cross-contamination with agent arenas. Agent heterogeneity (different corpora → different resonance landscapes) is the mechanism that provides collective capability: structures that are UNSAT in one agent's quotient space may be SAT in another's, and the merged orchestrator vocabulary can decode structures unreachable by any individual agent.
- **Weighted bundle correctness**: `weighted_bundle_phases` computes atan2(Σ w_i sin θ_i, Σ w_i cos θ_i) per dimension — the maximum-likelihood estimator of the circular mean under von Mises weighting. When all weights are equal, this reduces exactly to standard `bundle_phases`. The output satisfies the FHRR unit-magnitude invariant after `inject_phases` (which normalizes to |z_j| = 1).
- **Layered partial consensus soundness**: adjacency rules form a logical OR over independent per-layer correlations. Rule 2 (CFG ∧ Data) detects algorithmic isomorphism — structurally different code that implements the same control flow and data wiring. False positive rate is bounded by the product of per-layer false positive rates (independence assumption under high-dimensional FHRR concentration of measure).
- **Quotient isolation safety**: targeted wall broadcast preserves non-participating agents' design spaces. The quotient projection H → H/⟨V_error⟩ is applied only to agents whose synthesis contributed to the contradiction, preventing autoimmune collapse of orthogonal design regions. Drift reconciliation propagates aged walls (older than `drift_grace_period` rounds) to non-participating agents, preventing long-term topological fragmentation while preserving short-term isolation benefits.
- **Rule-aware effective correlation**: clique scoring and member weighting use rule-appropriate correlation metrics. For algo_iso edges, the effective correlation is mean(cfg_corr, data_corr); for struct_iso edges, mean(ast_corr, cfg_corr); for full edges, final_corr. This prevents geometric discrimination where algo_iso cliques are penalized by their inherently low final_corr (which is dominated by the differing AST component after cross-layer binding).
- **Phase quantization precision**: layer phases are quantized to uint16 during serialization, mapping [-π, π] to [0, 65535]. This provides ~0.0001 radian precision (~0.006 degrees), which is orders of magnitude below the FHRR correlation noise floor at d ≥ 64. The quantization error is bounded by π/32768 ≈ 9.6×10⁻⁵ radians per dimension. Final phases remain at float32 precision for decoding fidelity.
- **Batch correlation exactness**: `correlate_matrix` computes the identical FHRR cosine Re(V_a^H · V_b) / d as `compute_correlation`, executed over all N*(N-1)/2 pairs in a single Rust FFI call. The result is bitwise identical to N*(N-1)/2 individual calls within float32 arithmetic. No approximation, no hashing, no false negatives — the speedup derives entirely from eliminating Python→Rust call overhead and exploiting cache-friendly sequential memory access.

## Changelog

### v0.7.1 — Hierarchical Nesting Decoder + Empirical Validation

**Upgrade 1 — Hierarchical Nesting Constraints.** The legacy atom-based decoder now encodes parent-child containment relationships via Z3 `parent_{Type}` variables, not just flat ordering. CFG branch and loop constraints generate nesting implications: `If → body` and `While/For → body` statements are assembled as children, not siblings. `compile_model()` builds a proper AST tree from the Z3 model's parent assignments. Incremental consistency filtering (push/pop) ensures graceful fallback to flat placement when nesting would cause UNSAT.

**Upgrade 2 — Empirical Validation Report.** New `--mode empirical` CLI mode runs a reproducible stdlib analysis pipeline and outputs `empirical_validation.json` + `empirical_validation.md`. Reports ingestion stats, resonance matrix statistics, verified structural family cliques (with intra/inter-clique resonance), top-20 cross-directory pairs with per-layer decomposition, and structural outliers. Memory-bounded to run within 4GB.

**Fix 1 — Consensus Arena Capacity.** `SwarmOrchestrator` temporary arena capacity formula changed from `4N + 100` to `max(5N + 50, 200)`, preventing capacity exhaustion when layered consensus injects 4x candidate handles.

**Fix 2 — Agent Per-Layer Phase Check.** Unified `ast_phases`/`cfg_phases`/`data_phases` null checks in `ThdseAgent` to `is not None and len > 0`, fixing inconsistent truthiness check that could silently drop empty AST phase arrays.

**Tests.** 6 new tests: 4 nesting decoder (if-body nested, loop-body nested, flat fallback, regression guard) + 2 empirical report (smoke, pair decomposition). All 94 tests pass.

### v0.7.0 — Three Structural Upgrades + Three Structural Fixes

This release addresses six algebraic deficiencies in the swarm consensus, wall broadcast, bundling, and communication mechanisms.

**Upgrade 1 — Layered Partial Topological Consensus.** The single-scalar adjacency criterion (correlate final handles) is replaced with a per-layer partial match. Three rules (logical OR) determine adjacency: (1) full structural match on the cross-layer-bound vector, (2) algorithmic isomorphism — identical CFG + identical data wiring regardless of AST, (3) structural isomorphism — identical AST + identical CFG. `PhaseMessage` now carries optional per-layer phase arrays (`ast_phases`, `cfg_phases`, `data_phases`), and `ThdseAgent` emits them from `LayeredProjection`. When per-layer phases are not provided, the consensus falls back to the original single-scalar rule (backward compatible). `ConsensusResult` includes `adjacency_rule_counts` tracking which rules fired, and per-layer centroid phases.

**Upgrade 2 — Localized Quotient Isolation.** Wall broadcasts now target only agents that participated in the consensus clique (`_broadcast_wall_targeted`). Non-participating agents' design spaces remain intact. The orchestrator also extracts the actual V_error from the decoder's meta-grammar log (UNSAT core) rather than using the consensus centroid as the contradiction vector. `SwarmRoundResult` includes `wall_target_agents` tracking which agents received each wall. The old `_broadcast_wall` is preserved as a deprecated alias.

**Upgrade 3 — Weighted Resonance Bundling.** `weighted_bundle_phases` in `arena_ops.py` computes the circular weighted mean: atan2(Σ w_i sin θ_i, Σ w_i cos θ_i). Weights derive from each member's mean resonance with all other clique members, clamped to [0.1, 1.0]. This prevents weakly resonating members from diluting the consensus signal. When all weights are equal, the result is identical to standard `bundle_phases`.

**Fix 1 — Rule-Aware Effective Correlation.** Clique scoring and member weighting previously used `final_corr` for all edges. This geometrically discriminated against algo_iso cliques: if two agents synthesize identical control flow with different syntax, their `final_corr` is near-zero (AST dominates after cross-layer binding), causing them to lose tie-breakers and receive floor weight (0.1). Now, `_effective_corr()` returns the correlation appropriate to the rule that connected each edge: mean(cfg_corr, data_corr) for algo_iso, mean(ast_corr, cfg_corr) for struct_iso, final_corr for full match. Both `clique_score` and member weights use this rule-aware metric.

**Fix 2 — Protocol Bandwidth: uint16 Phase Quantization.** Layer phases (AST, CFG, Data) are now serialized as uint16 instead of float32, mapping [-π, π] to [0, 65535]. This provides ~0.0001 radian precision — orders of magnitude below the FHRR correlation noise floor at d ≥ 64. For d=256 with 3 layers: 1536 bytes (uint16) instead of 3072 bytes (float32), a 50% reduction in layer bandwidth. Final phases remain float32 for decoding precision.

**Fix 3 — Drift Reconciliation (Anti-Tower-of-Babel).** Localized quotient isolation prevents short-term autoimmune collapse but causes long-term topological fragmentation: agents receiving different wall projections develop divergent FHRR coordinate systems until they can no longer communicate. The orchestrator now maintains a `_wall_ledger` tracking all walls and which agents have received them. Every `drift_reconciliation_interval` rounds (default: 5), walls older than `drift_grace_period` rounds (default: 3) are propagated to agents that haven't received them. This balances isolation safety (agents have time to exploit structures along the contradiction axis) against coordinate alignment (eventual convergence to shared quotient space). `SwarmConfig` gains `drift_reconciliation_interval` and `drift_grace_period` parameters. `SwarmRoundResult` gains `walls_reconciled`. `SwarmResult` gains `total_walls_reconciled`.

**Tests.** 12 new tests total (6 from upgrades + 6 from fixes): layered consensus via CFG isomorphism, full-match fallback, localized quotient isolation, weighted bundle correctness, weighted bundle FHRR unit magnitude, adjacency rule counts, algo-iso effective scoring, quantized serialization with layers, quantized serialization with None layers, drift reconciliation timing, drift reconciliation idempotency, wall ledger tracking. All 88 tests pass (66 existing + 22 swarm).

### v0.6.0 — THDSE-Swarm: Deterministic Multi-Agent Collective Intelligence

This release introduces the Swarm subsystem: N independent THDSE agents that communicate exclusively via raw FHRR phase arrays and achieve algebraic consensus through resonance clique extraction.

**VSA2VSA Protocol.** Agents exchange `PhaseMessage` values containing raw float32 phase arrays (packed via `struct.pack` for precision preservation) with minimal JSON metadata headers. No text communication, no natural language, no LLM calls — only algebraic signals.

**ThdseAgent.** Each agent is a complete, independent THDSE pipeline with its own `FhrrArena`, projector, synthesizer, decoder, wall archive, sandbox, and vocabulary expander. SERL gating ensures only candidates with fitness ≥ 0.4 are emitted to the swarm communication channel; low-quality candidates never leave the agent.

**Resonance Clique Consensus.** Candidate vectors from all agents are injected into a temporary arena (created fresh each round), correlated via a single `correlate_matrix` FFI call, and submitted to Bron-Kerbosch clique extraction. The largest clique above the consensus threshold (default: ρ > 0.85) is bundled into a centroid vector representing the algebraic consensus.

**Global Quotient Collapse.** When the orchestrator's decoder returns UNSAT for the consensus centroid, the contradiction vector V_error is broadcast to ALL agents, each of which immediately projects its entire arena to the quotient space H/⟨V_error⟩. This collapses the contradiction axis from every agent's memory simultaneously — a coordinated algebraic operation that no single agent could perform in isolation.

**Merged Vocabulary.** The orchestrator's decoder merges `SubTreeVocabulary` atoms from all agents, enabling it to decode structures that no individual agent could decode alone. This is the mechanism by which heterogeneous corpora produce collective capability exceeding any individual.

**Corpus Partitioning.** When fewer corpus directories than agents are provided, the CLI automatically partitions the stdlib into N non-overlapping segments, ensuring agent heterogeneity.

**CLI Integration.** `--mode swarm` with `--n-agents` (default: 3) and `--consensus-threshold` (default: 0.85). Outputs `swarm_result.json` with per-round and per-agent metrics.

**Tests.** 10 new integration tests: agent initialization and corpus ingestion, local synthesis candidate emission, wall broadcast propagation, consensus from identical candidates (clique size 3), no consensus from orthogonal candidates, two-agent swarm smoke test, heterogeneous candidate verification, Global Quotient Collapse consistency across 3 agents, protocol serialization round-trip, and swarm SERL entry point validation. All 76 tests pass (66 existing + 10 new).

### v0.5.0 — SERL: Synthesis-Execution-Reingestion Loop

This release closes the open-loop pipeline by introducing execution evaluation, fitness-gated vocabulary expansion, and a closed-loop orchestrator that measures F_eff (effective design space expansion) across synthesis cycles.

**Execution Sandbox.** A subprocess-isolated environment evaluates synthesized Python code against 10 type-diverse deterministic test vectors (int, float, str, list, dict, None, bool). The fitness function rewards both type coverage (accepting diverse input types) and value diversity (producing distinct return values) independently, with TypeError on typed functions explicitly not penalized. Subprocess isolation via `subprocess.run()` with timeout prevents `__subclasses__` escapes, filesystem access, and import hijacking.

**Vocabulary Self-Expansion.** When synthesized code passes the fitness threshold (0.4), its AST sub-trees are extracted, canonicalized, and registered as new vocabulary atoms — expanding the decoder's representational capacity for subsequent cycles. The expansion is idempotent (SHA-256 dedup) and strictly fitness-gated (garbage never enters the vocabulary).

**SERL Loop Orchestrator.** Each cycle synthesizes from cliques, decodes to source, executes in the sandbox, and expands the vocabulary for fitness-passing results. Stagnation detection halts the loop after K consecutive zero-expansion cycles. The output `SERLResult` records per-cycle metrics and the aggregate `f_eff_expansion_rate` — the quantitative answer to "does this system achieve design space self-expansion?"

**Ouroboros SERL.** `run_serl_self()` on `AxiomaticSynthesizer` runs the SERL loop on the engine's own 9 source files. Empirical result: f_eff_expansion_rate = 0.0 (space is closed for self-application) — confirming the CAGE-RSI negative diagnosis quantitatively.

**CLI Integration.** `--mode serl` runs the full loop with JSON output (`serl_result.json`) and human-readable summary to stdout.

**Tests.** 20 new tests: 9 sandbox (full fitness gradient), 6 vocabulary expander (threshold gating, dedup, projection, activation), 5 SERL loop (cycle history, graceful termination, vocabulary expansion proof, garbage rejection, stagnation detection).

### v0.4.1 — Batch Correlation and Layer-Independent Exact Filtering

This release eliminates the O(N²) Python→Rust round-trip overhead in all pairwise correlation consumers by introducing batch correlation primitives in the Rust HDC core and refactoring the Python-side consumers to exploit them.

**Batch Correlation Primitives (Rust).** Two new methods on `FhrrArena`: `correlate_matrix(handles)` computes the full upper-triangle of the pairwise correlation matrix in a single FFI call, returning a flat `Vec<f32>` of length N*(N-1)/2; `correlate_matrix_subset(queries, targets)` computes the full |Q|×|T| asymmetric correlation matrix. Both execute the same FHRR cosine as `compute_correlation` — algebraically exact, zero approximation. The computation remains O(N²·d) but runs entirely inside Rust with sequential memory access, reducing wall-clock time by eliminating per-pair Python→Rust marshalling overhead.

**ResonanceMatrix Refactor.** `compute_full()` replaces N*(N+1)/2 individual `compute_correlation` calls with a single `correlate_matrix` call plus N self-correlation calls. Output is bitwise identical within float32 epsilon.

**RefactoringDetector Layer-Independent Exact Filtering.** The O(N²) Python loop calling `compare_layers()` (4 correlations per pair) is replaced by per-layer batch matrix computation. Rule 1 (structural duplicate: sim_ast > 0.7 ∧ sim_cfg > 0.7) computes the full AST correlation matrix, filters surviving pairs, then checks CFG correlations from a pre-computed CFG matrix. Rule 2 (interface unification: sim_data > 0.8 ∧ sim_ast < 0.3) computes the full data-dependency correlation matrix, filters, and checks the AST ceiling from the already-computed AST matrix. This approach avoids pre-filtering on the composite `final_handle` (which would dilute layer-specific signals and introduce false negatives for Rule 2), while reducing the call count from 4·N*(N-1)/2 individual FFI calls to 2–3 batch calls.

**WallArchive Refactor.** `compute_wall_resonance()` replaces n*(n+1)/2 individual calls with a single `correlate_matrix` call plus n self-correlation calls.

### v0.4.0 — Boundary Cartography

This release introduces the Boundary Cartography subsystem, which extends the engine's synthesis capabilities from open-loop clique-based generation to closed-loop, boundary-aware design space exploration.

**Wall Archive.** A persistent store for Contradiction Vectors (V_error) produced by the constraint decoder upon UNSAT events. Each wall record captures the arena handle, phase array (enabling cross-session persistence), UNSAT core labels, and provenance metadata. Walls are classified as active, resolved, or orphaned; revalidation is triggered deterministically upon design space modifications (dimension expansion, vocabulary update) via Z3 re-solve. Bron-Kerbosch clique extraction over the wall resonance matrix identifies coherent boundary regions — clusters of structurally similar contradictions.

**Residual Dimension Estimation.** A spectral analysis module that quantifies the accessible fraction of the design space after projecting out all known walls. Deterministic hash-seeded probe vectors are projected through the multi-quotient space, and approximate singular values of the resulting correlation matrix are computed via Gershgorin disc bounds. The effective dimension d_eff serves as a convergence certificate: when d_eff < 0.1 · d, the space is classified as closed.

**Open Direction Extraction.** Principal component analysis over the residual space identifies the top-k directions of maximum variance — the unexplored regions where synthesis is most likely to produce novel, non-contradictory structures. Directed synthesis selects corpus axioms by correlation with these open directions rather than by standard resonance cliques, biasing generation away from known boundary walls.

**Multi-Quotient Projection (Rust/AVX2).** The HDC core is extended with `project_to_multi_quotient_space`, which removes an entire k-dimensional subspace simultaneously via Gram-Schmidt orthogonalization with complex inner products followed by sequential single-vector projection. This eliminates order-dependent numerical drift inherent in naïve sequential application, with complexity O(k² · d + k · head · d).

**Cartography CLI Mode.** A new `--mode cartography` integrates the full iterative loop — wall collection, boundary mapping, residual estimation, open direction extraction, directed synthesis, and convergence detection — into the CLI entry point, with structured output to `boundary_map.json`.

### v0.3.0 — Architectural Upgrade (4-Leap Expansion)

This release introduces four interconnected subsystems that collectively advance the engine from prototype-stage resonance analysis toward a structurally grounded code synthesis and evolution analysis platform.

**Leap 1 — Sub-Tree Vocabulary Synthesis.** The decoder's abstract atom vocabulary (individual AST node types) is supplanted by a corpus-derived sub-tree vocabulary. Concrete AST fragments of depth 2–4 are extracted, canonicalized via alpha-renaming and type-marker substitution, deduplicated by SHA-256, and projected into the FHRR space. A union-find–based variable threading engine ensures consistent naming when composing selected sub-trees into a single program. The SMT encoding is extended accordingly: Z3 selects which sub-trees to include and in what order, and the compiler assembles real AST fragments verified via `compile()`.

**Leap 2 — Structural Diff Engine.** A layer-decomposed similarity analysis module decomposes the scalar resonance metric into per-layer components (sim_ast, sim_cfg, sim_data), enabling diagnostic explanations of structural similarity and divergence. Two downstream detectors leverage this decomposition: a refactoring detector that identifies structural duplicates and interface unification candidates, and a temporal analysis module that classifies version-to-version changes by their topological signature (structural refactor, logic change, variable rewiring, complete rewrite, or cosmetic change).

**Leap 3 — Ouroboros Self-Diagnostic.** The autopoietic self-reference capability is extended with a quantitative self-diagnostic: the engine ingests its own source modules, computes per-layer resonance among them, and outputs a structured report. A complementary synthesis quality test performs the full closed-loop pipeline (synthesize → decode → re-project → cosine similarity) to measure round-trip fidelity.

**Leap 4 — CLI and Pipeline Integration.** The CLI entry point is restructured around a `--mode` selector supporting five operational modes: `analysis` (default resonance pipeline), `diff` (pairwise structural comparison), `refactor` (corpus-wide duplicate and unification detection), `synthesis` (clique extraction and code generation), and `self-diagnostic` (Ouroboros self-analysis). The report generator is extended with sections for structural diffs, refactoring recommendations, and temporal change classifications.
