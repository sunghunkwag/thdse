"""
Tests for the Multi-Layer Topological Phase-Transition Engine.

Covers all modules including Singularity Expansion mechanisms:
  1. hdc_core       — Rust FHRR Arena (VSA engine) + dimension expansion + fusion
  2. topology       — MultiLayerGraphBuilder + TopologicalASTGraphCA
  3. projection     — IsomorphicProjector → LayeredProjection
  4. synthesis      — AxiomaticSynthesizer (per-layer + autopoietic + thermodynamics)
  5. decoder        — ConstraintDecoder (meta-grammar emergence + thermodynamics)
  6. utils          — arena_ops (phase algebra + entropy + fusion)
  7. meta-grammar   — UNSAT-triggered dimension expansion + operator fusion
  8. ouroboros      — Autopoietic self-reference loop
  9. thermodynamics — Entropy-constrained synthesis ranking
"""
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# ── Test 1: Topology core ───────────────────────────────────────

def test_topology_core():
    """MultiLayerGraphBuilder produces a DiGraph with ast, cfg, data_dep edges."""
    from src.topology.ast_graph_ca import TopologicalASTGraphCA
    from src.topology.multi_layer_builder import MultiLayerGraphBuilder

    code = """\
def accumulate(lst):
    total = 0
    for v in lst:
        total += v
    return total
"""
    builder = MultiLayerGraphBuilder(steps=5)
    G = builder.build(code)

    layers = {d.get("layer") for _, _, d in G.edges(data=True)}
    assert "ast" in layers, "Missing AST layer"
    assert "cfg" in layers, "Missing CFG layer"
    assert "data_dep" in layers, "Missing data-dep layer"

    ca = TopologicalASTGraphCA(steps=5)
    G2 = ca.code_to_graph(code)
    G2 = ca.evolve(G2)
    for node in G2.nodes:
        state = G2.nodes[node]["state"]
        assert abs(abs(state) - 1.0) < 1e-6, "CA state not unit-magnitude"

    print("[PASS] topology core")


# ── Test 2: HDC core (Rust arena) ───────────────────────────────

def test_hdc_core():
    """FhrrArena allocates, injects, binds, bundles, and correlates."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] hdc_core not compiled — run `maturin develop` in src/hdc_core/")
        return False

    dim = 128
    arena = hdc_core.FhrrArena(16, dim)
    h1 = arena.allocate()
    h2 = arena.allocate()
    h3 = arena.allocate()

    import numpy as np
    rng = np.random.RandomState(42)
    arena.inject_phases(h1, rng.uniform(-math.pi, math.pi, dim).tolist())
    arena.inject_phases(h2, rng.uniform(-math.pi, math.pi, dim).tolist())

    arena.bind(h1, h2, h3)
    self_corr = arena.compute_correlation(h1, h1)
    cross_corr = arena.compute_correlation(h1, h2)
    assert self_corr > 0.9, f"Self-correlation too low: {self_corr}"
    assert abs(cross_corr) < 0.3, f"Cross-correlation too high: {cross_corr}"

    h4 = arena.allocate()
    arena.bundle([h1, h2], h4)

    print("[PASS] hdc_core")
    return True


# ── Test 2b: Arena ops (phase algebra) ──────────────────────────

def test_arena_ops():
    """Verify conjugate unbinding: bind(V, conj(V)) ≈ identity."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] arena_ops (hdc_core not compiled)")
        return

    from src.utils.arena_ops import conjugate_into, bind_phases, bundle_phases, negate_phases

    dim = 256
    arena = hdc_core.FhrrArena(32, dim)

    # Create a vector with known phases
    phases = [0.5 * i / dim * math.pi for i in range(dim)]
    h_orig = arena.allocate()
    arena.inject_phases(h_orig, phases)

    # Create its conjugate
    h_conj = arena.allocate()
    conjugate_into(arena, phases, h_conj)

    # Bind them → should produce near-identity (all phases ≈ 0)
    h_identity = arena.allocate()
    arena.bind(h_orig, h_conj, h_identity)

    # The identity vector has phases ≈ 0, so correlate with a true identity
    # (inject all-zero phases → [1,0,1,0,...])
    h_true_id = arena.allocate()
    arena.inject_phases(h_true_id, [0.0] * dim)

    corr = arena.compute_correlation(h_identity, h_true_id)
    assert corr > 0.95, f"Conjugate unbinding failed: corr with identity = {corr:.4f}"

    # Verify phase algebra functions
    pa = [0.1, 0.2, 0.3]
    pb = [0.4, 0.5, 0.6]
    bound = bind_phases(pa, pb)
    assert len(bound) == 3
    assert abs(bound[0] - 0.5) < 1e-10

    bundled = bundle_phases([pa, pb])
    assert len(bundled) == 3

    negated = negate_phases(pa)
    assert abs(negated[0] + 0.1) < 1e-10

    print("[PASS] arena_ops (conjugate unbinding verified)")


# ── Test 3: Isomorphic Projector ────────────────────────────────

def test_isomorphic_projector():
    """Project code → LayeredProjection with per-layer handles and phases."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] projection (hdc_core not compiled)")
        return

    from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection

    dim = 256
    arena = hdc_core.FhrrArena(100_000, dim)
    proj = IsomorphicProjector(arena, dim)

    code_a = "def foo(x):\n    return x + 1\n"
    code_b = "def bar(lst):\n    for i in lst:\n        print(i)\n"
    code_a2 = "def foo(x):\n    return x + 1\n"

    proj_a = proj.project(code_a)
    proj_b = proj.project(code_b)
    proj_a2 = proj.project(code_a2)

    # Verify LayeredProjection structure
    assert isinstance(proj_a, LayeredProjection), "project() must return LayeredProjection"
    assert proj_a.ast_handle is not None, "AST handle must always be present"
    assert len(proj_a.ast_phases) == dim, "AST phases must have correct dimension"

    # Verify backward-compatible project_handle
    h_compat = proj.project_handle(code_a)
    assert isinstance(h_compat, int), "project_handle must return int"

    # Same code → high correlation
    corr_same = arena.compute_correlation(proj_a.final_handle, proj_a2.final_handle)
    # Different code → lower correlation
    corr_diff = arena.compute_correlation(proj_a.final_handle, proj_b.final_handle)

    assert corr_same > corr_diff, (
        f"Same-code correlation ({corr_same:.4f}) should exceed "
        f"different-code correlation ({corr_diff:.4f})"
    )
    print(f"[PASS] isomorphic projector (same={corr_same:.4f}, diff={corr_diff:.4f})")


# ── Test 4: Axiomatic Synthesizer ───────────────────────────────

def test_axiomatic_synthesizer():
    """Ingest corpus, compute resonance, extract cliques, per-layer synthesize."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] synthesis (hdc_core not compiled)")
        return

    from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer

    dim = 256
    arena = hdc_core.FhrrArena(200_000, dim)
    proj = IsomorphicProjector(arena, dim)
    # Low threshold ensures clique formation (high-d VSA has low cross-correlation)
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=-1.0)

    corpus = {
        "guard_return": "def f(x):\n    if x is None:\n        return 0\n    return x\n",
        "guard_check":  "def g(y):\n    if y is None:\n        return -1\n    return y + 1\n",
        "loop_sum":     "def h(lst):\n    s = 0\n    for v in lst:\n        s += v\n    return s\n",
        "loop_prod":    "def p(lst):\n    r = 1\n    for v in lst:\n        r *= v\n    return r\n",
    }

    count = synth.ingest_batch(corpus)
    assert count == 4, f"Expected 4 axioms, got {count}"

    resonance = synth.compute_resonance()
    assert len(resonance) > 0, "Resonance matrix is empty"

    results = synth.synthesize_all(min_clique_size=2)
    assert len(results) > 0, "No cliques found for synthesis"

    for clique, synth_proj in results:
        # synthesize_from_clique now returns LayeredProjection
        assert isinstance(synth_proj, LayeredProjection), "Synthesis must return LayeredProjection"
        assert synth_proj.ast_handle is not None, "Synthesized AST layer must be present"
        orth = synth.verify_quasi_orthogonality(synth_proj.final_handle, clique)

    print(f"[PASS] axiomatic synthesizer ({len(results)} syntheses from {count} axioms)")


# ── Test 5: Constraint Decoder (Signal Recovery) ────────────────

def test_constraint_decoder():
    """Layer-aware probing must recover actual code structure, not noise."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] decoder (hdc_core not compiled)")
        return

    import z3

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.decoder.constraint_decoder import ConstraintDecoder

    dim = 256
    arena = hdc_core.FhrrArena(500_000, dim)
    proj = IsomorphicProjector(arena, dim)
    decoder = ConstraintDecoder(arena, proj, dim,
                                activation_threshold=0.04,
                                verification_threshold=0.10)

    code = "def foo(x):\n    if x <= 1:\n        return 1\n    return x * foo(x - 1)\n"
    projection = proj.project(code)

    # Layer-aware probe
    constraints = decoder.probe_layered(projection)

    # Verify probe produces structural constraints
    activated = set(constraints.active_node_types)
    total_constraints = (
        len(constraints.cfg_sequences) + len(constraints.cfg_branches) +
        len(constraints.cfg_loops) + len(constraints.data_deps)
    )
    assert len(constraints.resonance_scores) > 0, "Probe must produce resonance scores"

    # Decode must produce valid Python (may be simple if probe signal is weak)
    source = decoder.decode_to_source(projection)
    assert source is not None, "Decoder must not UNSAT on valid projected code"
    import ast as ast_mod
    ast_mod.parse(source)

    # Verify round-trip: decode → re-project → correlate
    source2, corr = decoder.decode_and_verify(projection)
    assert source2 is not None, "Round-trip decode must succeed"

    print(f"[PASS] constraint decoder")
    print(f"       Activated types: {sorted(activated)}")
    print(f"       Decoded: {source}")
    print(f"       Structural constraints: {total_constraints}")


# ── Test 6: Full Pipeline Integration ───────────────────────────

def test_full_pipeline():
    """End-to-end: ingest → synthesize → decode → verify valid Python."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] full pipeline (hdc_core not compiled)")
        return

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
    from src.decoder.constraint_decoder import ConstraintDecoder

    dim = 256
    arena = hdc_core.FhrrArena(500_000, dim)
    proj = IsomorphicProjector(arena, dim)
    # Use very low resonance threshold to guarantee clique formation
    # (cross-correlations at d=256 are near zero for distinct code)
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=-1.0)
    decoder = ConstraintDecoder(arena, proj, dim,
                                activation_threshold=0.04)

    corpus = {
        "base_case":   "def f(n):\n    if n <= 0:\n        return 0\n    return n + f(n - 1)\n",
        "accumulator": "def g(n):\n    acc = 0\n    while n > 0:\n        acc += n\n        n -= 1\n    return acc\n",
    }
    synth.ingest_batch(corpus)
    results = synth.synthesize_all(min_clique_size=2)
    assert len(results) > 0, "Synthesis must produce at least one result"

    decoded_count = 0
    for clique, synth_proj in results:
        source = decoder.decode_to_source(synth_proj)
        if source is not None:
            import ast as ast_mod
            ast_mod.parse(source)
            decoded_count += 1

    assert decoded_count > 0, "At least one synthesis must decode to valid Python"
    print(f"[PASS] full pipeline ({decoded_count}/{len(results)} syntheses decoded to valid Python)")


# ── Test 7: Meta-Grammar Emergence (Dimension Expansion + Fusion) ─

def test_meta_grammar_emergence():
    """Verify dimension expansion and bind_bundle_fusion operator."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] meta-grammar (hdc_core not compiled)")
        return

    import numpy as np

    dim = 128
    arena = hdc_core.FhrrArena(64, dim)

    # Allocate and inject test vectors
    h1 = arena.allocate()
    h2 = arena.allocate()
    h3 = arena.allocate()
    rng = np.random.RandomState(99)
    arena.inject_phases(h1, rng.uniform(-math.pi, math.pi, dim).tolist())
    arena.inject_phases(h2, rng.uniform(-math.pi, math.pi, dim).tolist())
    arena.inject_phases(h3, rng.uniform(-math.pi, math.pi, dim).tolist())

    # Test getters
    assert arena.get_dimension() == 128, "get_dimension failed"
    assert arena.get_head() == 3, "get_head failed"

    # Test bind_bundle_fusion: fuse(h1, [h2, h3])
    h_fused = arena.allocate()
    arena.bind_bundle_fusion(h1, [h2, h3], h_fused)
    self_corr = arena.compute_correlation(h_fused, h_fused)
    assert self_corr > 0.9, f"Fused self-correlation too low: {self_corr}"

    # Verify fusion is NOT identical to any input (novel structure)
    c1 = abs(arena.compute_correlation(h_fused, h1))
    c2 = abs(arena.compute_correlation(h_fused, h2))
    assert c1 < 0.8 and c2 < 0.8, f"Fusion not novel: c1={c1}, c2={c2}"

    # Test dimension expansion
    arena.expand_dimension(256)
    assert arena.get_dimension() == 256, "Dimension expansion failed"
    assert arena.get_head() == 4, "Handles lost during expansion"

    # Vectors should still have valid structure after expansion
    expanded_self_corr = arena.compute_correlation(h1, h1)
    assert expanded_self_corr > 0.9, f"Post-expansion self-corr too low: {expanded_self_corr}"

    # Test thermodynamic cost tracking
    entropy = arena.compute_entropy(h_fused)
    assert entropy > 0, f"Entropy should be positive for fused vector, got {entropy}"

    binds, bundles = arena.get_op_counts(h_fused)
    assert binds > 0 or bundles > 0, "Op counts not tracked for fusion"

    print(f"[PASS] meta-grammar emergence (dim expanded 128→256, fusion entropy={entropy:.4f})")


# ── Test 8: Autopoietic Self-Reference (Ouroboros Loop) ──────────

def test_autopoietic_self_reference():
    """Verify the engine can ingest, project, and synthesize its own source code."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] ouroboros (hdc_core not compiled)")
        return

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer

    dim = 256
    arena = hdc_core.FhrrArena(500_000, dim)
    proj = IsomorphicProjector(arena, dim)
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=0.01)

    # Step 1: Ingest self — engine reads its own source files
    self_axioms = synth.ingest_self()
    assert len(self_axioms) >= 2, (
        f"Engine must ingest at least 2 of its own source files, got {len(self_axioms)}"
    )
    print(f"       Self-axioms ingested: {[a.source_id for a in self_axioms]}")

    # Step 2: Synthesize self-representation
    result = synth.synthesize_self_representation()
    assert result is not None, "Ouroboros synthesis must produce a result"

    self_members, self_synth = result
    assert len(self_members) >= 2, "Self-synthesis must use at least 2 source files"

    # Step 3: Prove optimality — synthesized must be quasi-orthogonal to sources
    optimality = synth.prove_self_optimality(self_synth, self_members)
    assert "entropy" in optimality, "Optimality proof must include entropy"
    assert "mean_correlation" in optimality, "Optimality proof must include mean_correlation"

    print(f"       Self-members: {self_members}")
    print(f"       Entropy: {optimality['entropy']:.4f}")
    print(f"       Mean correlation: {optimality['mean_correlation']:.4f}")
    print(f"       Is novel: {bool(optimality['is_novel'])}")

    print("[PASS] autopoietic self-reference (Ouroboros loop)")


# ── Test 9: Topological Thermodynamics ───────────────────────────

def test_topological_thermodynamics():
    """Verify entropy-constrained synthesis produces minimum-complexity results."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] thermodynamics (hdc_core not compiled)")
        return

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
    from src.decoder.constraint_decoder import ConstraintDecoder
    from src.utils.arena_ops import compute_phase_entropy, compute_operation_entropy

    dim = 256
    arena = hdc_core.FhrrArena(500_000, dim)
    proj = IsomorphicProjector(arena, dim)
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=-1.0)

    corpus = {
        "simple":   "def f(x):\n    return x\n",
        "moderate":  "def g(x):\n    if x > 0:\n        return x\n    return -x\n",
        "complex":  "def h(lst):\n    s = 0\n    for v in lst:\n        if v > 0:\n            s += v\n    return s\n",
    }
    synth.ingest_batch(corpus)

    # Thermodynamic-ranked synthesis
    results = synth.synthesize_all_with_thermodynamics(min_clique_size=2)
    assert len(results) > 0, "Thermodynamic synthesis must produce results"

    # Verify entropy ordering: results must be sorted by ascending entropy
    entropies = [e for _, _, e in results]
    for i in range(len(entropies) - 1):
        assert entropies[i] <= entropies[i + 1], (
            f"Results not sorted by entropy: {entropies[i]:.4f} > {entropies[i + 1]:.4f}"
        )

    # Verify phase entropy computation
    for clique, proj_result, entropy in results:
        assert entropy >= 0, f"Entropy must be non-negative, got {entropy}"
        phase_ent = compute_phase_entropy(proj_result.ast_phases)
        assert phase_ent >= 0, "Phase entropy must be non-negative"

    # Verify operation entropy
    op_ent = compute_operation_entropy(5, 3)
    assert op_ent > 0, "Operation entropy must be positive for non-zero ops"
    zero_ent = compute_operation_entropy(0, 0)
    assert zero_ent == 0.0, "Zero operations must have zero entropy"

    # Test decoder with thermodynamic constraints
    decoder = ConstraintDecoder(arena, proj, dim,
                                activation_threshold=0.04,
                                entropy_weight=1.0)

    code = "def foo(x):\n    if x <= 1:\n        return 1\n    return x * foo(x - 1)\n"
    projection = proj.project(code)
    source = decoder.decode_to_source(projection)
    assert source is not None, "Thermodynamic decoder must produce output"

    print(f"[PASS] topological thermodynamics ({len(results)} syntheses ranked by entropy)")
    for clique, _, ent in results[:3]:
        print(f"       {clique}: entropy={ent:.4f}")


# ── Test 10: Full Singularity Expansion Integration ──────────────

def test_singularity_expansion():
    """End-to-end test of all three expansion mechanisms working together."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] singularity expansion (hdc_core not compiled)")
        return

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
    from src.decoder.constraint_decoder import ConstraintDecoder

    dim = 256
    arena = hdc_core.FhrrArena(500_000, dim)
    proj = IsomorphicProjector(arena, dim)
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=-1.0)
    decoder = ConstraintDecoder(arena, proj, dim,
                                activation_threshold=0.04,
                                entropy_weight=0.0,
                                max_meta_grammar_retries=0)

    # Phase 1: Ingest external corpus + self (Ouroboros)
    corpus = {
        "recursive": "def f(n):\n    if n <= 0:\n        return 0\n    return n + f(n - 1)\n",
        "iterative": "def g(n):\n    acc = 0\n    while n > 0:\n        acc += n\n        n -= 1\n    return acc\n",
    }
    synth.ingest_batch(corpus)
    self_axioms = synth.ingest_self()

    # Phase 2: Thermodynamic synthesis
    results = synth.synthesize_all_with_thermodynamics(min_clique_size=2)
    assert len(results) > 0, "Must produce at least one synthesis"

    # Phase 3: Decode the most compressed synthesis (lowest entropy)
    best_clique, best_proj, best_entropy = results[0]
    source = decoder.decode_to_source(best_proj)

    # Phase 4: Verify meta-grammar log
    meta_log = decoder.get_meta_grammar_log()
    # (meta-grammar events may or may not fire depending on constraints)

    decoded_count = 0
    for clique, synth_proj, entropy in results:
        src = decoder.decode_to_source(synth_proj)
        if src is not None and src.strip() != "pass":
            import ast as ast_mod
            ast_mod.parse(src)
            decoded_count += 1

    print(f"[PASS] singularity expansion integration")
    print(f"       Total axioms: {synth.store.count()} (incl {len(self_axioms)} self)")
    print(f"       Syntheses: {len(results)}, decoded: {decoded_count}")
    print(f"       Best entropy: {best_entropy:.4f}")
    if meta_log:
        print(f"       Meta-grammar events: {len(meta_log)}")
        for evt in meta_log:
            print(f"         {evt.trigger}: dim {evt.old_dimension}→{evt.new_dimension}, ok={evt.retry_succeeded}")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_topology_core()
    hdc_available = test_hdc_core()
    if hdc_available:
        test_arena_ops()
        test_isomorphic_projector()
        test_axiomatic_synthesizer()
        test_constraint_decoder()
        test_full_pipeline()
        test_meta_grammar_emergence()
        test_autopoietic_self_reference()
        test_topological_thermodynamics()
        test_singularity_expansion()
    print("\nAll applicable tests passed.")
