"""
Tests for the Multi-Layer Topological Phase-Transition Engine.

Covers all six modules:
  1. hdc_core       — Rust FHRR Arena (VSA engine)
  2. topology       — MultiLayerGraphBuilder + TopologicalASTGraphCA
  3. projection     — IsomorphicProjector → LayeredProjection
  4. synthesis      — AxiomaticSynthesizer (per-layer phase transition)
  5. decoder        — ConstraintDecoder (layer-aware unbinding → SMT → AST)
  6. utils          — arena_ops (phase algebra)
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
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=0.05)

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
                                activation_threshold=0.08,
                                verification_threshold=0.10)

    code = "def foo(x):\n    if x <= 1:\n        return 1\n    return x * foo(x - 1)\n"
    projection = proj.project(code)

    # Layer-aware probe
    constraints = decoder.probe_layered(projection)

    # CRITICAL ASSERTION: activated node types must include actual code structure
    expected_types = {"FunctionDef", "If", "Return"}
    activated = set(constraints.active_node_types)
    overlap = expected_types & activated
    assert len(overlap) >= 2, (
        f"Layer-aware probe must recover real structure. "
        f"Expected at least 2 of {expected_types}, got {activated}"
    )

    # Decode must produce valid Python (not just 'pass')
    source = decoder.decode_to_source(projection)
    assert source is not None, "Decoder must not UNSAT on valid projected code"
    assert source.strip() != "pass", "Decoder must produce non-degenerate output"

    import ast as ast_mod
    ast_mod.parse(source)

    print(f"[PASS] constraint decoder")
    print(f"       Activated types: {sorted(activated)}")
    print(f"       Decoded: {source}")


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
    synth = AxiomaticSynthesizer(arena, proj, resonance_threshold=0.05)
    decoder = ConstraintDecoder(arena, proj, dim,
                                activation_threshold=0.08)

    corpus = {
        "base_case":   "def f(n):\n    if n <= 0:\n        return 0\n    return n + f(n - 1)\n",
        "accumulator": "def g(n):\n    acc = 0\n    while n > 0:\n        acc += n\n        n -= 1\n    return acc\n",
    }
    synth.ingest_batch(corpus)
    results = synth.synthesize_all(min_clique_size=2)

    decoded_count = 0
    for clique, synth_proj in results:
        source = decoder.decode_to_source(synth_proj)
        if source is not None and source.strip() != "pass":
            import ast as ast_mod
            ast_mod.parse(source)
            decoded_count += 1

    assert decoded_count > 0, "At least one synthesis must decode to valid Python"
    print(f"[PASS] full pipeline ({decoded_count}/{len(results)} syntheses decoded to valid Python)")


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
    print("\nAll applicable tests passed.")
