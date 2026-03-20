"""
Smoke tests for the two extracted core modules:
  1. hdc_core (Rust FHRR Arena — VSA engine)
  2. topology (MultiLayerGraphBuilder + TopologicalASTGraphCA)
"""
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def test_topology_core():
    """MultiLayerGraphBuilder produces a DiGraph with ast, cfg, data_dep edges."""
    from src.topology.ast_graph_ca import TopologicalASTGraphCA
    from src.topology.multi_layer_builder import MultiLayerGraphBuilder

    code = """\
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    builder = MultiLayerGraphBuilder(steps=5)
    G = builder.build(code)

    layers = {d.get("layer") for _, _, d in G.edges(data=True)}
    assert "ast" in layers, "Missing AST layer"
    assert "cfg" in layers, "Missing CFG layer"
    assert "data_dep" in layers, "Missing data-dep layer"

    # CA evolution should populate complex-valued states
    ca = TopologicalASTGraphCA(steps=5)
    G2 = ca.code_to_graph(code)
    G2 = ca.evolve(G2)
    for node in G2.nodes:
        state = G2.nodes[node]["state"]
        assert abs(abs(state) - 1.0) < 1e-6, "CA state not unit-magnitude"

    print("[PASS] topology core")


def test_hdc_core():
    """FhrrArena allocates, injects, binds, bundles, and correlates."""
    try:
        import hdc_core
    except ImportError:
        print("[SKIP] hdc_core not compiled — run `maturin develop` in src/hdc_core/")
        return

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


if __name__ == "__main__":
    test_topology_core()
    test_hdc_core()
    print("\nAll core module tests passed.")
