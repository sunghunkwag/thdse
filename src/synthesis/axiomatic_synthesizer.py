"""
Axiomatic Synthesizer — Deterministic phase-transition engine that accumulates
'Axiom Vectors' from stable codebases and synthesizes novel structural
hypervectors through mathematically provable binding operations.

No randomness. No LLMs. Every operation is a deterministic function of the
input corpus and the VSA algebra (FHRR bind/bundle/correlate).

Mathematical foundation:
  - Each stable codebase c_i is projected to an axiom vector A_i ∈ S^{d-1} (unit torus).
  - Structural resonance between axioms: ρ(A_i, A_j) = correlate(A_i, A_j).
  - Per-layer synthesis preserves layer decomposition:
      S_ast  = A₁_ast  ⊗ A₂_ast  ⊗ ... ⊗ Aₖ_ast
      S_cfg  = A₁_cfg  ⊗ A₂_cfg  ⊗ ... ⊗ Aₖ_cfg
      S_data = A₁_data ⊗ A₂_data ⊗ ... ⊗ Aₖ_data
      S_final = S_ast ⊗ S_cfg ⊗ S_data
"""

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.utils.arena_ops import bind_phases, negate_phases


@dataclass
class Axiom:
    """An axiom vector with provenance metadata and layered projection."""
    projection: LayeredProjection
    source_id: str
    resonance_profile: Dict[str, float] = field(default_factory=dict)

    @property
    def handle(self) -> int:
        """Backward-compatible: the final cross-layer-bound handle."""
        return self.projection.final_handle


class AxiomStore:
    """Flat, indexed store of axiom vectors within a shared arena."""

    def __init__(self, arena: Any):
        self.arena = arena
        self.axioms: Dict[str, Axiom] = {}

    def register(self, source_id: str, projection: LayeredProjection) -> Axiom:
        axiom = Axiom(projection=projection, source_id=source_id)
        self.axioms[source_id] = axiom
        return axiom

    def get(self, source_id: str) -> Optional[Axiom]:
        return self.axioms.get(source_id)

    def all_ids(self) -> List[str]:
        return sorted(self.axioms.keys())

    def count(self) -> int:
        return len(self.axioms)


class ResonanceMatrix:
    """Computes and caches the full pairwise correlation matrix over the axiom store.

    Uses final_handle (cross-layer bound) for resonance — this captures
    the full structural similarity including cross-layer interactions.
    """

    def __init__(self, arena: Any, store: AxiomStore):
        self.arena = arena
        self.store = store
        self._matrix: Dict[Tuple[str, str], float] = {}

    def compute_full(self) -> Dict[Tuple[str, str], float]:
        self._matrix.clear()
        ids = self.store.all_ids()
        for i, id_a in enumerate(ids):
            ax_a = self.store.axioms[id_a]
            for id_b in ids[i:]:
                ax_b = self.store.axioms[id_b]
                corr = self.arena.compute_correlation(ax_a.handle, ax_b.handle)
                self._matrix[(id_a, id_b)] = corr
                self._matrix[(id_b, id_a)] = corr
                ax_a.resonance_profile[id_b] = corr
                ax_b.resonance_profile[id_a] = corr
        return self._matrix

    def get(self, id_a: str, id_b: str) -> float:
        return self._matrix.get((id_a, id_b), 0.0)


class AxiomaticSynthesizer:
    """Ingests stable codebases, discovers resonance cliques, and synthesizes
    novel structural hypervectors through deterministic per-layer binding.

    Pipeline:
      1. Ingest: code → IsomorphicProjector → LayeredProjection → AxiomStore
      2. Resonate: compute pairwise correlation matrix
      3. Clique extraction: find all maximal sets where ∀ pairs exceed threshold τ
      4. Synthesize: per-layer chain-bind → cross-layer bind → LayeredProjection
    """

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        resonance_threshold: float = 0.15,
    ):
        self.arena = arena
        self.projector = projector
        self.store = AxiomStore(arena)
        self.resonance = ResonanceMatrix(arena, self.store)
        self.tau = resonance_threshold
        self._synthesis_log: List[Tuple[List[str], LayeredProjection]] = []

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, source_id: str, code: str) -> Axiom:
        """Project a codebase and register it as an axiom."""
        projection = self.projector.project(code)
        return self.store.register(source_id, projection)

    def ingest_batch(self, corpus: Dict[str, str]) -> int:
        """Ingest multiple codebases. Returns count of axioms registered."""
        for source_id in sorted(corpus.keys()):
            self.ingest(source_id, corpus[source_id])
        return self.store.count()

    # ── Resonance analysis ───────────────────────────────────────

    def compute_resonance(self) -> Dict[Tuple[str, str], float]:
        return self.resonance.compute_full()

    # ── Clique extraction (Bron-Kerbosch, deterministic) ─────────

    def extract_cliques(self, min_size: int = 2) -> List[List[str]]:
        ids = self.store.all_ids()
        adj: Dict[str, set] = {sid: set() for sid in ids}
        for id_a in ids:
            for id_b in ids:
                if id_a != id_b and self.resonance.get(id_a, id_b) > self.tau:
                    adj[id_a].add(id_b)

        cliques: List[List[str]] = []
        self._bron_kerbosch(set(), set(ids), set(), adj, cliques)
        return sorted(
            [c for c in cliques if len(c) >= min_size],
            key=lambda c: (-len(c), c),
        )

    def _bron_kerbosch(
        self, R: set, P: set, X: set, adj: Dict[str, set],
        results: List[List[str]],
    ):
        if not P and not X:
            results.append(sorted(R))
            return
        pivot_candidates = sorted(P | X)
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
        candidates = sorted(P - adj[pivot])
        for v in candidates:
            self._bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, results)
            P = P - {v}
            X = X | {v}

    # ── Per-layer synthesis (Phase Transition) ───────────────────

    def _chain_bind_handles(self, handles: List[int]) -> int:
        """Chain-bind a list of arena handles: h₀ ⊗ h₁ ⊗ ... ⊗ hₖ."""
        current = handles[0]
        for h in handles[1:]:
            out = self.arena.allocate()
            self.arena.bind(current, h, out)
            current = out
        return current

    def _chain_bind_phase_arrays(self, phase_arrays: List[List[float]]) -> List[float]:
        """Chain-bind phase arrays: phases₀ + phases₁ + ... + phasesₖ."""
        current = phase_arrays[0]
        for pa in phase_arrays[1:]:
            current = bind_phases(current, pa)
        return current

    def synthesize_from_clique(self, clique: List[str]) -> LayeredProjection:
        """Per-layer chain-bind all axioms in a clique, then cross-layer bind.

        S_ast  = A₁_ast  ⊗ A₂_ast  ⊗ ... ⊗ Aₖ_ast
        S_cfg  = A₁_cfg  ⊗ A₂_cfg  ⊗ ... ⊗ Aₖ_cfg
        S_data = A₁_data ⊗ A₂_data ⊗ ... ⊗ Aₖ_data
        S_final = S_ast ⊗ S_cfg ⊗ S_data
        """
        if len(clique) < 2:
            raise ValueError("Synthesis requires at least 2 axioms in a clique.")

        projections = [self.store.axioms[sid].projection for sid in clique]

        # AST layer (always present)
        ast_handles = [p.ast_handle for p in projections]
        ast_phase_arrays = [p.ast_phases for p in projections]
        synth_ast_h = self._chain_bind_handles(ast_handles)
        synth_ast_phases = self._chain_bind_phase_arrays(ast_phase_arrays)

        # CFG layer (may be None for some axioms — skip those)
        cfg_handles = [p.cfg_handle for p in projections if p.cfg_handle is not None]
        cfg_phase_arrays = [p.cfg_phases for p in projections if p.cfg_phases is not None]
        if len(cfg_handles) >= 2:
            synth_cfg_h = self._chain_bind_handles(cfg_handles)
            synth_cfg_phases = self._chain_bind_phase_arrays(cfg_phase_arrays)
        elif len(cfg_handles) == 1:
            synth_cfg_h = cfg_handles[0]
            synth_cfg_phases = cfg_phase_arrays[0]
        else:
            synth_cfg_h = None
            synth_cfg_phases = None

        # Data-dep layer (may be None for some axioms)
        data_handles = [p.data_handle for p in projections if p.data_handle is not None]
        data_phase_arrays = [p.data_phases for p in projections if p.data_phases is not None]
        if len(data_handles) >= 2:
            synth_data_h = self._chain_bind_handles(data_handles)
            synth_data_phases = self._chain_bind_phase_arrays(data_phase_arrays)
        elif len(data_handles) == 1:
            synth_data_h = data_handles[0]
            synth_data_phases = data_phase_arrays[0]
        else:
            synth_data_h = None
            synth_data_phases = None

        # Cross-layer bind → final
        result_h = synth_ast_h
        if synth_cfg_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, synth_cfg_h, merged)
            result_h = merged
        if synth_data_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, synth_data_h, merged)
            result_h = merged

        projection = LayeredProjection(
            final_handle=result_h,
            ast_handle=synth_ast_h,
            cfg_handle=synth_cfg_h,
            data_handle=synth_data_h,
            ast_phases=synth_ast_phases,
            cfg_phases=synth_cfg_phases,
            data_phases=synth_data_phases,
        )
        self._synthesis_log.append((list(clique), projection))
        return projection

    def synthesize_all(
        self, min_clique_size: int = 2
    ) -> List[Tuple[List[str], LayeredProjection]]:
        """Run full synthesis pipeline.
        Returns list of (clique_members, LayeredProjection).
        """
        self.compute_resonance()
        cliques = self.extract_cliques(min_size=min_clique_size)

        results = []
        for clique in cliques:
            synth_proj = self.synthesize_from_clique(clique)
            results.append((clique, synth_proj))

        return results

    def get_synthesis_log(self) -> List[Tuple[List[str], LayeredProjection]]:
        return list(self._synthesis_log)

    # ── Diagnostics ──────────────────────────────────────────────

    def verify_quasi_orthogonality(
        self, synth_handle: int, clique: List[str]
    ) -> Dict[str, float]:
        return {
            sid: self.arena.compute_correlation(synth_handle, self.store.axioms[sid].handle)
            for sid in clique
        }
