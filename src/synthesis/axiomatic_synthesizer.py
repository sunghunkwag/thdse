"""
Axiomatic Synthesizer — Deterministic phase-transition engine that accumulates
'Axiom Vectors' from stable codebases and synthesizes novel structural
hypervectors through mathematically provable binding operations.

No randomness. No LLMs. Every operation is a deterministic function of the
input corpus and the VSA algebra (FHRR bind/bundle/correlate).

Mathematical foundation:
  - Each stable codebase c_i is projected to an axiom vector A_i ∈ S^{d-1} (unit torus).
  - Structural resonance between axioms: ρ(A_i, A_j) = correlate(A_i, A_j).
  - Synthesis: given a resonance-connected clique {A_1, ..., A_k} where
    ∀i,j: ρ(A_i, A_j) > τ, the synthesized vector is:
        S = A_1 ⊗ A_2 ⊗ ... ⊗ A_k  (chain bind)
    This produces a vector that is quasi-orthogonal to each individual axiom
    yet encodes their shared structural invariants via phase superposition.
"""

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.projection.isomorphic_projector import IsomorphicProjector


@dataclass
class Axiom:
    """An axiom vector with provenance metadata."""
    handle: int
    source_id: str
    resonance_profile: Dict[str, float] = field(default_factory=dict)


class AxiomStore:
    """Flat, indexed store of axiom vectors within a shared arena."""

    def __init__(self, arena: Any):
        self.arena = arena
        self.axioms: Dict[str, Axiom] = {}  # source_id → Axiom

    def register(self, source_id: str, handle: int) -> Axiom:
        axiom = Axiom(handle=handle, source_id=source_id)
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

    This is a deterministic, symmetric matrix:
        R[i,j] = arena.compute_correlation(A_i.handle, A_j.handle)
    """

    def __init__(self, arena: Any, store: AxiomStore):
        self.arena = arena
        self.store = store
        self._matrix: Dict[Tuple[str, str], float] = {}

    def compute_full(self) -> Dict[Tuple[str, str], float]:
        """(Re)compute the full pairwise resonance matrix."""
        self._matrix.clear()
        ids = self.store.all_ids()
        for i, id_a in enumerate(ids):
            ax_a = self.store.axioms[id_a]
            for id_b in ids[i:]:
                ax_b = self.store.axioms[id_b]
                corr = self.arena.compute_correlation(ax_a.handle, ax_b.handle)
                self._matrix[(id_a, id_b)] = corr
                self._matrix[(id_b, id_a)] = corr
                # Update per-axiom resonance profiles
                ax_a.resonance_profile[id_b] = corr
                ax_b.resonance_profile[id_a] = corr
        return self._matrix

    def get(self, id_a: str, id_b: str) -> float:
        return self._matrix.get((id_a, id_b), 0.0)


class AxiomaticSynthesizer:
    """Ingests stable codebases, discovers resonance cliques, and synthesizes
    novel structural hypervectors through deterministic binding.

    Pipeline:
      1. Ingest: code → IsomorphicProjector → axiom handle → AxiomStore
      2. Resonate: compute pairwise correlation matrix
      3. Clique extraction: find all maximal sets where ∀ pairs exceed threshold τ
      4. Synthesize: chain-bind each clique → novel hypervector
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
        self._synthesis_log: List[Tuple[List[str], int]] = []

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, source_id: str, code: str) -> Axiom:
        """Project a codebase and register it as an axiom."""
        handle = self.projector.project(code)
        return self.store.register(source_id, handle)

    def ingest_batch(self, corpus: Dict[str, str]) -> int:
        """Ingest multiple codebases. Returns count of axioms registered."""
        for source_id in sorted(corpus.keys()):
            self.ingest(source_id, corpus[source_id])
        return self.store.count()

    # ── Resonance analysis ───────────────────────────────────────

    def compute_resonance(self) -> Dict[Tuple[str, str], float]:
        """Compute full pairwise resonance matrix."""
        return self.resonance.compute_full()

    # ── Clique extraction (Bron-Kerbosch, deterministic) ─────────

    def extract_cliques(self, min_size: int = 2) -> List[List[str]]:
        """Find all maximal cliques in the resonance graph where every
        pair of axioms exceeds the threshold τ.

        Uses iterative Bron-Kerbosch with pivot (deterministic: sorted inputs).
        """
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
        self,
        R: set,
        P: set,
        X: set,
        adj: Dict[str, set],
        results: List[List[str]],
    ):
        """Bron-Kerbosch with pivot selection (deterministic: lexicographic)."""
        if not P and not X:
            results.append(sorted(R))
            return
        # Pivot: choose vertex in P ∪ X with max connections to P
        pivot_candidates = sorted(P | X)
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
        candidates = sorted(P - adj[pivot])
        for v in candidates:
            self._bron_kerbosch(
                R | {v},
                P & adj[v],
                X & adj[v],
                adj,
                results,
            )
            P = P - {v}
            X = X | {v}

    # ── Synthesis (Phase Transition) ─────────────────────────────

    def synthesize_from_clique(self, clique: List[str]) -> int:
        """Chain-bind all axioms in a clique to produce a novel hypervector.

        Given axioms [A_1, A_2, ..., A_k]:
            S = A_1 ⊗ A_2 ⊗ ... ⊗ A_k

        The result S is quasi-orthogonal to each individual A_i (by the
        concentration of measure in high-dimensional FHRR spaces) yet
        encodes the compositional intersection of their phase structures.
        """
        if len(clique) < 2:
            raise ValueError("Synthesis requires at least 2 axioms in a clique.")

        handles = [self.store.axioms[sid].handle for sid in clique]

        # Chain-bind: h0 ⊗ h1 → tmp, tmp ⊗ h2 → tmp2, ...
        current = handles[0]
        for h in handles[1:]:
            out = self.arena.allocate()
            self.arena.bind(current, h, out)
            current = out

        self._synthesis_log.append((list(clique), current))
        return current

    def synthesize_all(self, min_clique_size: int = 2) -> List[Tuple[List[str], int]]:
        """Run full synthesis pipeline:
        1. Compute resonance matrix
        2. Extract maximal cliques
        3. Chain-bind each clique

        Returns list of (clique_members, synthesized_handle).
        """
        self.compute_resonance()
        cliques = self.extract_cliques(min_size=min_clique_size)

        results = []
        for clique in cliques:
            synth_h = self.synthesize_from_clique(clique)
            results.append((clique, synth_h))

        return results

    def get_synthesis_log(self) -> List[Tuple[List[str], int]]:
        return list(self._synthesis_log)

    # ── Diagnostics ──────────────────────────────────────────────

    def verify_quasi_orthogonality(self, synth_handle: int, clique: List[str]) -> Dict[str, float]:
        """Verify that the synthesized vector is quasi-orthogonal to its source axioms.
        Returns {source_id: correlation} for each axiom in the clique."""
        return {
            sid: self.arena.compute_correlation(synth_handle, self.store.axioms[sid].handle)
            for sid in clique
        }
