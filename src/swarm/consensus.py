"""Resonance Clique Consensus — Algebraic consensus from candidate phase arrays.

Given N candidate vectors from M agents, find the subset that mutually resonate
above threshold, bundled into a centroid. Uses correlate_matrix (single FFI call)
and Bron-Kerbosch clique extraction.

No voting. No debate. Pure algebraic composition.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class ConsensusResult:
    """Result of swarm consensus computation."""
    centroid_phases: List[float]          # Bundled centroid of winning clique
    clique_member_indices: List[int]      # Which candidates are in the clique
    clique_agent_ids: List[int]           # Which agents contributed
    mean_resonance: float                 # Mean pairwise rho within clique
    clique_size: int                      # Number of members


def _bron_kerbosch(
    R: Set[int],
    P: Set[int],
    X: Set[int],
    adj: Dict[int, Set[int]],
    results: List[List[int]],
) -> None:
    """Bron-Kerbosch maximal clique enumeration (deterministic)."""
    if not P and not X:
        results.append(sorted(R))
        return
    pivot_candidates = sorted(P | X)
    if not pivot_candidates:
        return
    pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
    candidates = sorted(P - adj[pivot])
    for v in candidates:
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, results)
        P = P - {v}
        X = X | {v}


def compute_swarm_consensus(
    arena: Any,
    candidate_phases: List[List[float]],
    candidate_metadata: List[Dict[str, Any]],
    consensus_threshold: float = 0.85,
    min_clique_size: int = 2,
) -> Optional[ConsensusResult]:
    """Compute the algebraic consensus from candidate phase arrays.

    Algorithm:
      1. Inject all candidate phases into a temporary arena
      2. Call arena.correlate_matrix() -- single FFI call for NxN matrix
      3. Build adjacency graph where edge exists iff rho > consensus_threshold
      4. Run Bron-Kerbosch to find maximal cliques
      5. Select the largest clique (ties broken by mean resonance)
      6. Bundle all clique members into centroid vector
      7. Return centroid phases + clique membership

    Args:
        arena: A temporary FhrrArena for consensus computation.
               (NOT any agent's arena -- create a fresh one)
        candidate_phases: List of phase arrays from all agents.
        candidate_metadata: Parallel list of metadata dicts.
        consensus_threshold: Minimum rho for clique adjacency.
        min_clique_size: Minimum clique size for valid consensus.

    Returns:
        ConsensusResult or None if no clique meets the threshold.
    """
    n = len(candidate_phases)
    if n < min_clique_size:
        return None

    # Step 1: Inject all candidates into the temporary arena
    handles = []
    for phases in candidate_phases:
        h = arena.allocate()
        arena.inject_phases(h, phases)
        handles.append(h)

    # Step 2: Compute NxN correlation matrix via single FFI call
    # correlate_matrix returns flat upper triangle: [(0,1), (0,2), ..., (n-2,n-1)]
    if n >= 2:
        flat = arena.correlate_matrix(handles)
    else:
        flat = []

    # Unpack into full matrix
    corr_matrix: Dict[tuple, float] = {}
    k = 0
    for i in range(n):
        corr_matrix[(i, i)] = arena.compute_correlation(handles[i], handles[i])
        for j in range(i + 1, n):
            corr = flat[k]
            k += 1
            corr_matrix[(i, j)] = corr
            corr_matrix[(j, i)] = corr

    # Step 3: Build adjacency graph
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and corr_matrix.get((i, j), 0.0) > consensus_threshold:
                adj[i].add(j)

    # Step 4: Bron-Kerbosch clique extraction
    cliques: List[List[int]] = []
    _bron_kerbosch(set(), set(range(n)), set(), adj, cliques)

    # Filter by minimum size
    valid_cliques = [c for c in cliques if len(c) >= min_clique_size]
    if not valid_cliques:
        return None

    # Step 5: Select best clique (largest, then highest mean resonance)
    def clique_score(clique: List[int]) -> tuple:
        pairs = [(clique[i], clique[j])
                 for i in range(len(clique)) for j in range(i + 1, len(clique))]
        mean_res = sum(corr_matrix[(a, b)] for a, b in pairs) / max(len(pairs), 1)
        return (len(clique), mean_res)

    best_clique = max(valid_cliques, key=clique_score)
    _, mean_resonance = clique_score(best_clique)

    # Step 6: Bundle clique members into centroid
    clique_handles = [handles[i] for i in best_clique]
    centroid_h = arena.allocate()
    arena.bundle(clique_handles, centroid_h)
    centroid_phases = list(arena.extract_phases(centroid_h))

    # Extract agent IDs from metadata
    clique_agent_ids = []
    for idx in best_clique:
        if idx < len(candidate_metadata):
            agent_id = candidate_metadata[idx].get("agent_id", -1)
            clique_agent_ids.append(agent_id)
        else:
            clique_agent_ids.append(-1)

    return ConsensusResult(
        centroid_phases=centroid_phases,
        clique_member_indices=best_clique,
        clique_agent_ids=clique_agent_ids,
        mean_resonance=mean_resonance,
        clique_size=len(best_clique),
    )
