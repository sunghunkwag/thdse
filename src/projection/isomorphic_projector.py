"""
Isomorphic Projector — Maps a multi-layer directed graph (AST + CFG + Data-Dep)
into a single holographic FHRR hypervector via the hdc_core VSA arena.

Mathematical contract:
  - Control-flow sequences  →  bind (⊗) chain  (preserves ordering)
  - Data-dependency sets    →  bundle (⊕)       (preserves membership, order-invariant)
  - AST structural identity →  bind (type ⊗ position ⊗ CA-state)

The output is a single arena handle representing the entire code's topological
fingerprint in the complex hyperdimensional space.
"""

import hashlib
import math
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from src.topology.multi_layer_builder import MultiLayerGraphBuilder


class IsomorphicProjector:
    """Projects a multi-layer code graph into a single FHRR hypervector."""

    def __init__(self, arena: Any, dimension: int, ca_steps: int = 5):
        self.arena = arena
        self.dimension = dimension
        self.builder = MultiLayerGraphBuilder(steps=ca_steps)
        self._seed_cache: Dict[str, int] = {}

    # ── Deterministic phase generation ───────────────────────────

    def _deterministic_phases(self, seed: int) -> List[float]:
        """Generate a repeatable phase vector from a 32-bit seed.
        Uses md5-seeded linear congruential expansion — no PRNG state."""
        phases = []
        state = seed & 0xFFFFFFFF
        for _ in range(self.dimension):
            # Knuth LCG with md5 mixing every 64 steps for diffusion
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
            phases.append(phase)
        return phases

    def _get_or_create_handle(self, cache_key: str) -> int:
        """Allocate and inject a deterministic hypervector, cached by key."""
        if cache_key not in self._seed_cache:
            seed = int(hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:8], 16)
            handle = self.arena.allocate()
            self.arena.inject_phases(handle, self._deterministic_phases(seed))
            self._seed_cache[cache_key] = handle
        return self._seed_cache[cache_key]

    # ── Node-level projection ────────────────────────────────────

    def _project_node(self, node_id: int, data: dict) -> int:
        """Project a single graph node → handle = type ⊗ pos ⊗ ca_state."""
        node_type = data["type"]
        depth = data.get("depth", 0)
        sibling_idx = data.get("sibling_index", 0)

        type_h = self._get_or_create_handle(f"type:{node_type}")
        pos_h = self._get_or_create_handle(f"pos:{depth}:{sibling_idx}")

        # CA state → uniform phase injection
        ca_phase = float(np.angle(data["state"]))
        state_h = self.arena.allocate()
        self.arena.inject_phases(state_h, [ca_phase] * self.dimension)

        # type ⊗ pos
        tmp = self.arena.allocate()
        self.arena.bind(type_h, pos_h, tmp)

        # (type ⊗ pos) ⊗ ca_state
        node_h = self.arena.allocate()
        self.arena.bind(tmp, state_h, node_h)
        return node_h

    # ── Layer-specific edge projections ──────────────────────────

    def _project_cfg_chains(self, G: nx.DiGraph, node_handles: Dict[int, int]) -> List[int]:
        """CFG edges encode ordering → chain-bind along each maximal CFG path.

        For a path [n0 → n1 → n2], the encoding is:
            role_cfg ⊗ H(n0) ⊗ H(n1) ⊗ H(n2)
        This preserves the sequential ordering of control flow.
        """
        role_cfg = self._get_or_create_handle("role:cfg")

        # Extract CFG-only subgraph
        cfg_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("layer") == MultiLayerGraphBuilder.EDGE_CFG
        ]
        cfg_sub = nx.DiGraph(cfg_edges)

        # Find maximal paths (start from nodes with in-degree 0 in CFG subgraph)
        chain_handles = []
        sources = [n for n in cfg_sub.nodes() if cfg_sub.in_degree(n) == 0]
        if not sources:
            sources = list(cfg_sub.nodes())[:1]  # cycle: pick arbitrary start

        visited_edges = set()
        for src in sources:
            # Walk forward greedily
            path = [src]
            current = src
            while True:
                succs = [s for s in cfg_sub.successors(current)
                         if (current, s) not in visited_edges]
                if not succs:
                    break
                nxt = succs[0]
                visited_edges.add((current, nxt))
                path.append(nxt)
                current = nxt

            if len(path) < 2:
                continue

            # Chain-bind: role ⊗ h0 ⊗ h1 ⊗ ... ⊗ hN
            chain_h = self.arena.allocate()
            first_node = path[0]
            if first_node in node_handles:
                self.arena.bind(role_cfg, node_handles[first_node], chain_h)
            else:
                continue

            for step_node in path[1:]:
                if step_node not in node_handles:
                    continue
                next_chain = self.arena.allocate()
                self.arena.bind(chain_h, node_handles[step_node], next_chain)
                chain_h = next_chain

            chain_handles.append(chain_h)

        return chain_handles

    def _project_data_dep_sets(self, G: nx.DiGraph, node_handles: Dict[int, int]) -> List[int]:
        """Data-dep edges encode unordered def-use relationships → bundle per node.

        For each node with data-dep neighbors {d1, d2, ...}:
            role_data ⊗ bundle(H(d1), H(d2), ...)
        """
        role_data = self._get_or_create_handle("role:data_dep")

        dep_handles = []
        for node in G.nodes():
            neighbors = set()
            for _, v, d in G.out_edges(node, data=True):
                if d.get("layer") == MultiLayerGraphBuilder.EDGE_DATA:
                    neighbors.add(v)
            for u, _, d in G.in_edges(node, data=True):
                if d.get("layer") == MultiLayerGraphBuilder.EDGE_DATA:
                    neighbors.add(u)

            valid = [node_handles[n] for n in neighbors if n in node_handles]
            if not valid:
                continue

            dep_bundle = self.arena.allocate()
            self.arena.bundle(valid, dep_bundle)

            bound = self.arena.allocate()
            self.arena.bind(role_data, dep_bundle, bound)
            dep_handles.append(bound)

        return dep_handles

    # ── Full projection pipeline ─────────────────────────────────

    def project(self, code: str) -> int:
        """Map source code → single holographic hypervector handle.

        Returns an arena handle encoding the full multi-layer topology.
        """
        G = self.builder.build(code)
        G = self.builder.ca.evolve(G)

        # Phase 1: project every node independently
        node_handles: Dict[int, int] = {}
        for node in G.nodes():
            node_handles[node] = self._project_node(node, G.nodes[node])

        # Phase 2: AST structural layer — bundle all node vectors
        ast_bundle_h = self.arena.allocate()
        all_node_h = list(node_handles.values())
        if all_node_h:
            self.arena.bundle(all_node_h, ast_bundle_h)

        # Phase 3: CFG sequential layer — chain-bind paths, then bundle
        cfg_chains = self._project_cfg_chains(G, node_handles)
        if cfg_chains:
            cfg_layer_h = self.arena.allocate()
            self.arena.bundle(cfg_chains, cfg_layer_h)
        else:
            cfg_layer_h = None

        # Phase 4: Data-dep set layer — bundle all dep projections
        dep_projections = self._project_data_dep_sets(G, node_handles)
        if dep_projections:
            data_layer_h = self.arena.allocate()
            self.arena.bundle(dep_projections, data_layer_h)
        else:
            data_layer_h = None

        # Phase 5: Merge layers via bind (⊗) to produce final hologram
        # code_vector = ast_layer ⊗ cfg_layer ⊗ data_layer
        result_h = ast_bundle_h
        if cfg_layer_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, cfg_layer_h, merged)
            result_h = merged
        if data_layer_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, data_layer_h, merged)
            result_h = merged

        return result_h

    def project_graph(self, G: nx.DiGraph) -> int:
        """Project a pre-built, pre-evolved graph (for callers that already
        have the graph object). Skips build + evolve."""
        node_handles: Dict[int, int] = {}
        for node in G.nodes():
            node_handles[node] = self._project_node(node, G.nodes[node])

        ast_bundle_h = self.arena.allocate()
        all_node_h = list(node_handles.values())
        if all_node_h:
            self.arena.bundle(all_node_h, ast_bundle_h)

        cfg_chains = self._project_cfg_chains(G, node_handles)
        cfg_layer_h = None
        if cfg_chains:
            cfg_layer_h = self.arena.allocate()
            self.arena.bundle(cfg_chains, cfg_layer_h)

        dep_projections = self._project_data_dep_sets(G, node_handles)
        data_layer_h = None
        if dep_projections:
            data_layer_h = self.arena.allocate()
            self.arena.bundle(dep_projections, data_layer_h)

        result_h = ast_bundle_h
        if cfg_layer_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, cfg_layer_h, merged)
            result_h = merged
        if data_layer_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, data_layer_h, merged)
            result_h = merged

        return result_h
