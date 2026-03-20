"""
TopologicalEncoder v2 — Multi-layer graph encoding.

Adds control-flow edges (CFG) and data-dependency edges alongside
the existing AST parent-child edges. This reduces semantic loss in
the hypervector mapping but does NOT achieve "perfect isomorphism"
(Rice's theorem: general semantic equivalence is undecidable).
"""

import ast
import hashlib
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Tuple

from src.topology.ast_graph_ca import TopologicalASTGraphCA


# ── Multi-layer graph builder ────────────────────────────────────

class MultiLayerGraphBuilder:
    """Builds a DiGraph with three edge types: ast, cfg, data_dep."""

    EDGE_AST = "ast"
    EDGE_CFG = "cfg"
    EDGE_DATA = "data_dep"

    def __init__(self, steps: int = 5):
        self.ca = TopologicalASTGraphCA(steps=steps)

    def build(self, code: str) -> nx.DiGraph:
        tree = ast.parse(code)

        # Layer 1: AST edges (reuse existing CA builder)
        G = self.ca.code_to_graph(tree=tree)
        for u, v in G.edges:
            G.edges[u, v]["layer"] = self.EDGE_AST

        # Layer 2: intra-procedural CFG edges
        self._add_cfg_edges(G, tree)

        # Layer 3: def-use data dependency edges
        self._add_data_dep_edges(G, tree)

        return G

    # ── CFG edges ────────────────────────────────────────────────

    def _add_cfg_edges(self, G: nx.DiGraph, tree: ast.Module):
        """Sequential control flow between top-level statements
        and within function bodies.  Branch targets for If/While."""
        for node in ast.walk(tree):
            stmts = self._get_body(node)
            if stmts is None:
                continue
            for i in range(len(stmts) - 1):
                src, dst = id(stmts[i]), id(stmts[i + 1])
                if src in G.nodes and dst in G.nodes:
                    G.add_edge(src, dst, layer=self.EDGE_CFG)

            # branch edges
            for stmt in stmts:
                if isinstance(stmt, ast.If):
                    self._add_branch_cfg(G, stmt)
                elif isinstance(stmt, (ast.While, ast.For)):
                    self._add_loop_cfg(G, stmt)

    def _add_branch_cfg(self, G: nx.DiGraph, node: ast.If):
        if_id = id(node)
        if node.body:
            G.add_edge(if_id, id(node.body[0]), layer=self.EDGE_CFG)
        if node.orelse:
            G.add_edge(if_id, id(node.orelse[0]), layer=self.EDGE_CFG)

    def _add_loop_cfg(self, G: nx.DiGraph, node):
        loop_id = id(node)
        if node.body:
            # entry → first body stmt
            G.add_edge(loop_id, id(node.body[0]), layer=self.EDGE_CFG)
            # last body stmt → loop header (back-edge)
            G.add_edge(id(node.body[-1]), loop_id, layer=self.EDGE_CFG)

    @staticmethod
    def _get_body(node) -> list | None:
        if hasattr(node, "body") and isinstance(node.body, list):
            return node.body
        return None

    # ── Data-dependency edges ────────────────────────────────────

    def _add_data_dep_edges(self, G: nx.DiGraph, tree: ast.Module):
        """Lightweight def-use chains: for each Name(Load), draw an
        edge from the most recent Name(Store) of the same identifier
        within the same scope (function body or module)."""
        for node in ast.walk(tree):
            stmts = self._get_body(node)
            if stmts is None:
                continue
            self._scan_defuse(G, stmts)

    def _scan_defuse(self, G: nx.DiGraph, stmts: list):
        last_def: Dict[str, int] = {}   # var_name → node id of defining stmt
        for stmt in stmts:
            # collect uses first (read-before-write within the stmt)
            uses = set()
            defs = set()
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Load):
                        uses.add(child.id)
                    elif isinstance(child.ctx, (ast.Store, ast.Del)):
                        defs.add(child.id)

            stmt_id = id(stmt)
            for var in uses:
                if var in last_def and last_def[var] in G.nodes and stmt_id in G.nodes:
                    G.add_edge(last_def[var], stmt_id, layer=self.EDGE_DATA)

            for var in defs:
                last_def[var] = stmt_id


# ── Encoder ──────────────────────────────────────────────────────

# Edge-layer weight factors for binding
_LAYER_WEIGHTS = {
    MultiLayerGraphBuilder.EDGE_AST: 1.0,
    MultiLayerGraphBuilder.EDGE_CFG: 0.6,
    MultiLayerGraphBuilder.EDGE_DATA: 0.8,
}


class TopologicalEncoderV2:
    """Encodes code into FHRR hypervectors using AST + CFG + data-dep
    multi-layer graph.  Drop-in replacement for TopologicalEncoder."""

    def __init__(self, arena: Any, dimension: int):
        self.arena = arena
        self.dimension = dimension
        self.graph_builder = MultiLayerGraphBuilder(steps=5)
        self.ca = TopologicalASTGraphCA(steps=5)
        self.seed_cache: Dict[str, int] = {}

    # ── phase generation (unchanged) ─────────────────────────────

    def _generate_base_phases(self, seed: int) -> List[float]:
        rng = np.random.RandomState(seed & 0xFFFFFFFF)
        return rng.uniform(-np.pi, np.pi, self.dimension).tolist()

    def get_type_handle(self, node_type: str) -> int:
        if node_type not in self.seed_cache:
            type_hash = int(hashlib.md5(node_type.encode('utf-8')).hexdigest()[:8], 16)
            handle = self.arena.allocate()
            phases = self._generate_base_phases(type_hash)
            self.arena.inject_phases(handle, phases)
            self.seed_cache[node_type] = handle
        return self.seed_cache[node_type]

    def get_pos_handle(self, depth: int, sibling_index: int) -> int:
        fractional_pos = hash(depth) + 0.1 * sibling_index
        pos_key = f"__pos_{fractional_pos}__"
        if pos_key not in self.seed_cache:
            handle = self.arena.allocate()
            seed_int = int(hashlib.md5(str(fractional_pos).encode('utf-8')).hexdigest()[:8], 16)
            phases = self._generate_base_phases(seed_int)
            self.arena.inject_phases(handle, phases)
            self.seed_cache[pos_key] = handle
        return self.seed_cache[pos_key]

    def _get_edge_layer_handle(self, layer: str) -> int:
        """Distinct role vector per edge layer."""
        cache_key = f"__layer_{layer}__"
        if cache_key not in self.seed_cache:
            handle = self.arena.allocate()
            seed_int = int(hashlib.md5(layer.encode('utf-8')).hexdigest()[:8], 16)
            phases = self._generate_base_phases(seed_int)
            self.arena.inject_phases(handle, phases)
            self.seed_cache[cache_key] = handle
        return self.seed_cache[cache_key]

    # ── encoding pipeline ────────────────────────────────────────

    def encode_code(self, code: str) -> int:
        G = self.graph_builder.build(code)
        G = self.ca.evolve(G)

        node_handles = []
        for node in G.nodes:
            data = G.nodes[node]
            node_type = data['type']
            depth = data.get('depth', 0)
            sibling_index = data.get('sibling_index', 0)

            # state-phase handle
            state_phase = float(np.angle(data['state']))
            state_handle = self.arena.allocate()
            self.arena.inject_phases(state_handle, [state_phase] * self.dimension)

            # type ⊗ position ⊗ state
            type_h = self.get_type_handle(node_type)
            pos_h = self.get_pos_handle(depth, sibling_index)

            tmp1 = self.arena.allocate()
            self.arena.bind(type_h, pos_h, tmp1)

            node_hv = self.arena.allocate()
            self.arena.bind(tmp1, state_handle, node_hv)

            # neighbour context: bind each edge layer's neighbours
            edge_ctx_handles = []
            for layer_name, weight in _LAYER_WEIGHTS.items():
                neighbours = [
                    v for _, v, d in G.out_edges(node, data=True)
                    if d.get("layer") == layer_name
                ] + [
                    u for u, _, d in G.in_edges(node, data=True)
                    if d.get("layer") == layer_name
                ]
                if not neighbours:
                    continue
                layer_role = self._get_edge_layer_handle(layer_name)
                for nb in neighbours:
                    nb_data = G.nodes[nb]
                    nb_type_h = self.get_type_handle(nb_data['type'])
                    ctx_h = self.arena.allocate()
                    self.arena.bind(layer_role, nb_type_h, ctx_h)
                    edge_ctx_handles.append(ctx_h)

            if edge_ctx_handles:
                ctx_bundle = self.arena.allocate()
                self.arena.bundle(edge_ctx_handles, ctx_bundle)
                enriched = self.arena.allocate()
                self.arena.bind(node_hv, ctx_bundle, enriched)
                node_handles.append(enriched)
            else:
                node_handles.append(node_hv)

        graph_handle = self.arena.allocate()
        self.arena.bundle(node_handles, graph_handle)
        return graph_handle
