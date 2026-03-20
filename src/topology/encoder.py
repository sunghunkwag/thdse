import hashlib
import numpy as np
from typing import List, Dict, Any

from src.topology.ast_graph_ca import TopologicalASTGraphCA

class TopologicalEncoder:
    def __init__(self, arena: Any, dimension: int):
        self.arena = arena
        self.dimension = dimension
        self.ca = TopologicalASTGraphCA(steps=5)
        self.seed_cache: Dict[str, int] = {}

    def _generate_base_phases(self, seed: int) -> List[float]:
        rng = np.random.RandomState(seed & 0xFFFFFFFF)
        phases = rng.uniform(-np.pi, np.pi, self.dimension)
        return phases.tolist()

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

    def encode_code(self, code: str) -> int:
        G = self.ca.code_to_graph(code)
        G = self.ca.evolve(G)
        
        node_handles = []
        for node in G.nodes:
            node_data = G.nodes[node]
            node_type = node_data['type']
            depth = node_data.get('depth', 0)
            sibling_index = node_data.get('sibling_index', 0)
            
            state_phase = float(np.angle(node_data['state']))
            state_phases = [state_phase] * self.dimension
            
            state_handle = self.arena.allocate()
            self.arena.inject_phases(state_handle, state_phases)
            
            type_handle = self.get_type_handle(node_type)
            pos_handle = self.get_pos_handle(depth, sibling_index)
            
            temp_bind_1 = self.arena.allocate()
            self.arena.bind(type_handle, pos_handle, temp_bind_1)
            
            node_hv_handle = self.arena.allocate()
            self.arena.bind(temp_bind_1, state_handle, node_hv_handle)
            
            node_handles.append(node_hv_handle)
            
        graph_target_handle = self.arena.allocate()
        self.arena.bundle(node_handles, graph_target_handle)
        
        return graph_target_handle
