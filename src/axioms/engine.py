from typing import Dict, List, Any
from src.topology.encoder import TopologicalEncoder

class AxiomVectorDatabase:
    def __init__(self, arena: Any):
        self.arena = arena
        self.signatures: Dict[str, int] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def store_signature(self, bug_id: str, signature_handle: int, fixed_handle: int = None, divergence: float = 0.0):
        self.signatures[bug_id] = signature_handle
        self.metadata[bug_id] = {
            "fixed_handle": fixed_handle,
            "topological_divergence": divergence
        }

    def get_all_signatures(self) -> Dict[str, int]:
        return self.signatures

class AutoAxiomatizationEngine:
    def __init__(self, encoder: TopologicalEncoder):
        self.encoder = encoder
        self.arena = encoder.arena
        self.db = AxiomVectorDatabase(self.arena)

    def extract_and_store_axiom(self, bug_id: str, buggy_code: str, fixed_code: str) -> int:
        buggy_handle = self.encoder.encode_code(buggy_code)
        fixed_handle = self.encoder.encode_code(fixed_code)
        
        divergence_resonance = self.arena.compute_correlation(buggy_handle, fixed_handle)
        
        self.db.store_signature(
            bug_id=bug_id,
            signature_handle=buggy_handle,
            fixed_handle=fixed_handle,
            divergence=divergence_resonance
        )
        return buggy_handle

    def verify_trajectory(self, target_handle: int, threshold: float = 0.85) -> List[str]:
        detected_bugs = []
        for bug_id, axiom_handle in self.db.get_all_signatures().items():
            resonance = self.arena.compute_correlation(target_handle, axiom_handle)
            if resonance > threshold:
                detected_bugs.append(bug_id)
        return detected_bugs
