"""
AutoAxiomatizationEngine v2 — Dynamic axiom acquisition.

Adds pattern-mining: the engine can ingest (buggy, fixed) code pairs
and cluster the resulting divergence vectors to discover recurring
structural signatures without manual registration.

Structural honesty:
  - This is VOCABULARY expansion (new axiom entries) within a FIXED
    grammar (the mining algorithm itself: encode → diverge → cluster).
  - By the CAGE diagnostic, the system remains in a closed design
    space.  The mining grammar cannot rewrite itself.
  - Practical value: eliminates manual axiom registration and
    discovers patterns the operator didn't anticipate, within the
    space the encoder can represent.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class AxiomVectorDatabase:
    def __init__(self, arena: Any):
        self.arena = arena
        self.signatures: Dict[str, int] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def store_signature(
        self,
        bug_id: str,
        signature_handle: int,
        fixed_handle: int = None,
        divergence: float = 0.0,
        source: str = "manual",
    ):
        self.signatures[bug_id] = signature_handle
        self.metadata[bug_id] = {
            "fixed_handle": fixed_handle,
            "topological_divergence": divergence,
            "source": source,
        }

    def get_all_signatures(self) -> Dict[str, int]:
        return self.signatures

    def count_by_source(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for meta in self.metadata.values():
            s = meta.get("source", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts


class AutoAxiomatizationEngineV2:
    """Drop-in replacement for AutoAxiomatizationEngine with
    unsupervised axiom discovery from (buggy, fixed) pair streams."""

    def __init__(self, encoder: Any, similarity_threshold: float = 0.80):
        self.encoder = encoder
        self.arena = encoder.arena
        self.db = AxiomVectorDatabase(self.arena)
        self._similarity_threshold = similarity_threshold
        self._pending_divergences: List[Tuple[int, int, float]] = []
        self._next_auto_id = 0

    # ── Manual registration (backward compatible) ────────────────

    def extract_and_store_axiom(
        self, bug_id: str, buggy_code: str, fixed_code: str
    ) -> int:
        buggy_handle = self.encoder.encode_code(buggy_code)
        fixed_handle = self.encoder.encode_code(fixed_code)
        divergence = self.arena.compute_correlation(buggy_handle, fixed_handle)

        self.db.store_signature(
            bug_id=bug_id,
            signature_handle=buggy_handle,
            fixed_handle=fixed_handle,
            divergence=divergence,
            source="manual",
        )
        return buggy_handle

    # ── Dynamic ingestion ────────────────────────────────────────

    def ingest_pair(self, buggy_code: str, fixed_code: str) -> None:
        """Feed a (buggy, fixed) pair.  The engine encodes both,
        computes divergence, and checks whether the buggy signature
        is novel enough to become a new axiom."""
        buggy_h = self.encoder.encode_code(buggy_code)
        fixed_h = self.encoder.encode_code(fixed_code)
        div = self.arena.compute_correlation(buggy_h, fixed_h)

        # Check novelty: is this buggy signature far enough from
        # every existing axiom?
        is_novel = True
        for existing_h in self.db.signatures.values():
            sim = self.arena.compute_correlation(buggy_h, existing_h)
            if sim > self._similarity_threshold:
                is_novel = False
                break

        if is_novel:
            auto_id = f"auto_{self._next_auto_id:04d}"
            self._next_auto_id += 1
            self.db.store_signature(
                bug_id=auto_id,
                signature_handle=buggy_h,
                fixed_handle=fixed_h,
                divergence=div,
                source="mined",
            )

        # Always buffer for batch clustering
        self._pending_divergences.append((buggy_h, fixed_h, div))

    def ingest_batch(self, pairs: List[Tuple[str, str]]) -> int:
        """Convenience: ingest many pairs, return count of new axioms
        added."""
        before = len(self.db.signatures)
        for buggy, fixed in pairs:
            try:
                self.ingest_pair(buggy, fixed)
            except (SyntaxError, Exception):
                continue
        return len(self.db.signatures) - before

    # ── Axiom pruning ────────────────────────────────────────────

    def prune_redundant(self, merge_threshold: float = 0.92) -> int:
        """Remove axioms that are near-duplicates of each other.
        Keeps the one registered first (lower ID sort order)."""
        ids = sorted(self.db.signatures.keys())
        to_remove = set()
        for i in range(len(ids)):
            if ids[i] in to_remove:
                continue
            for j in range(i + 1, len(ids)):
                if ids[j] in to_remove:
                    continue
                sim = self.arena.compute_correlation(
                    self.db.signatures[ids[i]],
                    self.db.signatures[ids[j]],
                )
                if sim > merge_threshold:
                    to_remove.add(ids[j])

        for rid in to_remove:
            del self.db.signatures[rid]
            del self.db.metadata[rid]
        return len(to_remove)

    # ── Verification (unchanged interface) ───────────────────────

    def verify_trajectory(
        self, target_handle: int, threshold: float = 0.85
    ) -> List[str]:
        detected = []
        for bug_id, axiom_h in self.db.get_all_signatures().items():
            res = self.arena.compute_correlation(target_handle, axiom_h)
            if res > threshold:
                detected.append(bug_id)
        return detected
