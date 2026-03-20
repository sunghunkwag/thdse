"""
Refactoring Detector — Scans axiom pairs in the corpus to identify
structural duplicates and interface unification candidates.

Detection rules:
  1. Structural Duplicate:
     sim_ast > 0.7 AND sim_cfg > 0.7 AND files in different directories
     → "Consider extracting common code"

  2. Interface Unification Candidate:
     sim_data > 0.8 AND sim_ast < 0.3
     → "Same computation, different expression — unify interface"

Output: structured JSON compatible with the existing report system.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import hdc_core as _hdc_core
except ImportError:
    _hdc_core = None

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
from src.analysis.structural_diff import StructuralDiffEngine, LayerSimilarity


@dataclass
class RefactoringCandidate:
    """A detected refactoring opportunity.

    Attributes:
        kind: Type of refactoring ("structural_duplicate" or "interface_unification").
        file_a: Source ID of the first file.
        file_b: Source ID of the second file.
        sim_ast: AST layer similarity.
        sim_cfg: CFG layer similarity.
        sim_data: Data-dep layer similarity.
        recommendation: Human-readable refactoring recommendation.
    """
    kind: str
    file_a: str
    file_b: str
    sim_ast: float
    sim_cfg: Optional[float]
    sim_data: Optional[float]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "file_a": self.file_a,
            "file_b": self.file_b,
            "sim_ast": round(self.sim_ast, 6),
            "sim_cfg": round(self.sim_cfg, 6) if self.sim_cfg is not None else None,
            "sim_data": round(self.sim_data, 6) if self.sim_data is not None else None,
            "recommendation": self.recommendation,
        }


def _get_directory(source_id: str) -> str:
    """Extract the directory component from a source ID.

    Args:
        source_id: Source identifier (e.g., "project/subdir/file.py").

    Returns:
        Directory portion (e.g., "project/subdir"), or "" if no separator.
    """
    parts = source_id.rsplit("/", 1)
    return parts[0] if len(parts) > 1 else ""


class RefactoringDetector:
    """Scans axiom pairs to detect refactoring opportunities.

    Usage:
        detector = RefactoringDetector(arena, projector, synthesizer)
        candidates = detector.scan()
        report = detector.to_json(candidates)
    """

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        synthesizer: AxiomaticSynthesizer,
        dup_ast_threshold: float = 0.7,
        dup_cfg_threshold: float = 0.7,
        unify_data_threshold: float = 0.8,
        unify_ast_ceiling: float = 0.3,
    ) -> None:
        """Initialize the refactoring detector.

        Args:
            arena: FHRR arena instance.
            projector: IsomorphicProjector.
            synthesizer: AxiomaticSynthesizer with ingested axioms.
            dup_ast_threshold: Min AST similarity for duplicate detection.
            dup_cfg_threshold: Min CFG similarity for duplicate detection.
            unify_data_threshold: Min data-dep similarity for unification.
            unify_ast_ceiling: Max AST similarity for unification candidate.
        """
        self.arena = arena
        self.projector = projector
        self.synthesizer = synthesizer
        self.diff_engine = StructuralDiffEngine(arena, projector)
        self.dup_ast_threshold = dup_ast_threshold
        self.dup_cfg_threshold = dup_cfg_threshold
        self.unify_data_threshold = unify_data_threshold
        self.unify_ast_ceiling = unify_ast_ceiling

    def scan(self) -> List[RefactoringCandidate]:
        """Scan all axiom pairs for refactoring opportunities.

        Returns:
            List of RefactoringCandidate instances, sorted by relevance.
        """
        candidates: List[RefactoringCandidate] = []
        ids = self.synthesizer.store.all_ids()

        for i, id_a in enumerate(ids):
            axiom_a = self.synthesizer.store.get(id_a)
            if axiom_a is None:
                continue

            for id_b in ids[i + 1:]:
                axiom_b = self.synthesizer.store.get(id_b)
                if axiom_b is None:
                    continue

                layer_sim = self.diff_engine.compare_layers(
                    axiom_a.projection, axiom_b.projection,
                )

                # Rule 1: Structural Duplicate
                dir_a = _get_directory(id_a)
                dir_b = _get_directory(id_b)
                is_cross_dir = dir_a != dir_b and dir_a and dir_b

                if (is_cross_dir
                    and abs(layer_sim.sim_ast) > self.dup_ast_threshold
                    and layer_sim.sim_cfg is not None
                    and abs(layer_sim.sim_cfg) > self.dup_cfg_threshold):
                    candidates.append(RefactoringCandidate(
                        kind="structural_duplicate",
                        file_a=id_a,
                        file_b=id_b,
                        sim_ast=layer_sim.sim_ast,
                        sim_cfg=layer_sim.sim_cfg,
                        sim_data=layer_sim.sim_data,
                        recommendation=(
                            f"Structural duplicate detected: AST={layer_sim.sim_ast:.2f}, "
                            f"CFG={layer_sim.sim_cfg:.2f}. "
                            f"Consider extracting common code into a shared module."
                        ),
                    ))

                # Rule 2: Interface Unification Candidate
                if (layer_sim.sim_data is not None
                    and abs(layer_sim.sim_data) > self.unify_data_threshold
                    and abs(layer_sim.sim_ast) < self.unify_ast_ceiling):
                    candidates.append(RefactoringCandidate(
                        kind="interface_unification",
                        file_a=id_a,
                        file_b=id_b,
                        sim_ast=layer_sim.sim_ast,
                        sim_cfg=layer_sim.sim_cfg,
                        sim_data=layer_sim.sim_data,
                        recommendation=(
                            f"Same computation, different expression: "
                            f"Data={layer_sim.sim_data:.2f}, AST={layer_sim.sim_ast:.2f}. "
                            f"Candidate for interface unification."
                        ),
                    ))

        # Sort: structural duplicates first (higher combined sim), then unification
        candidates.sort(key=lambda c: (
            0 if c.kind == "structural_duplicate" else 1,
            -(abs(c.sim_ast) + abs(c.sim_cfg or 0) + abs(c.sim_data or 0)),
        ))

        return candidates

    def to_json(self, candidates: List[RefactoringCandidate]) -> Dict[str, Any]:
        """Serialize candidates to a JSON-compatible report structure.

        Args:
            candidates: List of detected refactoring candidates.

        Returns:
            Dictionary with structured report data.
        """
        duplicates = [c.to_dict() for c in candidates if c.kind == "structural_duplicate"]
        unifications = [c.to_dict() for c in candidates if c.kind == "interface_unification"]

        return {
            "refactoring_candidates": {
                "structural_duplicates": duplicates,
                "interface_unification_candidates": unifications,
                "total_candidates": len(candidates),
                "duplicate_count": len(duplicates),
                "unification_count": len(unifications),
            }
        }
