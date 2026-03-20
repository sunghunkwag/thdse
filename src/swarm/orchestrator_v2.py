"""
AutophagicSwarmOrchestrator v2 — Resonance-guided mutation.

Instead of blind 100-typist spray, uses axiom resonance scores
to construct directional feedback in each typist's prompt.

Structural honesty:
  - The "gradient" is resonance(code, axiom) projected into natural
    language instructions.  This is a LOSSY channel: the LLM may or
    may not interpret vector-space direction as meaningful code edits.
  - Calling this a "topological gradient" is terminologically
    inaccurate.  It is a vector-space similarity signal fed back
    through prompt engineering.
  - Practical value: prunes the mutation space and gives typists
    actionable constraints, which should raise survivor rate vs
    blind brute-force.
"""

import asyncio
from typing import List, Dict, Tuple
import httpx

from src.topology.ast_graph_ca import TopologicalASTGraphCA
from src.prophet.gnn_analyzer import TheProphet
from src.scribe.typist import LLM_Typist


class ResonanceGuidedOrchestrator:
    def __init__(self, encoder, axiom_engine, prophet: TheProphet):
        self.encoder = encoder
        self.arena = encoder.arena
        self.axiom_engine = axiom_engine
        self.prophet = prophet
        self.ca = TopologicalASTGraphCA(steps=5)

    # ── Resonance feedback extraction ────────────────────────────

    def _compute_resonance_report(self, code_handle: int) -> List[Dict]:
        """For each axiom, compute resonance and produce a structured
        report that can be injected into typist prompts."""
        report = []
        for bug_id, axiom_h in self.axiom_engine.db.get_all_signatures().items():
            res = self.arena.compute_correlation(code_handle, axiom_h)
            meta = self.axiom_engine.db.metadata.get(bug_id, {})
            report.append({
                "bug_id": bug_id,
                "resonance": res,
                "divergence": meta.get("topological_divergence", 0.0),
            })
        # Sort by resonance descending — highest-risk axioms first
        report.sort(key=lambda x: x["resonance"], reverse=True)
        return report

    def _format_resonance_directive(self, report: List[Dict], threshold: float = 0.4) -> str:
        """Convert resonance report into natural-language constraints
        for the typist prompt."""
        lines = []
        for entry in report:
            if entry["resonance"] > threshold:
                lines.append(
                    f"- Bug pattern '{entry['bug_id']}': resonance={entry['resonance']:.3f}. "
                    f"Your mutation must REDUCE similarity to this pattern."
                )
        if not lines:
            return "No strong axiom resonances detected. Apply general structural improvements."
        header = (
            "RESONANCE FEEDBACK (higher = more similar to known bugs):\n"
            "Prioritize edits that reduce the highest-resonance scores.\n"
        )
        return header + "\n".join(lines)

    # ── Two-phase pipeline ───────────────────────────────────────

    async def execute_pipeline(self, buggy_code: str, n_typists: int = 100) -> str:
        # Phase 0: encode input and get GNN attention mask
        ast_graph = self.ca.code_to_graph(buggy_code)
        attention_mask = self.prophet.generate_attention_mask(ast_graph, k=5)

        # Phase 1: blind first pass (small batch) to seed resonance
        phase1_count = max(n_typists // 5, 10)
        phase1_mutations = await self._run_typists(
            buggy_code, attention_mask,
            resonance_directive="",
            start_id=0,
            count=phase1_count,
        )

        # Encode phase-1 survivors and compute resonance
        best_handle, best_code = self._select_best(phase1_mutations)
        if best_handle is None:
            raise RuntimeError("Phase-1 extinction: no valid mutations.")

        resonance_report = self._compute_resonance_report(best_handle)
        directive = self._format_resonance_directive(resonance_report)

        # Phase 2: guided pass with resonance feedback
        phase2_count = n_typists - phase1_count
        phase2_mutations = await self._run_typists(
            buggy_code, attention_mask,
            resonance_directive=directive,
            start_id=phase1_count,
            count=phase2_count,
        )

        all_mutations = phase1_mutations + phase2_mutations
        final_handle, final_code = self._select_best(all_mutations)
        if final_handle is None:
            raise RuntimeError("Full extinction: all mutations eliminated.")

        self.arena.reset()
        return final_code

    # ── Helpers ──────────────────────────────────────────────────

    async def _run_typists(
        self,
        base_code: str,
        attention_mask: str,
        resonance_directive: str,
        start_id: int,
        count: int,
    ) -> List[str]:
        typists = [LLM_Typist(start_id + i) for i in range(count)]
        async with httpx.AsyncClient(timeout=15.0) as client:
            tasks = [
                self._guided_rewrite(t, client, base_code, attention_mask, resonance_directive)
                for t in typists
            ]
            return await asyncio.gather(*tasks)

    async def _guided_rewrite(
        self,
        typist: LLM_Typist,
        client: httpx.AsyncClient,
        base_code: str,
        attention_mask: str,
        resonance_directive: str,
    ) -> str:
        """Wraps typist.rewrite_ast with resonance directive prepended
        to the attention mask."""
        if resonance_directive:
            enriched_mask = f"{attention_mask}\n\n{resonance_directive}"
        else:
            enriched_mask = attention_mask
        return await typist.rewrite_ast(client, base_code, enriched_mask)

    def _select_best(self, mutations: List[str]) -> Tuple:
        """Encode all valid mutations, return the one with lowest
        maximum resonance against known axioms (= least buggy).
        Deduplicates mutations and resets encoder cache per candidate
        to prevent arena handle exhaustion."""
        unique_mutations = list(dict.fromkeys(mutations))
        candidates = []
        for code in unique_mutations:
            try:
                self.encoder.seed_cache.clear()
                h = self.encoder.encode_code(code)
            except (SyntaxError, Exception):
                continue
            max_res = 0.0
            for axiom_h in self.axiom_engine.db.signatures.values():
                res = self.arena.compute_correlation(h, axiom_h)
                if res > max_res:
                    max_res = res
            candidates.append((max_res, h, code))

        if not candidates:
            return None, None

        # Lowest max-resonance = furthest from all known bugs
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]
