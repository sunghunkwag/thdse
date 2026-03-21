"""SwarmOrchestrator — Central coordinator for the THDSE-Swarm.

Manages agent lifecycle, collects candidates, computes algebraic consensus
via correlate_matrix + Bron-Kerbosch, handles UNSAT broadcasting (Global
Quotient Collapse), and tracks swarm-level metrics.

Each round:
  1. Each agent runs local synthesis -> emits candidate PhaseMessages
  2. Orchestrator collects all candidates
  3. Compute consensus via correlate_matrix + Bron-Kerbosch
  4. Decode consensus centroid -> Z3 -> Python AST
  5. If SAT: execute in sandbox -> evaluate fitness
     - If fitness > threshold: broadcast consensus to all agents
  6. If UNSAT: extract V_error -> broadcast to ALL agents
     - Every agent immediately projects out V_error from its arena
  7. Track metrics
  8. Halt when: max_rounds reached OR stagnation_limit consecutive
     rounds with zero progress
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import hdc_core

from src.decoder.constraint_decoder import ConstraintDecoder
from src.decoder.subtree_vocab import SubTreeVocabulary
from src.execution.sandbox import ExecutionSandbox
from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection

from src.swarm.protocol import PhaseMessage, SwarmConfig
from src.swarm.agent import ThdseAgent
from src.swarm.consensus import ConsensusResult, compute_swarm_consensus

logger = logging.getLogger(__name__)


@dataclass
class SwarmRoundResult:
    """Result of a single swarm round."""
    round_index: int
    candidates_collected: int
    candidates_per_agent: List[int]
    consensus_reached: bool
    consensus_clique_size: int
    consensus_mean_resonance: float
    decoded_source: Optional[str]
    fitness: float
    walls_broadcast: int
    vocab_growth_per_agent: List[int]


@dataclass
class SwarmResult:
    """Aggregate result of the full swarm run."""
    rounds_completed: int
    total_candidates: int
    total_consensus_reached: int
    total_walls_broadcast: int
    total_vocab_growth: int
    best_fitness: float
    best_source: Optional[str]
    round_history: List[SwarmRoundResult] = field(default_factory=list)
    per_agent_stats: List[Dict[str, Any]] = field(default_factory=list)
    convergence_diagnosis: str = ""
    f_eff_expansion_rate: float = 0.0


class SwarmOrchestrator:
    """Orchestrates the THDSE-Swarm collective intelligence loop."""

    def __init__(self, config: SwarmConfig):
        """Initialize the orchestrator and all agents.

        Creates N ThdseAgent instances, each with its assigned corpus.
        Also creates the orchestrator's own arena/decoder for consensus
        decoding (separate from agent arenas).
        """
        self.config = config
        self.agents: List[ThdseAgent] = []

        # Validate corpus_paths length
        if len(config.corpus_paths) != config.n_agents:
            raise ValueError(
                f"corpus_paths has {len(config.corpus_paths)} entries, "
                f"expected {config.n_agents} (one per agent)"
            )

        # Check heterogeneity: warn if corpora overlap
        corpus_sets = [set(paths) for paths in config.corpus_paths]
        for i in range(len(corpus_sets)):
            for j in range(i + 1, len(corpus_sets)):
                if corpus_sets[i] and corpus_sets[j] and corpus_sets[i] == corpus_sets[j]:
                    logger.warning(
                        "Agents %d and %d have identical corpora. "
                        "Identical agents produce identical resonance landscapes "
                        "and degrade collective intelligence.", i, j,
                    )

        # Create agents
        for i in range(config.n_agents):
            agent = ThdseAgent(i, config, config.corpus_paths[i])
            self.agents.append(agent)

        # Orchestrator's own arena for consensus decoding
        self._consensus_arena = hdc_core.FhrrArena(
            config.arena_capacity, config.dimension,
        )
        self._consensus_projector = IsomorphicProjector(
            self._consensus_arena, config.dimension,
        )
        self._consensus_sandbox = ExecutionSandbox()

        # Orchestrator's merged vocabulary (built after agent ingestion)
        self._merged_vocab: Optional[SubTreeVocabulary] = None
        self._consensus_decoder: Optional[ConstraintDecoder] = None

    def _build_merged_vocab(self) -> None:
        """Merge SubTreeVocabulary from ALL agents into the orchestrator's decoder.

        Each agent's unique vocab atoms become part of the orchestrator's decoder.
        This means the orchestrator can decode structures that no single agent
        could decode alone.
        """
        self._merged_vocab = SubTreeVocabulary()

        for agent in self.agents:
            # Merge each agent's vocab atoms by ingesting their canonical sources
            for atom in agent.vocab.get_atoms().values():
                self._merged_vocab.ingest_source(atom.canonical_source)

        # Project merged vocab in orchestrator's arena
        self._merged_vocab.project_all(
            self._consensus_arena, self._consensus_projector,
        )

        # Create orchestrator decoder
        self._consensus_decoder = ConstraintDecoder(
            self._consensus_arena, self._consensus_projector,
            self.config.dimension,
            activation_threshold=0.04,
            subtree_vocab=self._merged_vocab,
        )

    def run(self) -> SwarmResult:
        """Execute the full swarm loop.

        Returns SwarmResult with complete round history and metrics.
        """
        # Phase 1: Ingest corpora for all agents
        for agent in self.agents:
            if agent.corpus_paths:
                agent.ingest_corpus()

        # Build merged vocabulary for consensus decoding
        self._build_merged_vocab()

        # Phase 2: Run swarm loop
        round_history: List[SwarmRoundResult] = []
        total_candidates = 0
        total_consensus = 0
        total_walls = 0
        best_fitness = 0.0
        best_source: Optional[str] = None
        consecutive_zero_progress = 0

        for round_idx in range(self.config.max_rounds):
            # Track vocab sizes before this round
            vocab_before = [agent.vocab.size() for agent in self.agents]

            # Step 1: Collect candidates from all agents
            all_candidates = self._collect_candidates()
            candidates_per_agent = [0] * self.config.n_agents
            for msg in all_candidates:
                candidates_per_agent[msg.sender_id] += 1

            total_candidates += len(all_candidates)

            # Step 2: Compute consensus
            consensus_reached = False
            clique_size = 0
            mean_resonance = 0.0
            decoded_source = None
            fitness = 0.0
            walls_this_round = 0

            if len(all_candidates) >= 2:
                # Create temporary arena for consensus
                tmp_arena = hdc_core.FhrrArena(
                    len(all_candidates) * 4 + 100, self.config.dimension,
                )

                candidate_phases = [msg.phases for msg in all_candidates]
                candidate_metadata = [msg.metadata for msg in all_candidates]

                consensus = compute_swarm_consensus(
                    tmp_arena,
                    candidate_phases,
                    candidate_metadata,
                    consensus_threshold=self.config.consensus_threshold,
                    min_clique_size=2,
                )

                if consensus is not None:
                    consensus_reached = True
                    clique_size = consensus.clique_size
                    mean_resonance = consensus.mean_resonance
                    total_consensus += 1

                    # Step 3: Decode consensus centroid
                    decoded_source, is_unsat = self._decode_consensus(
                        consensus.centroid_phases,
                    )

                    if decoded_source is not None:
                        # Step 4: Execute and evaluate
                        profile = self._consensus_sandbox.execute(decoded_source)
                        fitness = profile.fitness

                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_source = decoded_source

                        # Step 5: Broadcast consensus to agents if above threshold
                        if fitness >= self.config.fitness_threshold:
                            self._broadcast_consensus(consensus.centroid_phases)
                    else:
                        # UNSAT: broadcast wall to all agents
                        # Use the consensus centroid as V_error
                        self._broadcast_wall(consensus.centroid_phases)
                        walls_this_round += 1
                        total_walls += 1

            # Track vocab growth
            vocab_after = [agent.vocab.size() for agent in self.agents]
            vocab_growth = [
                vocab_after[i] - vocab_before[i]
                for i in range(self.config.n_agents)
            ]

            round_result = SwarmRoundResult(
                round_index=round_idx,
                candidates_collected=len(all_candidates),
                candidates_per_agent=candidates_per_agent,
                consensus_reached=consensus_reached,
                consensus_clique_size=clique_size,
                consensus_mean_resonance=mean_resonance,
                decoded_source=decoded_source,
                fitness=fitness,
                walls_broadcast=walls_this_round,
                vocab_growth_per_agent=vocab_growth,
            )
            round_history.append(round_result)

            # Check stagnation
            round_had_progress = (
                sum(vocab_growth) > 0
                or (consensus_reached and fitness >= self.config.fitness_threshold)
            )
            if round_had_progress:
                consecutive_zero_progress = 0
            else:
                consecutive_zero_progress += 1
                if consecutive_zero_progress >= self.config.stagnation_limit:
                    break

        # Compute final metrics
        per_agent_stats = [agent.get_local_stats() for agent in self.agents]
        total_vocab_growth = sum(
            sum(r.vocab_growth_per_agent) for r in round_history
        )

        # F_eff expansion rate
        f_eff = total_vocab_growth / max(total_candidates, 1)

        # Convergence diagnosis
        n_rounds = len(round_history)
        consensus_rounds = sum(1 for r in round_history if r.consensus_reached)
        stagnated = consecutive_zero_progress >= self.config.stagnation_limit

        if total_vocab_growth == 0 and total_consensus == 0:
            diagnosis = (
                f"No consensus reached in {n_rounds} rounds. "
                f"Agents produced {total_candidates} candidates total but none "
                f"resonated above threshold ({self.config.consensus_threshold}). "
                f"Consider lowering consensus_threshold or using more diverse corpora."
            )
        elif total_vocab_growth == 0:
            diagnosis = (
                f"Consensus reached {consensus_rounds}/{n_rounds} rounds but "
                f"no vocabulary growth. Decoded syntheses did not pass fitness "
                f"threshold ({self.config.fitness_threshold})."
            )
        elif stagnated:
            diagnosis = (
                f"Partial expansion -- {total_vocab_growth} new atoms across "
                f"{consensus_rounds}/{n_rounds} consensus rounds, "
                f"then {self.config.stagnation_limit} consecutive zero-progress rounds."
            )
        else:
            diagnosis = (
                f"Active expansion -- {total_vocab_growth} new atoms across "
                f"{consensus_rounds}/{n_rounds} consensus rounds "
                f"(expansion rate: {f_eff:.4f})."
            )

        return SwarmResult(
            rounds_completed=n_rounds,
            total_candidates=total_candidates,
            total_consensus_reached=total_consensus,
            total_walls_broadcast=total_walls,
            total_vocab_growth=total_vocab_growth,
            best_fitness=best_fitness,
            best_source=best_source,
            round_history=round_history,
            per_agent_stats=per_agent_stats,
            convergence_diagnosis=diagnosis,
            f_eff_expansion_rate=f_eff,
        )

    def _collect_candidates(self) -> List[PhaseMessage]:
        """Run local synthesis on all agents, collect emitted candidates.

        Sequential execution: agent 0 runs, then agent 1, etc.
        """
        all_candidates: List[PhaseMessage] = []
        for agent in self.agents:
            candidates = agent.run_local_synthesis(max_cliques=10)
            all_candidates.extend(candidates)
        return all_candidates

    def _broadcast_wall(self, wall_phases: List[float]) -> None:
        """Broadcast V_error to all agents.

        Each agent calls receive_wall_broadcast(wall_phases),
        which triggers immediate quotient space projection
        in that agent's local arena.
        """
        for agent in self.agents:
            agent.receive_wall_broadcast(wall_phases)

    def _broadcast_consensus(self, consensus_phases: List[float]) -> None:
        """Broadcast successful consensus to all agents.

        Each agent attempts to decode and expand vocab from
        the consensus vector.
        """
        for agent in self.agents:
            agent.receive_consensus(consensus_phases)

    def _decode_consensus(
        self, consensus_phases: List[float],
    ) -> Tuple[Optional[str], bool]:
        """Decode the consensus centroid to Python source.

        Uses the orchestrator's own decoder (not any agent's).

        Returns:
            (decoded_source_or_None, is_unsat)
        """
        if self._consensus_decoder is None:
            return None, True

        # Inject consensus into orchestrator's arena
        consensus_h = self._consensus_arena.allocate()
        self._consensus_arena.inject_phases(consensus_h, consensus_phases)

        # Create a LayeredProjection for decoding
        lp = LayeredProjection(
            final_handle=consensus_h,
            ast_handle=consensus_h,
            cfg_handle=None,
            data_handle=None,
            ast_phases=consensus_phases,
            cfg_phases=None,
            data_phases=None,
        )

        source = self._consensus_decoder.decode_to_source(lp)
        if source and source.strip() and source.strip() != "pass":
            return source, False
        return None, True

    def ingest_agent_corpora_dicts(
        self, corpora: List[Dict[str, str]],
    ) -> None:
        """Ingest corpora from dictionaries (for testing).

        Args:
            corpora: List of dicts, one per agent (source_id -> source_code).
        """
        if len(corpora) != self.config.n_agents:
            raise ValueError(
                f"Expected {self.config.n_agents} corpora, got {len(corpora)}"
            )

        for agent, corpus in zip(self.agents, corpora):
            agent.ingest_corpus_dict(corpus)

        # Build merged vocabulary
        self._build_merged_vocab()
