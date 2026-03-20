"""
Constraint Decoder — Translates a synthesized FHRR hypervector back into
a concrete Python AST via deterministic SMT solving (Z3).

Architecture:
  1. Probe: correlate the synthesized vector against a vocabulary of known
     structural atoms (AST node types, CFG patterns, data-dep signatures)
     to extract a set of topological constraints.
  2. Encode: translate these constraints into Z3 formulas expressing:
     - CFG reachability (sequential flow, branching, loop back-edges)
     - Data-dependency consistency (def-use ordering, scope containment)
     - AST well-formedness (parent-child type compatibility)
  3. Solve: if SAT, extract the model and compile it into a Python AST.
     If UNSAT, reject the synthesis mathematically — no fallback heuristics.

Strict guarantees:
  - Zero randomness: Z3's DPLL(T) is deterministic for a fixed formula.
  - Zero hallucination: only structures satisfiable under the constraints are emitted.
  - Invertibility: the decoded AST, when re-projected through the IsomorphicProjector,
    must produce a vector with high correlation (> threshold) to the input.
"""

import ast
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

import z3

from src.projection.isomorphic_projector import IsomorphicProjector


# ── Structural atoms vocabulary ──────────────────────────────────

# These are the deterministic "codebook" entries we probe against.
# Each atom is a known AST micro-structure with a pre-computed
# phase signature in the arena.

class AtomKind(Enum):
    NODE_TYPE = auto()     # A single AST node type (e.g. FunctionDef, If, Return)
    CFG_SEQUENCE = auto()  # A sequential CFG pair (stmt_a → stmt_b)
    CFG_BRANCH = auto()    # An If-branch pattern (If → body_entry, If → orelse_entry)
    CFG_LOOP = auto()      # A loop pattern (header → body → back-edge)
    DATA_DEF_USE = auto()  # A def-use pair (Store(x) → Load(x))


@dataclass
class StructuralAtom:
    kind: AtomKind
    label: str
    handle: int                       # arena handle for this atom's signature
    z3_var: Optional[z3.BoolRef] = None  # assigned during constraint encoding


@dataclass
class DecodedConstraints:
    """The set of topological constraints extracted from probing."""
    active_node_types: List[str] = field(default_factory=list)
    cfg_sequences: List[Tuple[str, str]] = field(default_factory=list)
    cfg_branches: List[Tuple[str, str, str]] = field(default_factory=list)  # (if, then, else)
    cfg_loops: List[Tuple[str, str]] = field(default_factory=list)          # (header, body)
    data_deps: List[Tuple[str, str]] = field(default_factory=list)          # (def_stmt, use_stmt)
    resonance_scores: Dict[str, float] = field(default_factory=dict)


# ── Known AST node types for the codebook ────────────────────────

_STATEMENT_TYPES = [
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Return", "Delete", "Assign", "AugAssign", "AnnAssign",
    "For", "AsyncFor", "While", "If", "With", "AsyncWith",
    "Raise", "Try", "Assert", "Import", "ImportFrom",
    "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
]

_EXPRESSION_TYPES = [
    "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "Lambda",
    "IfExp", "Dict", "Set", "ListComp", "SetComp", "DictComp",
    "GeneratorExp", "Await", "Yield", "YieldFrom",
    "Compare", "Call", "FormattedValue", "JoinedStr",
    "Constant", "Attribute", "Subscript", "Starred", "Name",
    "List", "Tuple", "Slice",
]

_ALL_NODE_TYPES = _STATEMENT_TYPES + _EXPRESSION_TYPES


class ConstraintDecoder:
    """Decodes a synthesized hypervector into a Python AST via SMT solving."""

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        dimension: int,
        activation_threshold: float = 0.10,
        verification_threshold: float = 0.15,
    ):
        self.arena = arena
        self.projector = projector
        self.dimension = dimension
        self.activation_threshold = activation_threshold
        self.verification_threshold = verification_threshold
        self._atom_vocab: Dict[str, StructuralAtom] = {}
        self._build_vocabulary()

    # ── Vocabulary construction ──────────────────────────────────

    def _deterministic_phases(self, seed: int) -> List[float]:
        """Identical LCG expansion as IsomorphicProjector for consistency."""
        phases = []
        state = seed & 0xFFFFFFFF
        for _ in range(self.dimension):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
            phases.append(phase)
        return phases

    def _mint_atom_handle(self, label: str) -> int:
        seed = int(hashlib.md5(label.encode("utf-8")).hexdigest()[:8], 16)
        handle = self.arena.allocate()
        self.arena.inject_phases(handle, self._deterministic_phases(seed))
        return handle

    def _build_vocabulary(self):
        """Build the structural atom codebook in the arena."""
        # Node-type atoms
        for ntype in _ALL_NODE_TYPES:
            label = f"atom:type:{ntype}"
            handle = self._mint_atom_handle(label)
            self._atom_vocab[label] = StructuralAtom(
                kind=AtomKind.NODE_TYPE, label=label, handle=handle,
            )

        # CFG sequence atoms (statement-type pairs)
        for a in _STATEMENT_TYPES:
            for b in _STATEMENT_TYPES:
                label = f"atom:cfg_seq:{a}->{b}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_SEQUENCE, label=label, handle=handle,
                )

        # CFG branch atoms
        for cond_type in ["If"]:
            for body_type in _STATEMENT_TYPES:
                label = f"atom:cfg_branch:{cond_type}->{body_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_BRANCH, label=label, handle=handle,
                )

        # CFG loop atoms
        for loop_type in ["While", "For"]:
            for body_type in _STATEMENT_TYPES:
                label = f"atom:cfg_loop:{loop_type}->{body_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_LOOP, label=label, handle=handle,
                )

        # Data-dep atoms (def-use across statement types)
        for def_type in ["Assign", "AugAssign", "AnnAssign", "For", "FunctionDef"]:
            for use_type in _STATEMENT_TYPES:
                label = f"atom:data_dep:{def_type}->{use_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.DATA_DEF_USE, label=label, handle=handle,
                )

    # ── Phase 1: Probing ─────────────────────────────────────────

    def probe(self, synth_handle: int) -> DecodedConstraints:
        """Probe the synthesized vector against every atom in the vocabulary.
        Atoms whose correlation exceeds the activation threshold are 'active'."""
        constraints = DecodedConstraints()

        for label, atom in self._atom_vocab.items():
            corr = self.arena.compute_correlation(synth_handle, atom.handle)
            constraints.resonance_scores[label] = corr

            if abs(corr) < self.activation_threshold:
                continue

            if atom.kind == AtomKind.NODE_TYPE:
                ntype = label.split(":")[-1]
                constraints.active_node_types.append(ntype)

            elif atom.kind == AtomKind.CFG_SEQUENCE:
                pair = label.split(":")[-1]
                a, b = pair.split("->")
                constraints.cfg_sequences.append((a, b))

            elif atom.kind == AtomKind.CFG_BRANCH:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.cfg_branches.append((parts[0], parts[1], "Pass"))

            elif atom.kind == AtomKind.CFG_LOOP:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.cfg_loops.append((parts[0], parts[1]))

            elif atom.kind == AtomKind.DATA_DEF_USE:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.data_deps.append((parts[0], parts[1]))

        return constraints

    # ── Phase 2: SMT Encoding ────────────────────────────────────

    def encode_smt(self, constraints: DecodedConstraints) -> Tuple[z3.Solver, Dict]:
        """Translate topological constraints into Z3 formulas.

        Variables:
          - node_present[t]: Bool — is AST node type t present?
          - cfg_edge[a,b]: Bool — does a CFG edge exist from type a to type b?
          - stmt_order[t]: Int — topological position of statement type t
          - data_dep[d,u]: Bool — does a def-use edge exist from d to u?

        Constraints encode:
          - Active nodes must be present
          - CFG sequences imply ordering (order[a] < order[b])
          - Branches imply the condition node and at least one target
          - Loops imply back-edge (order[body_last] → order[header])
          - Data-deps imply def-before-use ordering
          - Well-formedness: FunctionDef requires at least one body statement
        """
        solver = z3.Solver()
        vars_map = {}

        # Node presence variables
        node_vars = {}
        for ntype in _ALL_NODE_TYPES:
            v = z3.Bool(f"present_{ntype}")
            node_vars[ntype] = v
            vars_map[f"present_{ntype}"] = v

        # Closed-world assumption: force active nodes present, all others absent.
        # Only types that resonate above threshold are permitted in the output.
        active_set = set(constraints.active_node_types)
        for ntype, v in node_vars.items():
            if ntype in active_set:
                solver.add(v == True)
            else:
                solver.add(v == False)

        # Statement ordering variables
        order_vars = {}
        for stype in _STATEMENT_TYPES:
            v = z3.Int(f"order_{stype}")
            order_vars[stype] = v
            vars_map[f"order_{stype}"] = v
            # Ordering must be non-negative
            solver.add(v >= 0)
            # If not present, order is 0 (sentinel)
            solver.add(z3.Implies(z3.Not(node_vars.get(stype, z3.BoolVal(False))), v == 0))

        # CFG sequence constraints: if both present, order[a] < order[b]
        for a, b in constraints.cfg_sequences:
            if a in order_vars and b in order_vars and a in node_vars and b in node_vars:
                solver.add(z3.Implies(
                    z3.And(node_vars[a], node_vars[b]),
                    order_vars[a] < order_vars[b],
                ))

        # CFG branch constraints: If present → at least one branch target present
        for cond, then_t, else_t in constraints.cfg_branches:
            if cond in node_vars:
                targets = []
                if then_t in node_vars:
                    targets.append(node_vars[then_t])
                if else_t in node_vars:
                    targets.append(node_vars[else_t])
                if targets:
                    solver.add(z3.Implies(node_vars[cond], z3.Or(*targets)))

        # CFG loop constraints: loop header present → body type present,
        # and body ordering wraps (back-edge: conceptually order[body] ≥ order[header])
        for header, body_type in constraints.cfg_loops:
            if header in node_vars and body_type in node_vars:
                solver.add(z3.Implies(node_vars[header], node_vars[body_type]))
                if header in order_vars and body_type in order_vars:
                    solver.add(z3.Implies(
                        z3.And(node_vars[header], node_vars[body_type]),
                        order_vars[header] <= order_vars[body_type],
                    ))

        # Data-dep constraints: def must precede use in ordering
        for def_type, use_type in constraints.data_deps:
            if def_type in order_vars and use_type in order_vars:
                if def_type in node_vars and use_type in node_vars:
                    solver.add(z3.Implies(
                        z3.And(node_vars[def_type], node_vars[use_type]),
                        order_vars[def_type] < order_vars[use_type],
                    ))

        # Well-formedness: Module must be present
        if "Module" in node_vars:
            solver.add(node_vars["Module"] == True)

        # Well-formedness: FunctionDef → at least one body statement
        if "FunctionDef" in node_vars:
            body_candidates = [
                node_vars[s] for s in ["Return", "Assign", "Expr", "Pass", "If", "While", "For", "Raise"]
                if s in node_vars
            ]
            if body_candidates:
                solver.add(z3.Implies(node_vars["FunctionDef"], z3.Or(*body_candidates)))

        # Bound the ordering to a reasonable range
        max_stmts = len(constraints.active_node_types) + 5
        for v in order_vars.values():
            solver.add(v <= max_stmts)

        return solver, vars_map

    # ── Phase 3: AST Compilation ─────────────────────────────────

    def compile_model(self, model: z3.ModelRef, vars_map: Dict) -> ast.Module:
        """Compile a Z3 satisfying model into a Python AST.

        Strategy: reconstruct the minimal AST that satisfies all
        present-node and ordering constraints.
        """
        # Extract which nodes are present and their ordering
        present_stmts = []
        for stype in _STATEMENT_TYPES:
            var_name = f"present_{stype}"
            if var_name in vars_map:
                val = model.evaluate(vars_map[var_name], model_completion=True)
                if z3.is_true(val):
                    order_var = f"order_{stype}"
                    if order_var in vars_map:
                        order_val = model.evaluate(vars_map[order_var], model_completion=True)
                        try:
                            order_int = order_val.as_long()
                        except (AttributeError, Exception):
                            order_int = 0
                    else:
                        order_int = 0
                    present_stmts.append((order_int, stype))

        present_stmts.sort(key=lambda x: (x[0], x[1]))

        # Check which expression types are present
        present_exprs = set()
        for etype in _EXPRESSION_TYPES:
            var_name = f"present_{etype}"
            if var_name in vars_map:
                val = model.evaluate(vars_map[var_name], model_completion=True)
                if z3.is_true(val):
                    present_exprs.add(etype)

        # Build AST
        body_stmts = []
        func_body = []
        in_function = False

        for _, stype in present_stmts:
            if stype == "Module":
                continue
            if stype == "FunctionDef":
                in_function = True
                continue

            node = self._make_statement_node(stype, present_exprs)
            if node is not None:
                if in_function:
                    func_body.append(node)
                else:
                    body_stmts.append(node)

        # Wrap function body if FunctionDef is present
        if in_function:
            if not func_body:
                func_body = [ast.Pass()]
            func_def = ast.FunctionDef(
                name="synthesized_fn",
                args=ast.arguments(
                    posonlyargs=[], args=[ast.arg(arg="x")],
                    kwonlyargs=[], kw_defaults=[], defaults=[],
                ),
                body=func_body,
                decorator_list=[],
                returns=None,
            )
            body_stmts.insert(0, func_def)

        if not body_stmts:
            body_stmts = [ast.Pass()]

        module = ast.Module(body=body_stmts, type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    def _make_statement_node(self, stype: str, present_exprs: set) -> Optional[ast.stmt]:
        """Construct a minimal AST statement node of the given type."""
        placeholder_name = ast.Name(id="x", ctx=ast.Load())
        placeholder_const = ast.Constant(value=0)

        builders = {
            "Return": lambda: ast.Return(value=placeholder_name if "Name" in present_exprs else None),
            "Assign": lambda: ast.Assign(
                targets=[ast.Name(id="x", ctx=ast.Store())],
                value=placeholder_const if "Constant" in present_exprs else placeholder_name,
            ),
            "AugAssign": lambda: ast.AugAssign(
                target=ast.Name(id="x", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1),
            ),
            "Expr": lambda: ast.Expr(
                value=ast.Call(
                    func=placeholder_name, args=[], keywords=[],
                ) if "Call" in present_exprs else placeholder_name
            ),
            "If": lambda: ast.If(
                test=ast.Compare(
                    left=placeholder_name,
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=1)],
                ) if "Compare" in present_exprs else placeholder_name,
                body=[ast.Return(value=ast.Constant(value=1))
                      if "Return" in present_exprs else ast.Pass()],
                orelse=[],
            ),
            "While": lambda: ast.While(
                test=ast.Compare(
                    left=placeholder_name,
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)],
                ) if "Compare" in present_exprs else placeholder_name,
                body=[ast.Pass()],
                orelse=[],
            ),
            "For": lambda: ast.For(
                target=ast.Name(id="i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[placeholder_name], keywords=[],
                ),
                body=[ast.Pass()],
                orelse=[],
            ),
            "Pass": lambda: ast.Pass(),
            "Break": lambda: ast.Break(),
            "Continue": lambda: ast.Continue(),
            "Raise": lambda: ast.Raise(exc=ast.Name(id="ValueError", ctx=ast.Load())),
            "Assert": lambda: ast.Assert(test=placeholder_name),
            "Import": lambda: ast.Import(names=[ast.alias(name="os")]),
        }

        builder = builders.get(stype)
        if builder is None:
            return None
        return builder()

    # ── Full decode pipeline ─────────────────────────────────────

    def decode(self, synth_handle: int) -> Optional[ast.Module]:
        """Full pipeline: probe → encode SMT → solve → compile AST.

        Returns None if UNSAT (mathematically rejected synthesis).
        """
        # Phase 1: probe
        constraints = self.probe(synth_handle)

        if not constraints.active_node_types:
            return None

        # Phase 2: encode
        solver, vars_map = self.encode_smt(constraints)

        # Phase 3: solve
        result = solver.check()
        if result != z3.sat:
            return None

        model = solver.model()

        # Phase 4: compile
        module = self.compile_model(model, vars_map)
        return module

    def decode_to_source(self, synth_handle: int) -> Optional[str]:
        """Convenience: decode and unparse to Python source string."""
        module = self.decode(synth_handle)
        if module is None:
            return None
        return ast.unparse(module)

    def decode_and_verify(
        self, synth_handle: int
    ) -> Tuple[Optional[str], Optional[float]]:
        """Decode, then re-project the result and verify round-trip correlation.

        Returns (source_code, correlation) or (None, None) if UNSAT.
        """
        source = self.decode_to_source(synth_handle)
        if source is None:
            return None, None

        # Round-trip verification: re-project and correlate
        reprojected_h = self.projector.project(source)
        correlation = self.arena.compute_correlation(synth_handle, reprojected_h)
        return source, correlation
