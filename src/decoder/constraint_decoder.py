"""
Constraint Decoder — Translates a synthesized FHRR hypervector back into
a concrete Python AST via deterministic SMT solving (Z3).

Architecture:
  1. Probe (layer-aware): unbind non-target layers via conjugate, then
     correlate against layer-specific vocabulary atoms.
  2. Encode: translate constraints into Z3 formulas.
  3. Solve: if SAT → compile model to AST; if UNSAT → reject.

Algebraic fix (unbinding):
  Given final = ast ⊗ cfg ⊗ data, to recover AST-layer signal:
    recovered_ast = final ⊗ conj(cfg) ⊗ conj(data)
  Since bind(V, conj(V)) ≈ identity in FHRR, this cancels the
  non-target layers, exposing the AST bundle for direct probing.

Strict guarantees:
  - Zero randomness: Z3's DPLL(T) is deterministic for a fixed formula.
  - Zero hallucination: only satisfiable structures are emitted.
"""

import ast
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import z3

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.utils.arena_ops import conjugate_into, negate_phases


# ── Structural atoms vocabulary ──────────────────────────────────

class AtomKind(Enum):
    NODE_TYPE = auto()
    CFG_SEQUENCE = auto()
    CFG_BRANCH = auto()
    CFG_LOOP = auto()
    DATA_DEF_USE = auto()


@dataclass
class StructuralAtom:
    kind: AtomKind
    label: str
    handle: int
    z3_var: Optional[z3.BoolRef] = None


@dataclass
class DecodedConstraints:
    """The set of topological constraints extracted from probing."""
    active_node_types: List[str] = field(default_factory=list)
    cfg_sequences: List[Tuple[str, str]] = field(default_factory=list)
    cfg_branches: List[Tuple[str, str, str]] = field(default_factory=list)
    cfg_loops: List[Tuple[str, str]] = field(default_factory=list)
    data_deps: List[Tuple[str, str]] = field(default_factory=list)
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
        for ntype in _ALL_NODE_TYPES:
            label = f"atom:type:{ntype}"
            handle = self._mint_atom_handle(label)
            self._atom_vocab[label] = StructuralAtom(
                kind=AtomKind.NODE_TYPE, label=label, handle=handle,
            )

        for a in _STATEMENT_TYPES:
            for b in _STATEMENT_TYPES:
                label = f"atom:cfg_seq:{a}->{b}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_SEQUENCE, label=label, handle=handle,
                )

        for cond_type in ["If"]:
            for body_type in _STATEMENT_TYPES:
                label = f"atom:cfg_branch:{cond_type}->{body_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_BRANCH, label=label, handle=handle,
                )

        for loop_type in ["While", "For"]:
            for body_type in _STATEMENT_TYPES:
                label = f"atom:cfg_loop:{loop_type}->{body_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_LOOP, label=label, handle=handle,
                )

        for def_type in ["Assign", "AugAssign", "AnnAssign", "For", "FunctionDef"]:
            for use_type in _STATEMENT_TYPES:
                label = f"atom:data_dep:{def_type}->{use_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.DATA_DEF_USE, label=label, handle=handle,
                )

    # ── Unbinding helpers ────────────────────────────────────────

    def _recover_layer(
        self, projection: LayeredProjection, target: str,
    ) -> int:
        """Unbind non-target layers from the final handle to recover a single layer.

        Given final = ast ⊗ cfg ⊗ data:
          recover "ast" → final ⊗ conj(cfg) ⊗ conj(data)
          recover "cfg" → final ⊗ conj(ast) ⊗ conj(data)
          recover "data" → final ⊗ conj(ast) ⊗ conj(cfg)

        If a layer is None (absent), it was never bound in — skip it.
        """
        result_h = projection.final_handle

        # Determine which layers to unbind (all except target)
        layers_to_unbind = []
        if target != "ast":
            # ast is always present; always unbind it unless it IS the target
            layers_to_unbind.append(projection.ast_phases)
        if target != "cfg" and projection.cfg_phases is not None:
            layers_to_unbind.append(projection.cfg_phases)
        if target != "data" and projection.data_phases is not None:
            layers_to_unbind.append(projection.data_phases)

        for phases in layers_to_unbind:
            conj_h = self.arena.allocate()
            conjugate_into(self.arena, phases, conj_h)
            unbound = self.arena.allocate()
            self.arena.bind(result_h, conj_h, unbound)
            result_h = unbound

        return result_h

    # ── Phase 1: Layer-aware probing ─────────────────────────────

    def probe_layered(self, projection: LayeredProjection) -> DecodedConstraints:
        """Probe each layer independently after unbinding other layers.

        - Node-type atoms (AST layer): probe against recovered AST
        - CFG atoms: probe against recovered CFG
        - Data-dep atoms: probe against recovered Data-dep
        """
        constraints = DecodedConstraints()

        # Recover each layer
        recovered_ast = self._recover_layer(projection, "ast")
        recovered_cfg = self._recover_layer(projection, "cfg") if projection.cfg_handle is not None else None
        recovered_data = self._recover_layer(projection, "data") if projection.data_handle is not None else None

        for label, atom in self._atom_vocab.items():
            # Route each atom kind to the correct recovered layer
            if atom.kind == AtomKind.NODE_TYPE:
                target_h = recovered_ast
            elif atom.kind in (AtomKind.CFG_SEQUENCE, AtomKind.CFG_BRANCH, AtomKind.CFG_LOOP):
                target_h = recovered_cfg
            elif atom.kind == AtomKind.DATA_DEF_USE:
                target_h = recovered_data
            else:
                continue

            if target_h is None:
                continue

            corr = self.arena.compute_correlation(target_h, atom.handle)
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

    def probe(self, input_: Union[int, LayeredProjection]) -> DecodedConstraints:
        """Probe — dispatches to layer-aware or legacy based on input type.

        If a LayeredProjection is provided, uses layer-aware unbinding (correct).
        If a bare int handle is provided, falls back to direct correlation (legacy).
        """
        if isinstance(input_, LayeredProjection):
            return self.probe_layered(input_)

        # Legacy fallback: direct correlation against composed vector (algebraically broken
        # for cross-layer-bound vectors, but preserved for backward compatibility)
        synth_handle = input_
        constraints = DecodedConstraints()
        for label, atom in self._atom_vocab.items():
            corr = self.arena.compute_correlation(synth_handle, atom.handle)
            constraints.resonance_scores[label] = corr
            if abs(corr) < self.activation_threshold:
                continue
            if atom.kind == AtomKind.NODE_TYPE:
                constraints.active_node_types.append(label.split(":")[-1])
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
        """Translate topological constraints into Z3 formulas."""
        solver = z3.Solver()
        vars_map = {}

        node_vars = {}
        for ntype in _ALL_NODE_TYPES:
            v = z3.Bool(f"present_{ntype}")
            node_vars[ntype] = v
            vars_map[f"present_{ntype}"] = v

        # Closed-world assumption: force active nodes present, all others absent.
        active_set = set(constraints.active_node_types)
        for ntype, v in node_vars.items():
            if ntype in active_set:
                solver.add(v == True)
            else:
                solver.add(v == False)

        order_vars = {}
        for stype in _STATEMENT_TYPES:
            v = z3.Int(f"order_{stype}")
            order_vars[stype] = v
            vars_map[f"order_{stype}"] = v
            solver.add(v >= 0)
            solver.add(z3.Implies(z3.Not(node_vars.get(stype, z3.BoolVal(False))), v == 0))

        for a, b in constraints.cfg_sequences:
            if a in order_vars and b in order_vars and a in node_vars and b in node_vars:
                solver.add(z3.Implies(
                    z3.And(node_vars[a], node_vars[b]),
                    order_vars[a] < order_vars[b],
                ))

        for cond, then_t, else_t in constraints.cfg_branches:
            if cond in node_vars:
                targets = []
                if then_t in node_vars:
                    targets.append(node_vars[then_t])
                if else_t in node_vars:
                    targets.append(node_vars[else_t])
                if targets:
                    solver.add(z3.Implies(node_vars[cond], z3.Or(*targets)))

        for header, body_type in constraints.cfg_loops:
            if header in node_vars and body_type in node_vars:
                solver.add(z3.Implies(node_vars[header], node_vars[body_type]))
                if header in order_vars and body_type in order_vars:
                    solver.add(z3.Implies(
                        z3.And(node_vars[header], node_vars[body_type]),
                        order_vars[header] <= order_vars[body_type],
                    ))

        for def_type, use_type in constraints.data_deps:
            if def_type in order_vars and use_type in order_vars:
                if def_type in node_vars and use_type in node_vars:
                    solver.add(z3.Implies(
                        z3.And(node_vars[def_type], node_vars[use_type]),
                        order_vars[def_type] < order_vars[use_type],
                    ))

        if "Module" in node_vars:
            solver.add(node_vars["Module"] == True)

        if "FunctionDef" in node_vars:
            body_candidates = [
                node_vars[s] for s in ["Return", "Assign", "Expr", "Pass", "If", "While", "For", "Raise"]
                if s in node_vars
            ]
            if body_candidates:
                solver.add(z3.Implies(node_vars["FunctionDef"], z3.Or(*body_candidates)))

        max_stmts = len(constraints.active_node_types) + 5
        for v in order_vars.values():
            solver.add(v <= max_stmts)

        return solver, vars_map

    # ── Phase 3: AST Compilation ─────────────────────────────────

    def compile_model(self, model: z3.ModelRef, vars_map: Dict) -> ast.Module:
        """Compile a Z3 satisfying model into a Python AST."""
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

        present_exprs = set()
        for etype in _EXPRESSION_TYPES:
            var_name = f"present_{etype}"
            if var_name in vars_map:
                val = model.evaluate(vars_map[var_name], model_completion=True)
                if z3.is_true(val):
                    present_exprs.add(etype)

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

    def decode(self, input_: Union[int, LayeredProjection]) -> Optional[ast.Module]:
        """Full pipeline: probe → encode SMT → solve → compile AST.

        Accepts LayeredProjection (layer-aware) or bare int (legacy).
        Returns None if UNSAT.
        """
        constraints = self.probe(input_)

        if not constraints.active_node_types:
            return None

        solver, vars_map = self.encode_smt(constraints)

        result = solver.check()
        if result != z3.sat:
            return None

        model = solver.model()
        module = self.compile_model(model, vars_map)
        return module

    def decode_to_source(self, input_: Union[int, LayeredProjection]) -> Optional[str]:
        """Convenience: decode and unparse to Python source string."""
        module = self.decode(input_)
        if module is None:
            return None
        return ast.unparse(module)

    def decode_and_verify(
        self, input_: Union[int, LayeredProjection]
    ) -> Tuple[Optional[str], Optional[float]]:
        """Decode, then re-project and verify round-trip correlation."""
        source = self.decode_to_source(input_)
        if source is None:
            return None, None

        reprojected = self.projector.project(source)
        if isinstance(input_, LayeredProjection):
            correlation = self.arena.compute_correlation(input_.final_handle, reprojected.final_handle)
        else:
            correlation = self.arena.compute_correlation(input_, reprojected.final_handle)
        return source, correlation
