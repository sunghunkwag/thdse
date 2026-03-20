import httpx
import ast

class RuleBasedASTPatcher(ast.NodeTransformer):
    def __init__(self, typist_id: int):
        self.typist_id = typist_id

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if isinstance(node.iter, ast.Name):
            list_name = node.iter.id
            is_mutating = False
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name) and child.func.value.id == list_name:
                        if child.func.attr in ['remove', 'append', 'pop', 'clear', 'extend']:
                            is_mutating = True
                            break
            if is_mutating:
                node.iter = ast.Subscript(
                    value=ast.Name(id=list_name, ctx=ast.Load()),
                    slice=ast.Slice(lower=None, upper=None, step=None),
                    ctx=ast.Load()
                )
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        
        # Detect: missing recursion base case
        func_name = node.name
        is_recursive = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == func_name:
                is_recursive = True
                break
                
        if is_recursive:
            has_base_case = any(isinstance(stmt, ast.If) for stmt in node.body)
            if not has_base_case and node.args.args:
                first_arg = node.args.args[0].arg
                base_case = ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=first_arg, ctx=ast.Load()),
                        ops=[ast.LtE()],
                        comparators=[ast.Constant(value=1)]
                    ),
                    body=[ast.Return(value=ast.Constant(value=1))],
                    orelse=[]
                )
                node.body.insert(0, base_case)
                
        return node

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        
        # Detect: off-by-one in list indexing - return lst[len(lst)]
        if isinstance(node.slice, ast.Call) and isinstance(node.slice.func, ast.Name) and node.slice.func.id == 'len':
            if len(node.slice.args) == 1 and isinstance(node.slice.args[0], ast.Name) and isinstance(node.value, ast.Name):
                if node.slice.args[0].id == node.value.id:
                    node.slice = ast.BinOp(
                        left=node.slice,
                        op=ast.Sub(),
                        right=ast.Constant(value=1)
                    )
        return node

    def visit_Return(self, node: ast.Return):
        self.generic_visit(node)
        
        # Detect: attribute access chains with no None guard
        if node.value is None:
            return node
            
        bases = set()
        for child in ast.walk(node.value):
            if isinstance(child, ast.Attribute):
                curr = child
                while isinstance(curr, ast.Attribute) or isinstance(curr, ast.Call):
                    if isinstance(curr, ast.Call):
                        curr = curr.func
                    elif isinstance(curr, ast.Attribute):
                        curr = curr.value
                if isinstance(curr, ast.Name):
                    bases.add(curr.id)
                    
        if bases:
            base_var = list(bases)[0]
            guard = ast.If(
                test=ast.Compare(
                    left=ast.Name(id=base_var, ctx=ast.Load()),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)]
                ),
                body=[node],
                orelse=[]
            )
            return guard
            
        return node

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        
        # Detect: infinite loops
        loop_vars = set()
        for child in ast.walk(node.test):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                loop_vars.add(child.id)
                
        is_mutated = False
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id in loop_vars:
                            is_mutated = True
                elif isinstance(child, ast.AugAssign):
                    if isinstance(child.target, ast.Name) and child.target.id in loop_vars:
                        is_mutated = True
                        
        if loop_vars and not is_mutated:
            var_name = list(loop_vars)[0]
            decay_rate = (self.typist_id % 5) + 1
            
            patch_stmt = ast.AugAssign(
                target=ast.Name(id=var_name, ctx=ast.Store()),
                op=ast.Sub(),
                value=ast.Constant(value=decay_rate)
            )
            
            new_body = []
            for stmt in node.body:
                if isinstance(stmt, ast.Pass):
                    new_body.append(patch_stmt)
                new_body.append(stmt)
                
            if not any(isinstance(stmt, ast.Pass) for stmt in node.body):
                new_body.append(patch_stmt)
                
            node.body = new_body
            
        return node


class LLM_Typist:
    def __init__(self, typist_id: int):
        self.typist_id = typist_id

    def _rule_based_fallback(self, base_code: str) -> str:
        try:
            tree = ast.parse(base_code)
            patcher = RuleBasedASTPatcher(self.typist_id)
            patched_tree = patcher.visit(tree)
            ast.fix_missing_locations(patched_tree)
            return ast.unparse(patched_tree)
        except Exception:
            return base_code

    async def rewrite_ast(self, client: httpx.AsyncClient, base_code: str, attention_mask: str) -> str:
        prompt = (
            "You are a localized semantic typist within an autonomous swarm.\n"
            "Correct the structural faults within the provided code.\n"
            f"GNN Topological Attention Mask:\n{attention_mask}\n\n"
            f"Original Source Code:\n{base_code}\n\n"
            "Output solely the raw valid Python code without explanatory text."
        )
        
        try:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "swarm-typist-v1",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "n": 1
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.ReadTimeout, httpx.RequestError, Exception):
            return self._rule_based_fallback(base_code)
