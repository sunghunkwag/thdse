import httpx
import ast

class InfiniteLoopPatcher(ast.NodeTransformer):
    def __init__(self, typist_id: int):
        self.typist_id = typist_id

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        
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
            patcher = InfiniteLoopPatcher(self.typist_id)
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
