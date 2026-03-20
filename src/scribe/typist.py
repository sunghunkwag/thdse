import httpx

class LLM_Typist:
    def __init__(self, typist_id: int):
        self.typist_id = typist_id

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
        except httpx.ReadTimeout:
            return base_code + f"\n# Timeout recovery injection typing sub-agent {self.typist_id}\n"
        except Exception:
            return base_code
