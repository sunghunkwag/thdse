import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import hdc_core
except ImportError:
    print("Warning: hdc_core not found. Ensure the Rust module is compiled and available in PYTHONPATH.")
    hdc_core = None

from src.topology.encoder import TopologicalEncoder
from src.axioms.engine import AutoAxiomatizationEngine
from src.prophet.gnn_analyzer import TheProphet
from src.swarm.orchestrator import AutophagicSwarmOrchestrator

async def bootstrap_thdse():
    if hdc_core is None:
        print("Fatal: hdc_core compilation barrier reached.")
        return

    arena = hdc_core.FhrrArena(2000, 10000)
    encoder = TopologicalEncoder(arena, 10000)
    axiom_engine = AutoAxiomatizationEngine(encoder)
    prophet = TheProphet(in_channels=16, hidden_channels=32)
    
    orchestrator = AutophagicSwarmOrchestrator(encoder, axiom_engine, prophet)
    
    target_code = """
def process(data):
    x = 10
    while x > 0:
        pass
    return x
"""

    print("Executing Topological Autophagy Epoch...")
    try:
        resolved_code = await orchestrator.execute_pipeline(target_code)
        print("Mathematical Synthesis State Reached:\n")
        print(resolved_code)
    except Exception as e:
        print(f"Pipeline Interruption: {e}")

if __name__ == "__main__":
    asyncio.run(bootstrap_thdse())
