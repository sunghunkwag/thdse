"""
V2 Benchmark: Tests the Resonance-Guided Orchestrator against
three canonical Python bug patterns using the multi-layer
topological encoder (AST + CFG + Data-Dep).
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import hdc_core
from src.topology.encoder_v2 import TopologicalEncoderV2
from src.axioms.engine_v2 import AutoAxiomatizationEngineV2
from src.prophet.gnn_analyzer import TheProphet
from src.swarm.orchestrator_v2 import ResonanceGuidedOrchestrator

DIMENSION = 1024  # Compact dimension for V2 multi-layer graph overhead
CAPACITY = 50000

async def run():
    cases = [
        ("Test 1 - Off-by-one",
         "\ndef get_last(lst):\n    return lst[len(lst)]\n"),
        ("Test 2 - None Reference",
         "\ndef get_name(user):\n    return user.name.upper()\n"),
        ("Test 3 - Missing Recursion Base Case",
         "\ndef factorial(n):\n    return n * factorial(n - 1)\n"),
    ]

    for name, code in cases:
        print(f"=== {name} ===")
        arena = hdc_core.FhrrArena(CAPACITY, DIMENSION)
        encoder = TopologicalEncoderV2(arena, DIMENSION)
        axiom_engine = AutoAxiomatizationEngineV2(encoder)
        prophet = TheProphet(in_channels=16, hidden_channels=32)
        orchestrator = ResonanceGuidedOrchestrator(
            encoder, axiom_engine, prophet
        )
        try:
            res = await orchestrator.execute_pipeline(code, n_typists=20)
            print("Result:\n" + res.strip())
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    asyncio.run(run())
