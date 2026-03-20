import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import hdc_core
except ImportError:
    print("Warning: hdc_core not found. Ensure the Rust module is compiled and available in PYTHONPATH.")
    hdc_core = None

from src.topology.encoder_v2 import TopologicalEncoderV2
from src.axioms.engine_v2 import AutoAxiomatizationEngineV2
from src.prophet.gnn_analyzer import TheProphet
from src.swarm.orchestrator_v2 import ResonanceGuidedOrchestrator

# ── Known bug pairs for axiom seeding ────────────────────────────

SEED_PAIRS = [
    # (buggy, fixed) — infinite loop
    (
        """\
def process(data):
    x = 10
    while x > 0:
        pass
    return x
""",
        """\
def process(data):
    x = 10
    while x > 0:
        x -= 1
    return x
""",
    ),
    # (buggy, fixed) — off-by-one
    (
        """\
def get_last(lst):
    return lst[len(lst)]
""",
        """\
def get_last(lst):
    return lst[len(lst) - 1]
""",
    ),
    # (buggy, fixed) — None dereference
    (
        """\
def get_name(user):
    return user.name.upper()
""",
        """\
def get_name(user):
    if user is not None:
        return user.name.upper()
    return None
""",
    ),
]


async def bootstrap_thdse():
    if hdc_core is None:
        print("Fatal: hdc_core compilation barrier reached.")
        return

    arena = hdc_core.FhrrArena(100000, 10000)
    encoder = TopologicalEncoderV2(arena, 10000)
    axiom_engine = AutoAxiomatizationEngineV2(encoder)
    prophet = TheProphet(in_channels=16, hidden_channels=32)

    # Seed axiom DB with known bug patterns
    print("Seeding axiom DB with known bug pairs...")
    added = axiom_engine.ingest_batch(SEED_PAIRS)
    print(f"  {added} axiom(s) registered from {len(SEED_PAIRS)} pairs.")
    counts = axiom_engine.db.count_by_source()
    print(f"  Axiom sources: {counts}")

    orchestrator = ResonanceGuidedOrchestrator(encoder, axiom_engine, prophet)

    target_code = """\
def process(data):
    x = 10
    while x > 0:
        pass
    return x
"""

    print("Executing Resonance-Guided Autophagy Epoch...")
    try:
        resolved_code = await orchestrator.execute_pipeline(target_code)
        print("Synthesis complete:\n")
        print(resolved_code)
    except Exception as e:
        print(f"Pipeline interruption: {e}")


if __name__ == "__main__":
    asyncio.run(bootstrap_thdse())
