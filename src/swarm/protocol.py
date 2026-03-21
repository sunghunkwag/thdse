"""VSA2VSA Communication Protocol — Data structures and serialization for
inter-agent communication via raw FHRR phase arrays.

No text. No parsing. Only phase arrays (List[float]) and minimal metadata.
The JSON header is metadata only; the float payload uses struct.pack for
exact precision preservation.
"""

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PhaseMessage:
    """A single message in the VSA2VSA protocol."""
    sender_id: int                    # Agent index
    message_type: str                 # "candidate" | "wall" | "target" | "ack"
    phases: List[float]               # The phase array (d floats)
    metadata: Dict[str, Any]          # fitness, entropy, source_context, etc.
    timestamp: int                    # Monotonic counter (not wall clock)


@dataclass
class SwarmConfig:
    """Configuration for a swarm run."""
    n_agents: int                     # Number of agents
    dimension: int                    # FHRR dimension (all agents share this)
    arena_capacity: int               # Per-agent arena capacity
    consensus_threshold: float        # rho threshold for clique consensus (default: 0.85)
    fitness_threshold: float          # Minimum fitness for vocab expansion (default: 0.4)
    max_rounds: int                   # Maximum orchestration rounds
    stagnation_limit: int             # Consecutive zero-progress rounds before halt
    corpus_paths: List[List[str]]     # Per-agent corpus directory paths
    # corpus_paths[i] = list of directories for agent i
    # MUST have len(corpus_paths) == n_agents
    # Each agent's corpus MUST be different


# ── Serialization ────────────────────────────────────────────────

# Wire format:
#   [4 bytes: header_len (uint32 big-endian)]
#   [header_len bytes: JSON header (UTF-8)]
#   [remaining bytes: float32 array via struct.pack]

def serialize_message(msg: PhaseMessage) -> bytes:
    """Serialize a PhaseMessage to bytes for IPC.

    Format: JSON header (message_type, sender_id, metadata, timestamp)
    followed by raw float32 array for phases.

    The phases are raw floats packed via struct, not encoded as strings.
    The JSON header is minimal metadata only.
    """
    header = json.dumps({
        "sender_id": msg.sender_id,
        "message_type": msg.message_type,
        "metadata": msg.metadata,
        "timestamp": msg.timestamp,
    }, separators=(",", ":")).encode("utf-8")

    header_len = len(header)
    # Pack: 4-byte header length + header + float32 array
    phases_packed = struct.pack(f">{len(msg.phases)}f", *msg.phases)
    return struct.pack(">I", header_len) + header + phases_packed


def deserialize_message(data: bytes) -> PhaseMessage:
    """Deserialize bytes back to PhaseMessage."""
    if len(data) < 4:
        raise ValueError("Message too short: missing header length")

    header_len = struct.unpack(">I", data[:4])[0]
    if len(data) < 4 + header_len:
        raise ValueError("Message truncated: header incomplete")

    header_bytes = data[4:4 + header_len]
    header = json.loads(header_bytes.decode("utf-8"))

    phases_bytes = data[4 + header_len:]
    n_floats = len(phases_bytes) // 4
    if n_floats > 0:
        phases = list(struct.unpack(f">{n_floats}f", phases_bytes))
    else:
        phases = []

    return PhaseMessage(
        sender_id=header["sender_id"],
        message_type=header["message_type"],
        phases=phases,
        metadata=header.get("metadata", {}),
        timestamp=header["timestamp"],
    )
