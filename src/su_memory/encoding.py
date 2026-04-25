"""
Memory Encoding Type Definitions
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class MemoryEncoding:
    """Memory encoding with semantic attributes"""
    category: str      # Semantic category
    energy: str        # Energy type (wood/fire/earth/metal/water)
    pattern: int       # Pattern index
    intensity: float   # Energy intensity
    time_stem: str    # Time stem (10-element cycle)
    time_branch: str  # Time branch (12-element cycle)
    causal_depth: int # Causal chain depth
    # Extended semantic fields
    pattern_name: str = ""
    category_probs: Optional[Dict[str, float]] = None
    energy_scores: Optional[Dict[str, float]] = None
