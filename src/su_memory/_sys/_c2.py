"""
Energy Module - Energy State and Dynamics

Provides energy type classification and state management:
- Five energy types (wood, fire, earth, metal, water)
- Enhancement and suppression relationships
- State strength modifiers
"""

from enum import Enum
from typing import Dict, Tuple
from dataclasses import dataclass


class EnergyType(Enum):
    """Five energy types"""
    WOOD = 0
    FIRE = 1
    EARTH = 2
    METAL = 3
    WATER = 4

    @property
    def element(self) -> str:
        return ("wood", "fire", "earth", "metal", "water")[self.value]

    @property
    def nature(self) -> str:
        return ("growth", "warmth", "bearing", "contraction", "moisture")[self.value]

    @property
    def movement(self) -> str:
        return ("ascending", "light", "transformation", "cleansing", "descending")[self.value]

    @property
    def direction(self) -> str:
        return ("east", "south", "center", "west", "north")[self.value]

    @property
    def season(self) -> str:
        return ("spring", "summer", "late_summer", "autumn", "winter")[self.value]


# Alias for backward compatibility
Wuxing = EnergyType


# Enhancement sequence: key enhances value
ENERGY_ENHANCE_MAP: Dict[EnergyType, EnergyType] = {
    EnergyType.WOOD: EnergyType.FIRE,
    EnergyType.FIRE: EnergyType.EARTH,
    EnergyType.EARTH: EnergyType.METAL,
    EnergyType.METAL: EnergyType.WATER,
    EnergyType.WATER: EnergyType.WOOD,
}

# Alias for backward compatibility
WUXING_SHENG = ENERGY_ENHANCE_MAP

# String version for cross-module compatibility
ENERGY_ENHANCE: Dict[str, str] = {
    "wood": "fire", "fire": "earth", "earth": "metal", "metal": "water", "water": "wood",
}


# Suppression sequence: key suppresses value
ENERGY_SUPPRESS_MAP: Dict[EnergyType, EnergyType] = {
    EnergyType.WOOD: EnergyType.EARTH,
    EnergyType.EARTH: EnergyType.WATER,
    EnergyType.WATER: EnergyType.FIRE,
    EnergyType.FIRE: EnergyType.METAL,
    EnergyType.METAL: EnergyType.WOOD,
}

# Alias for backward compatibility
WUXING_KE = ENERGY_SUPPRESS_MAP

# String version for cross-module compatibility
ENERGY_SUPPRESS: Dict[str, str] = {
    "wood": "earth", "earth": "water", "water": "fire", "fire": "metal", "metal": "wood",
}


# ========================
# State Strength Modifiers
# ========================

STATE_STRENGTH_MAP = {
    "strong": 2.0,    # Strong state
    "balanced": 1.3,   # Balanced state
    "rested": 1.0,    # Rested state
    "restrained": 0.5, # Restrained state
    "declined": 0.3,  # Declined state
}

# Chinese aliases for compatibility
WUXING_STATE_MULTIPLIERS = {
    "旺": 2.0,
    "相": 1.3,
    "休": 1.0,
    "囚": 0.5,
    "死": 0.3,
}


def _get_enhancer(target: EnergyType) -> EnergyType:
    """Find the energy type that enhances target"""
    for k, v in ENERGY_ENHANCE_MAP.items():
        if v == target:
            return k
    return target


def get_energy_state(target: EnergyType, current_season: EnergyType) -> Tuple[str, float]:
    """
    Get energy state based on current season

    Args:
        target: Target energy type
        current_season: Current season energy type

    Returns:
        (state_name, strength_multiplier)
    """
    if target == current_season:
        return "strong", 2.0
    if ENERGY_ENHANCE_MAP.get(current_season) == target:
        return "balanced", 1.3
    if _get_enhancer(current_season) == target:
        return "rested", 1.0
    if ENERGY_SUPPRESS_MAP.get(target) == current_season:
        return "restrained", 0.5
    if ENERGY_SUPPRESS_MAP.get(current_season) == target:
        return "declined", 0.3
    return "rested", 1.0


def check_state_interaction(attacker: EnergyType, defender: EnergyType,
                            attacker_intensity: float, defender_intensity: float) -> str:
    """
    Check energy state interaction
    Returns: "normal" | "overwhelming" | "counter"
    """
    if ENERGY_SUPPRESS_MAP.get(attacker) != defender:
        return "normal"
    if defender_intensity <= 0:
        return "overwhelming"
    if attacker_intensity / defender_intensity > 2.0:
        return "overwhelming"
    if attacker_intensity <= 0:
        return "counter"
    if defender_intensity / attacker_intensity > 2.0:
        return "counter"
    return "normal"


def energy_similarity(e1: EnergyType, e2: EnergyType) -> float:
    """
    Calculate similarity between two energy types (0.0~1.0)
    - Same type: 1.0
    - Enhancement: 0.7
    - Neutral: 0.3
    - Suppression: 0.1
    """
    if e1 == e2:
        return 1.0
    if ENERGY_ENHANCE_MAP.get(e1) == e2 or ENERGY_ENHANCE_MAP.get(e2) == e1:
        return 0.7
    if ENERGY_SUPPRESS_MAP.get(e1) == e2 or ENERGY_SUPPRESS_MAP.get(e2) == e1:
        return 0.1
    return 0.3


@dataclass
class EnergyState:
    energy_type: EnergyType
    intensity: float = 1.0
    status: str = "balanced"

    def get_effective_intensity(self, environment: 'EnergyState' = None) -> float:
        if environment is None:
            return self.intensity
        state_name, multiplier = get_energy_state(self.energy_type, environment.energy_type)
        self.status = state_name
        return self.intensity * multiplier


class EnergyNetwork:
    def __init__(self):
        self.memory_states: Dict[str, EnergyState] = {}

    def register_memory(self, memory_id: str, energy_type: EnergyType) -> None:
        self.memory_states[memory_id] = EnergyState(energy_type=energy_type)

    def propagate_energy(self, source_id: str, delta: float) -> None:
        source = self.memory_states.get(source_id)
        if not source:
            return
        target_energy = ENERGY_ENHANCE_MAP.get(source.energy_type)
        if not target_energy:
            return
        for mem_id, state in self.memory_states.items():
            if state.energy_type == target_energy:
                state.intensity += delta

    def get_dominant_energy(self) -> EnergyType:
        if not self.memory_states:
            return EnergyType.EARTH
        counts: Dict[EnergyType, float] = {}
        for state in self.memory_states.values():
            e = state.energy_type
            counts[e] = counts.get(e, 0) + state.intensity
        return max(counts, key=counts.get)


def energy_from_category(category_name: str) -> EnergyType:
    """Map category name to energy type"""
    CATEGORY_ENERGY_MAP = {
        "creative": "metal", "lake": "metal",
        "light": "fire", "thunder": "wood", "wind": "wood",
        "abyss": "water", "mountain": "earth", "receptive": "earth",
    }
    energy_name = CATEGORY_ENERGY_MAP.get(category_name, "earth")
    for e in EnergyType:
        if e.element == energy_name:
            return e
    return EnergyType.EARTH


# Backward compatibility aliases
wuxing_from_bagua = energy_from_category
