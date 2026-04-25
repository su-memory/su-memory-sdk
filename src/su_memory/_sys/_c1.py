"""
SemanticCategory Module - Memory Semantic Classification

Provides semantic categorization for memory items based on:
- Category classification
- Direction/orientation
- Nature/character
- Energy type mapping
"""

from enum import Enum
from typing import Dict, List


class SemanticCategory(Enum):
    """Eight semantic categories for memory classification"""
    CREATIVE   = ("creative", "metal", "northwest", "assertive", "authority")
    LAKE       = ("lake", "metal", "west", "joyful", "exchange")
    LIGHT      = ("light", "fire", "south", "bright", "knowledge")
    THUNDER    = ("thunder", "wood", "east", "dynamic", "trigger")
    WIND       = ("wind", "wood", "southeast", "penetrating", "connection")
    ABYSS      = ("abyss", "water", "north", "challenging", "risk")
    MOUNTAIN   = ("mountain", "earth", "northeast", "stable", "boundary")
    RECEPTIVE  = ("receptive", "earth", "southwest", "supportive", "foundation")

    def __init__(self, name: str, energy: str, direction: str, nature: str, character: str):
        self.name = name
        self.energy = energy
        self.direction = direction
        self.nature = nature
        self.character = character

    @classmethod
    def from_identifier(cls, identifier: str) -> 'SemanticCategory':
        """Get category from identifier"""
        for cat in cls:
            if identifier in (cat.name, cat.energy, cat.direction):
                return cat
        raise ValueError(f"Unknown category identifier: {identifier}")

    def get_info(self) -> Dict[str, str]:
        """Get all category information"""
        return {
            "name": self.name,
            "energy": self.energy,
            "direction": self.direction,
            "nature": self.nature,
            "character": self.character,
        }


# Category anchor texts for semantic matching
CATEGORY_ANCHORS: Dict[str, str] = {
    "creative": "authority rules leadership decision command system order",
    "lake": "joy satisfaction preference choice happiness reward exchange",
    "light": "knowledge understanding culture education research wisdom文明",
    "thunder": "change action event sudden dynamic start emergency",
    "wind": "relationship connection network communication transfer influence",
    "abyss": "difficulty risk danger problem pressure challenge crisis",
    "mountain": "goal boundary stop stable limit persistence direction",
    "receptive": "background basis environment support包容 condition state",
}

# Keyword mappings for fallback scoring
KEYWORDS_TO_CATEGORY: Dict[str, List[str]] = {
    "creative": ["authority", "system", "rule", "lead", "decide", "command"],
    "lake": ["like", "satisfied", "happy", "preference", "choice", "reward"],
    "light": ["know", "understand", "believe", "knowledge", "culture", "research"],
    "thunder": ["happen", "sudden", "change", "event", "dynamic", "start"],
    "wind": ["relationship", "connection", "friend", "family", "team", "network"],
    "abyss": ["problem", "danger", "difficulty", "failure", "risk", "pressure"],
    "mountain": ["goal", "plan", "stop", "limit", "purpose", "future", "direction"],
    "receptive": ["background", "basis", "environment", "condition", "fact", "state"],
}

# Memory type to category mapping
MEMORY_TYPE_TO_CATEGORY: Dict[str, SemanticCategory] = {
    "fact": SemanticCategory.CREATIVE,
    "preference": SemanticCategory.LAKE,
    "event": SemanticCategory.THUNDER,
    "relationship": SemanticCategory.WIND,
    "knowledge": SemanticCategory.LIGHT,
    "danger": SemanticCategory.ABYSS,
    "goal": SemanticCategory.MOUNTAIN,
    "background": SemanticCategory.RECEPTIVE,
}

# Category associations
CATEGORY_ASSOCIATIONS: Dict[str, List[str]] = {
    "creative": ["authority", "lead", "system", "rule", "metal", "northwest"],
    "lake": ["exchange", "youth", "communication", "metal", "west"],
    "light": ["knowledge", "culture", "fire", "south", "wisdom"],
    "thunder": ["change", "dynamic", "wood", "east", "trigger"],
    "wind": ["connection", "network", "wood", "southeast", "communication"],
    "abyss": ["risk", "water", "north", "challenge", "crisis"],
    "mountain": ["boundary", "stability", "earth", "northeast", "limit"],
    "receptive": ["foundation", "support", "earth", "southwest", "包容"],
}

# Energy to category mapping
ENERGY_TO_CATEGORY: Dict[str, List[SemanticCategory]] = {
    "metal": [SemanticCategory.CREATIVE, SemanticCategory.LAKE],
    "fire": [SemanticCategory.LIGHT],
    "wood": [SemanticCategory.THUNDER, SemanticCategory.WIND],
    "water": [SemanticCategory.ABYSS],
    "earth": [SemanticCategory.MOUNTAIN, SemanticCategory.RECEPTIVE],
}
