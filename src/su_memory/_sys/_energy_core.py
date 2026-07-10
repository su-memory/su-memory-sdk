"""
Five Elements Energy Core Engine

This module implements the core energy system for the Human Layer (Ren Ceng),
providing comprehensive functionality for:
- Relationship calculation (category interaction)
- Strength state determination (旺相休囚死)
- Energy pattern analysis (格局分析)
- Balance regulation rules (制化规则)
- Energy flow simulation (能量流转)

Architecture: Human Layer (Ren) - Energy System
"""

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

# Import from causal.py for compatibility
# ============================================================
# Data Structures
# ============================================================
import numpy as np

from su_memory.algebra.affinity import AffinityMatrix

# Import enums from _enums.py
from ._enums import EnergyPattern, EnergyRelation, EnergyType, StrengthState

# Import energy mappings from _terms.py
from ._terms import (

    ENERGY_COLOR,
    ENERGY_DIRECTION,
    ENERGY_EMOTION,
    ENERGY_ENHANCE,
    ENERGY_INDUSTRY,
    ENERGY_ORGAN,
    ENERGY_SEASON,
    ENERGY_SUPPRESS,
    ENERGY_TASTE,
)


@dataclass
class EnergyState:
    """
    Energy state representation.

    Attributes:
        energy_type: The type of energy (wood, fire, earth, metal, water)
        strength: The strength state (WANG/XIANG/XIU/QIU/SI)
        intensity: Energy intensity value (0.0 - 1.0)
    """
    energy_type: EnergyType
    strength: StrengthState
    intensity: float

    @property
    def is_enhanced(self) -> bool:
        """Check if the energy is in an enhanced state."""
        return self.strength in [StrengthState.WANG, StrengthState.XIANG]

    def __repr__(self) -> str:
        return f"EnergyState({self.energy_type.name}, {self.strength.name}, {self.intensity:.2f})"


@dataclass
class EnergyBalanceResult:
    """
    Energy balance analysis result.

    Attributes:
        status: Balance status ("balanced" or "imbalanced")
        pattern: The detected energy pattern type
        ratios: Dictionary of energy ratios (sum to 1.0)
        dominant: The dominant energy type
        suggestions: List of adjustment suggestions
    """
    status: str
    pattern: EnergyPattern
    ratios: dict[str, float]
    dominant: str
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "pattern": self.pattern.name,
            "ratios": self.ratios,
            "dominant": self.dominant,
            "suggestions": self.suggestions,
        }


@dataclass
class EnergyFlow:
    """
    Energy flow representation between energy types.

    Attributes:
        source: Source energy type
        target: Target energy type
        relation: Type of relation (ENHANCE/SUPPRESS/OVERCONSTRAINT/REVERSE)
        intensity: Flow intensity (0.0 - 1.0)
    """
    source: EnergyType
    target: EnergyType
    relation: EnergyRelation
    intensity: float

    def __repr__(self) -> str:
        return f"EnergyFlow({self.source.name} -> {self.target.name}, {self.relation.name}, {self.intensity:.2f})"


# ============================================================
# Energy Core Engine
# ============================================================

class EnergyCore:
    """
    Five Elements Energy Core Engine.

    This class provides comprehensive functionality for the Five Elements
    energy system used in the Human Layer of the semantic memory framework.

    Features:
        - Relationship calculation (category interaction)
        - Strength state determination (旺相休囚死)
        - Energy pattern analysis (格局分析)
        - Balance regulation (制化规则)
        - Energy flow simulation (能量流转)

    Example:
        >>> ec = EnergyCore()
        >>> state = ec.get_energy_state("semantic", 2)  # 寅月
        >>> print(state.strength)  # StrengthState.WANG
        >>> ec.get_enhance_relation("semantic", "causal")  # True
    """

    # Five element order for cycle operations
    ENERGY_ORDER = ["semantic", "causal", "spacetime", "generative", "trust"]
    ENERGY_NAMES = {
        "semantic": "Semantic (语义维度)",
        "causal": "Causal (因果维度)",
        "spacetime": "Spacetime (时空维度)",
        "generative": "Generative (生成维度)",
        "trust": "Trust (信任维度)",
    }

    # Monthly strength table (旺相休囚死)
    # Key: branch index (0-11), Value: [木, 火, 土, 金, 水] strength states
    MONTHLY_STRENGTH: dict[int, list[StrengthState]] = {
        # 子 (0) - water: water WANG, wood XIANG, fire QIU, earth SI, metal XIU
        0: [StrengthState.XIANG, StrengthState.XIU, StrengthState.SI, StrengthState.XIU, StrengthState.WANG],
        # 丑 (1) - earth: earth XIANG
        1: [StrengthState.XIU, StrengthState.XIANG, StrengthState.XIANG, StrengthState.XIU, StrengthState.XIANG],
        # 寅 (2) - wood: wood WANG
        2: [StrengthState.WANG, StrengthState.XIANG, StrengthState.XIU, StrengthState.QIU, StrengthState.SI],
        # 卯 (3) - wood: wood WANG
        3: [StrengthState.WANG, StrengthState.XIANG, StrengthState.XIU, StrengthState.QIU, StrengthState.SI],
        # 辰 (4) - earth: earth WANG
        4: [StrengthState.XIU, StrengthState.XIU, StrengthState.WANG, StrengthState.XIU, StrengthState.QIU],
        # 巳 (5) - fire: fire WANG
        5: [StrengthState.SI, StrengthState.WANG, StrengthState.XIANG, StrengthState.XIU, StrengthState.QIU],
        # 午 (6) - fire: fire WANG
        6: [StrengthState.SI, StrengthState.WANG, StrengthState.XIANG, StrengthState.XIU, StrengthState.QIU],
        # 未 (7) - earth: earth WANG
        7: [StrengthState.XIU, StrengthState.XIU, StrengthState.WANG, StrengthState.XIU, StrengthState.QIU],
        # 申 (8) - metal: metal WANG
        8: [StrengthState.QIU, StrengthState.SI, StrengthState.XIU, StrengthState.WANG, StrengthState.XIANG],
        # 酉 (9) - metal: metal WANG
        9: [StrengthState.QIU, StrengthState.SI, StrengthState.XIU, StrengthState.WANG, StrengthState.XIANG],
        # 戌 (10) - earth: earth WANG
        10: [StrengthState.XIU, StrengthState.XIU, StrengthState.WANG, StrengthState.XIU, StrengthState.QIU],
        # 亥 (11) - water: water WANG, wood XIANG
        11: [StrengthState.XIANG, StrengthState.QIU, StrengthState.SI, StrengthState.XIU, StrengthState.WANG],
    }

    # Energy strength multipliers
    STRENGTH_MULTIPLIER = {
        StrengthState.WANG: 1.2,   # Strongest
        StrengthState.XIANG: 1.0,  # Balanced
        StrengthState.XIU: 0.8,    # Rested
        StrengthState.QIU: 0.5,    # Confined
        StrengthState.SI: 0.3,     # Weakest
    }

    # Balance thresholds
    BALANCE_THRESHOLD_HIGH = 0.35  # Above this = dominant
    BALANCE_THRESHOLD_LOW = 0.10   # Below this = deficient

    # Energy flow coefficients
    ENHANCE_FLOW_RATE = 0.15       # Enhancement transfer rate
    SUPPRESS_FLOW_RATE = -0.10     # Suppression reduction rate
    OVERCONSTRAINT_RATE = -0.20   # Overconstraint severe reduction
    REVERSE_RATE = 0.08           # Reverse reaction minor enhancement

    def __init__(self):
        """Initialize the Energy Core Engine."""
        # Create reverse mappings for quick lookup
        self._enhance_reverse = {v: k for k, v in ENERGY_ENHANCE.items()}
        self._suppress_reverse = {v: k for k, v in ENERGY_SUPPRESS.items()}

        # Pre-compute mutual relationships
        self._enhance_relations = self._build_bidirectional_enhance()
        self._suppress_relations = self._build_bidirectional_suppress()

        # ── 矩阵化核心：5×5 耦合矩阵 + 守恒流转矩阵 ──
        # 用 AffinityMatrix (algebra 层纯线性代数) 统一能量耦合关系。
        self._affinity = AffinityMatrix(labels=tuple(self.ENERGY_ORDER))
        # 守恒流转矩阵：一步 x_{t+1} = clip(x_t + Flow @ x_t)
        # 守恒条件: 每列和 = 0 (能量不灭)。
        #
        # 生 (enhance) = 纯转移: 源→目标, 源减多少目标加多少。
        #   Flow[j_gen, i] += ENH   Flow[i,i] -= ENH  → 列和贡献: ENH-ENH=0 ✓
        # 克 (suppress) = 转移非消灭: 被克者→克者 (能量不灭)。
        #   Flow[i, j_sup] += SUPP  Flow[j_sup,j_sup] -= SUPP → 列和: SUPP-SUPP=0 ✓
        #
        # 修复历史 bug: 旧版 Flow[i,i] -= ENH*0.5 (半扣, 能量凭空产生)
        #   + Flow[j_sup,j_sup] += SUPPRESS_RATE (负系数作用自身, 能量凭空消灭)
        #   → 列和 = -0.025, 200 步后总能量从 1.0 坍缩到 0.006。
        Flow = np.zeros((5, 5), dtype=np.float64)
        _enh = self.ENHANCE_FLOW_RATE          # 0.15
        _supp = abs(self.SUPPRESS_FLOW_RATE)   # 0.10 (取绝对值, 作为转移量)
        for i in range(5):
            e_src = self.ENERGY_ORDER[i]
            # 生: 源 i → 目标 j_gen (等量转移)
            j_gen = self._energy_index(ENERGY_ENHANCE[e_src])
            Flow[j_gen, i] += _enh
            Flow[i, i]     -= _enh
            # 克: 被克者 j_sup → 克者 i (能量转移)
            j_sup = self._energy_index(ENERGY_SUPPRESS[e_src])
            Flow[i, j_sup]     += _supp
            Flow[j_sup, j_sup] -= _supp
        self._flow_matrix = Flow

        # ENERGY_ORDER -> index 查找表（加速 dict↔vector 转换）
        self._energy_idx = {e: i for i, e in enumerate(self.ENERGY_ORDER)}

    def _build_bidirectional_enhance(self) -> dict[str, str]:
        """Build bidirectional enhance relationship mapping."""
        result = {}
        for src, tgt in ENERGY_ENHANCE.items():
            result[src] = tgt
            result[tgt] = self._enhance_reverse.get(src, "")
        return result

    def _build_bidirectional_suppress(self) -> dict[str, str]:
        """Build bidirectional (mutual) suppress relationship mapping."""
        result = {}
        for src, tgt in ENERGY_SUPPRESS.items():
            result[src] = tgt
            result[tgt] = self._suppress_reverse.get(src, "")
        return result

    def _normalize_energy(self, energy: str) -> str:
        """Normalize energy type to standard form.

        Backward compatibility: old five-element naming (wood/fire/earth/metal/water),
        whether passed as a string *or* as an EnergyType enum member, is mapped to the
        new standard naming (semantic/causal/spacetime/generative/trust).
        """
        # Backward compatibility: old five-element naming → new standard naming
        _old_to_new = {
            "wood": "semantic", "fire": "causal", "earth": "spacetime",
            "metal": "generative", "water": "trust",
        }
        if isinstance(energy, EnergyType):
            normalized = energy.name.lower()
        else:
            normalized = energy.lower()
        return _old_to_new.get(normalized, normalized)

    def _energy_index(self, energy: str) -> int:
        """Get the index of energy type in the cycle (0-4)."""
        energy = self._normalize_energy(energy)
        return self.ENERGY_ORDER.index(energy)

    # ============================================================
    # Relationship Calculation Methods
    # ============================================================

    def get_enhance_relation(self, e1: str, e2: str) -> bool:
        """
        Check if e1 enhances e2 (木生火, 火生土, etc.).

        Args:
            e1: Source energy type
            e2: Target energy type

        Returns:
            True if e1 generates e2, False otherwise

        Example:
            >>> ec.get_enhance_relation("semantic", "causal")
            True
            >>> ec.get_enhance_relation("causal", "semantic")
            False
        """
        e1 = self._normalize_energy(e1)
        e2 = self._normalize_energy(e2)
        return ENERGY_ENHANCE.get(e1) == e2

    def get_suppress_relation(self, e1: str, e2: str) -> bool:
        """
        Check if e1 suppresses e2 (semantic→spacetime, spacetime→trust, etc.).

        Note: Per task requirements, suppression is bidirectional - if e1 suppresses
        e2, then e2 also suppresses e1 (mutual control relationship).

        Args:
            e1: Source energy type
            e2: Target energy type

        Returns:
            True if e1 and e2 have suppression relationship, False otherwise

        Example:
            >>> ec.get_suppress_relation("semantic", "spacetime")
            True
            >>> ec.get_suppress_relation("spacetime", "semantic")  # Bidirectional
            True
        """
        e1 = self._normalize_energy(e1)
        e2 = self._normalize_energy(e2)

        # Bidirectional suppression - if either suppresses the other
        if ENERGY_SUPPRESS.get(e1) == e2 or ENERGY_SUPPRESS.get(e2) == e1:
            return True
        return False

    def get_overconstraint_relation(self, e1: str, e2: str) -> bool:
        """
        Check if e1 over-constrains e2 (相乘: excessive control).

        Overconstraint occurs when:
        - e1 is in WANG (strong) state
        - e1 normally suppresses e2

        Args:
            e1: Source energy type
            e2: Target energy type

        Returns:
            True if e1 over-constrains e2, False otherwise
        """
        e1 = self._normalize_energy(e1)
        e2 = self._normalize_energy(e2)
        return ENERGY_SUPPRESS.get(e1) == e2

    def get_reverse_relation(self, e1: str, e2: str) -> bool:
        """
        Check if e1 reverses against e2 (相侮: reverse control).

        Reverse control occurs when:
        - e1 is weak (SI/QIU state)
        - e2 normally suppresses e1
        - e1 counter-suppresses e2

        Args:
            e1: Source energy type
            e2: Target energy type

        Returns:
            True if e1 reverses against e2, False otherwise
        """
        e1 = self._normalize_energy(e1)
        e2 = self._normalize_energy(e2)
        # Reverse is when the suppressed element counter-attacks
        # In five elements: wood controls earth but earth can reverse against wood
        reverse_pairs = [
            ("spacetime", "semantic"),    # spacetime reverses against semantic
            ("trust", "spacetime"),       # trust reverses against spacetime
            ("causal", "trust"),          # causal reverses against trust
            ("generative", "causal"),     # generative reverses against causal
            ("semantic", "generative"),   # semantic reverses against generative
        ]
        return (e1, e2) in reverse_pairs

    def analyze_interaction(self, e1: str, e2: str) -> list[EnergyRelation]:
        """
        Analyze all interactions between two energy types.

        Args:
            e1: First energy type
            e2: Second energy type

        Returns:
            List of EnergyRelation types between e1 and e2

        Example:
            >>> ec.analyze_interaction("semantic", "causal")
            [EnergyRelation.ENHANCE]
        """
        e1 = self._normalize_energy(e1)
        e2 = self._normalize_energy(e2)

        relations = []

        if e1 == e2:
            relations.append(EnergyRelation.SAME)
            return relations

        # Check enhancement
        if self.get_enhance_relation(e1, e2):
            relations.append(EnergyRelation.ENHANCE)
        elif self.get_enhance_relation(e2, e1):
            relations.append(EnergyRelation.ENHANCE)

        # Check suppression
        if self.get_suppress_relation(e1, e2):
            relations.append(EnergyRelation.SUPPRESS)
        if self.get_suppress_relation(e2, e1):
            relations.append(EnergyRelation.SUPPRESS)

        # Check reverse (相侮)
        if self.get_reverse_relation(e1, e2):
            relations.append(EnergyRelation.REVERSE)
        if self.get_reverse_relation(e2, e1):
            relations.append(EnergyRelation.REVERSE)

        return relations if relations else [EnergyRelation.SAME]

    # ============================================================
    # Strength State Methods
    # ============================================================

    def get_energy_state(
        self,
        energy_type: str,
        month_branch: int
    ) -> EnergyState:
        """
        Get the energy state for a given energy type and month.

        Args:
            energy_type: The energy type (wood, fire, earth, metal, water)
            month_branch: The month branch index (0-11, corresponding to Earthly Branches)

        Returns:
            EnergyState with strength and intensity values

        Example:
            >>> state = ec.get_energy_state("semantic", 2)  # 寅月
            >>> state.strength
            <StrengthState.WANG: 0>
            >>> state.intensity
            1.2
        """
        energy_type = self._normalize_energy(energy_type)

        # Validate month branch
        if month_branch not in self.MONTHLY_STRENGTH:
            raise ValueError(f"Invalid month branch: {month_branch}. Must be 0-11.")

        # Get energy index
        idx = self._energy_index(energy_type)

        # Get strength state from monthly table
        strength = self.MONTHLY_STRENGTH[month_branch][idx]

        # Calculate intensity
        intensity = self.STRENGTH_MULTIPLIER.get(strength, 1.0)

        # Map to EnergyType enum (idx matches EnergyType enum order)
        return EnergyState(
            energy_type=EnergyType(idx),
            strength=strength,
            intensity=intensity
        )

    def get_strength_from_branch(self, branch: int) -> dict[str, StrengthState]:
        """
        Get strength states for all energy types in a given month.

        Args:
            branch: The month branch index (0-11)

        Returns:
            Dictionary mapping energy types to their strength states

        Example:
            >>> ec.get_strength_from_branch(2)  # 寅月
            {'wood': <StrengthState.WANG>, 'fire': <StrengthState.XIANG>, ...}
        """
        if branch not in self.MONTHLY_STRENGTH:
            raise ValueError(f"Invalid branch: {branch}. Must be 0-11.")

        strengths = self.MONTHLY_STRENGTH[branch]
        return {
            self.ENERGY_ORDER[i]: strengths[i]
            for i in range(len(self.ENERGY_ORDER))
        }

    # ============================================================
    # Balance Analysis Methods
    # ============================================================

    def analyze_balance(self, energies: dict[str, float]) -> EnergyBalanceResult:
        """
        Analyze the energy balance state.

        Args:
            energies: Dictionary of energy types and their values (should sum to 1.0)

        Returns:
            EnergyBalanceResult with analysis details

        Example:
            >>> energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
            >>> result = ec.analyze_balance(energies)
            >>> result.status
            'balanced'
        """
        # Normalize energies
        total = sum(energies.values())
        if total == 0:
            raise ValueError("Energy values cannot all be zero.")

        ratios = {k: v / total for k, v in energies.items()}

        # Find dominant energy
        dominant = max(ratios, key=ratios.get)
        dominant_ratio = ratios[dominant]

        # Determine balance status
        if dominant_ratio > self.BALANCE_THRESHOLD_HIGH:
            status = "imbalanced"
        elif dominant_ratio > 0.25:  # More than one dominant
            status = "balanced"
        else:
            status = "balanced"

        # Determine pattern
        pattern = self._determine_pattern(ratios)

        # Generate suggestions
        suggestions = self._generate_balance_suggestions(ratios, dominant)

        return EnergyBalanceResult(
            status=status,
            pattern=pattern,
            ratios=ratios,
            dominant=dominant,
            suggestions=suggestions
        )

    def _determine_pattern(self, ratios: dict[str, float]) -> EnergyPattern:
        """Determine the energy pattern based on ratios."""
        max_ratio = max(ratios.values())
        min_ratio = min(ratios.values())

        # Check for dedicated strength (专旺格)
        if max_ratio >= 0.6:
            return EnergyPattern.ZHUAN_WANG

        # Check for following strength (从旺格)
        if min_ratio <= 0.05:
            return EnergyPattern.CONG_WANG

        # Check for reverse pattern (反局格)
        dominant = max(ratios, key=ratios.get)
        dominated = min(ratios, key=ratios.get)

        # If dominant suppresses dominated severely
        if self.get_suppress_relation(dominant, dominated) and ratios[dominant] > 0.35:
            return EnergyPattern.FAN_WANG

        # Check for regulation pattern (制化格)
        if self._has_regulation_potential(ratios):
            return EnergyPattern.ZHI_HUA

        # Default to coordination pattern
        return EnergyPattern.PEI_HE

    def _has_regulation_potential(self, ratios: dict[str, float]) -> bool:
        """Check if the energy distribution has regulation potential."""
        # Check if there are both strong and weak energies that can regulate each other
        for e1 in self.ENERGY_ORDER:
            for e2 in self.ENERGY_ORDER:
                if e1 != e2 and self.get_suppress_relation(e1, e2):
                    if ratios.get(e1, 0) > 0.2 and ratios.get(e2, 0) > 0.2:
                        return True
        return False

    def _generate_balance_suggestions(
        self,
        ratios: dict[str, float],
        dominant: str
    ) -> list[str]:
        """Generate suggestions for energy balance adjustment."""
        suggestions = []

        # Find deficient energies
        deficient = [e for e, r in ratios.items() if r < self.BALANCE_THRESHOLD_LOW]

        if deficient:
            # Suggest enhancing deficient energies
            for e in deficient:
                generator = self._enhance_reverse.get(e)
                if generator:
                    suggestions.append(
                        f"建议增强{e}能量，通过{generator}能量转化补充"
                    )

        # Check for over-dominance
        if ratios.get(dominant, 0) > 0.45:
            suppressor = ENERGY_SUPPRESS.get(dominant)
            if suppressor:
                suggestions.append(
                    f"注意{dominant}能量过旺，建议适当增强{suppressor}能量进行制约"
                )

        # General balance suggestions
        if not suggestions:
            suggestions.append("能量分布较为均衡，维持当前状态")

        return suggestions

    # ============================================================
    # Balance Rules Application
    # ============================================================

    def apply_balance_rules(
        self,
        energies: dict[str, float],
        pattern: EnergyPattern
    ) -> dict[str, float]:
        """
        Apply balance rules based on the energy pattern.

        Args:
            energies: Current energy distribution
            pattern: The energy pattern type

        Returns:
            Adjusted energy distribution after applying balance rules
        """
        result = energies.copy()
        total = sum(energies.values())

        if pattern == EnergyPattern.ZHUAN_WANG:
            # Dedicated strength: strengthen the dominant
            # No changes needed
            pass

        elif pattern == EnergyPattern.CONG_WANG:
            # Following strength: follow the strongest, weaken the weakest
            dominant = max(energies, key=energies.get)
            result[dominant] *= 1.2
            total = sum(result.values())

        elif pattern == EnergyPattern.ZHI_HUA:
            # Regulation pattern: apply controlled suppression
            result = self._apply_regulation(result)

        elif pattern == EnergyPattern.FAN_WANG:
            # Reverse pattern: reinforce the suppressed
            result = self._apply_reinforcement(result)

        elif pattern == EnergyPattern.PEI_HE:
            # Coordination pattern: smooth adjustments
            result = self._apply_coordination(result)

        # Normalize to original total
        new_total = sum(result.values())
        if new_total > 0:
            result = {k: v * total / new_total for k, v in result.items()}

        return result

    def _apply_regulation(self, energies: dict[str, float]) -> dict[str, float]:
        """Apply regulation rules for ZHI_HUA pattern（向量化：克制矩阵衰减）。"""
        v = self._to_vector(energies)
        # 构建克制衰减：对每对 (i 克 j)，j -= 0.1 * v[i]
        # suppress 矩阵 S[i,j]=1 表示 i 克 j；衰减 = -0.1 * (S^T @ v) 作用于 j
        S = np.zeros((5, 5), dtype=np.float64)
        for i in range(5):
            j_sup = self._energy_idx[ENERGY_SUPPRESS[self.ENERGY_ORDER[i]]]
            S[i, j_sup] = 1.0
        reduction = 0.1 * (S.T @ v)
        v = np.maximum(v - reduction, 0.0)
        return self._to_dict(v)

    def _apply_reinforcement(self, energies: dict[str, float]) -> dict[str, float]:
        """Apply reinforcement for FAN_WANG pattern（向量化：弱项经生源补给）。"""
        v = self._to_vector(energies)
        # 生源向量：generator[i] = 生成 i 的那个能量的下标
        gen_vec = np.zeros(5, dtype=np.float64)
        for i in range(5):
            g = self._enhance_reverse.get(self.ENERGY_ORDER[i])
            if g:
                gen_vec[i] = v[self._energy_idx[self._normalize_energy(g)]]
        weak_mask = v < self.BALANCE_THRESHOLD_LOW
        v = v + np.where(weak_mask, 0.1 * gen_vec, 0.0)
        return self._to_dict(v)

    def _apply_coordination(self, energies: dict[str, float]) -> dict[str, float]:
        """Apply coordination for PEI_HE pattern（向量化：向均值平滑）。"""
        v = self._to_vector(energies)
        avg = v.mean()
        v = v + (avg - v) * 0.05
        return self._to_dict(v)

    # ============================================================
    # Energy Flow Simulation
    # ============================================================

    def simulate_energy_flow(
        self,
        energies: dict[str, float],
        steps: int = 10
    ) -> list[dict[str, float]]:
        """
        Simulate energy flow over multiple steps.

        Args:
            energies: Initial energy distribution
            steps: Number of simulation steps

        Returns:
            List of energy distributions at each step
        """
        history = [energies.copy()]
        current = energies.copy()

        for _ in range(steps):
            next_state = self._calculate_flow_step(current)
            history.append(next_state)
            current = next_state

        return history

    def _calculate_flow_step(self, energies: dict[str, float]) -> dict[str, float]:
        """Calculate one step of energy flow（矩阵化：x_{t+1} = clip(Flow @ x_t)）。

        向量化的单步流转：把 dict 提升为 5×1 向量，与预构建的转移矩阵
        ``self._flow_matrix`` 做一次矩阵-向量乘，等价于原先对 enhance/suppress
        两轮 dict 循环的结果，但只需 O(25) 而非多次查表。
        """
        v = self._to_vector(energies)
        # 矩阵-向量乘 + 非负裁剪
        nxt = v + self._flow_matrix @ v
        np.maximum(nxt, 0.0, out=nxt)
        return self._to_dict(nxt)

    # ============================================================
    # Matrix-ized internals (向量化)
    # ============================================================

    def _to_vector(self, energies: dict[str, float]) -> np.ndarray:
        """dict -> length-5 numpy 向量（按 ENERGY_ORDER 对齐，缺失补 0）。"""
        v = np.zeros(5, dtype=np.float64)
        for e, val in energies.items():
            idx = self._energy_idx.get(self._normalize_energy(e))
            if idx is not None:
                v[idx] = float(val)
        return v

    def _to_dict(self, v: np.ndarray) -> dict[str, float]:
        """length-5 向量 -> dict（按 ENERGY_ORDER）。"""
        return {self.ENERGY_ORDER[i]: float(v[i]) for i in range(5)}

    def flow_matrix(self) -> np.ndarray:
        """暴露一步流转转移矩阵 Flow（5×5），供外部谱分析/可视化。"""
        return self._flow_matrix.copy()

    def affinity_matrix(self) -> np.ndarray:
        """暴露耦合矩阵 A（5×5 AffinityMatrix.matrix）。"""
        return self._affinity.matrix.copy()

    def stationary_distribution(self) -> np.ndarray:
        """能量系统的平稳分布（AffinityMatrix 主左特征向量）。"""
        return self._affinity.stationary_distribution()

    def balance_deviation(self, energies: dict[str, float]) -> float:
        """分布偏离平稳分布的 L2 距离（越大越失衡）。"""
        v = self._to_vector(energies)
        if v.sum() <= 0:
            return float('inf')
        return self._affinity.balance_deviation(v)

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_energy_attributes(self, energy_type: str) -> dict:
        """
        Get complete attributes for an energy type.

        Args:
            energy_type: The energy type to query

        Returns:
            Dictionary containing all attributes for the energy type
        """
        energy = self._normalize_energy(energy_type)

        return {
            "name": energy,
            "chinese_name": self.ENERGY_NAMES.get(energy, ""),
            "season": ENERGY_SEASON.get(energy, []),
            "direction": ENERGY_DIRECTION.get(energy, []),
            "color": ENERGY_COLOR.get(energy, []),
            "organ": ENERGY_ORGAN.get(energy, ""),
            "taste": ENERGY_TASTE.get(energy, ""),
            "emotion": ENERGY_EMOTION.get(energy, ""),
            "industry": ENERGY_INDUSTRY.get(energy, []),
            "enhances": ENERGY_ENHANCE.get(energy, ""),
            "suppresses": ENERGY_SUPPRESS.get(energy, ""),
            "enhanced_by": self._enhance_reverse.get(energy, ""),
            "suppressed_by": self._suppress_reverse.get(energy, ""),
        }

    def calculate_compatibility(
        self,
        energies1: dict[str, float],
        energies2: dict[str, float]
    ) -> float:
        """
        Calculate compatibility score between two energy distributions.

        Args:
            energies1: First energy distribution
            energies2: Second energy distribution

        Returns:
            Compatibility score (0.0 - 1.0), higher is more compatible

        Example:
            >>> e1 = {"semantic": 0.4, "causal": 0.2, "spacetime": 0.2, "generative": 0.1, "trust": 0.1}
            >>> e2 = {"semantic": 0.3, "causal": 0.3, "spacetime": 0.2, "generative": 0.1, "trust": 0.1}
            >>> ec.calculate_compatibility(e1, e2)
            0.85
        """
        if not energies1 or not energies2:
            return 0.0

        # Normalize
        total1 = sum(energies1.values())
        total2 = sum(energies2.values())

        if total1 == 0 or total2 == 0:
            return 0.0

        norm1 = {k: v / total1 for k, v in energies1.items()}
        norm2 = {k: v / total2 for k, v in energies2.items()}

        # Calculate enhancement compatibility
        score = 0.0
        max_score = 0.0

        for e1 in self.ENERGY_ORDER:
            for e2 in self.ENERGY_ORDER:
                max_score += 1.0

                # Same energy type
                if e1 == e2:
                    score += min(norm1.get(e1, 0), norm2.get(e2, 0)) / max(norm1.get(e1, 0.001), norm2.get(e2, 0.001))
                    continue

                # Enhancement relationship
                if self.get_enhance_relation(e1, e2):
                    score += norm1.get(e1, 0) * norm2.get(e2, 0) * 2

                # Suppression relationship reduces compatibility
                if self.get_suppress_relation(e1, e2):
                    score -= abs(norm1.get(e1, 0) - norm2.get(e2, 0)) * 0.5

        return max(0.0, min(1.0, score / max_score * 2))

    def get_energy_cycle(self) -> list[tuple[str, str]]:
        """
        Get the five elements generation cycle.

        Returns:
            List of (source, target) tuples for the cycle
        """
        return [(k, v) for k, v in ENERGY_ENHANCE.items()]

    def get_control_cycle(self) -> list[tuple[str, str]]:
        """
        Get the five elements control cycle.

        Returns:
            List of (source, target) tuples for the control cycle
        """
        return [(k, v) for k, v in ENERGY_SUPPRESS.items()]

    def get_opposing_pair(self, energy: str) -> tuple[str, str]:
        """
        Get the opposing energy pair for a given energy.

        Args:
            energy: The energy type

        Returns:
            Tuple of (enhance_target, suppress_target)
        """
        energy = self._normalize_energy(energy)
        return (
            ENERGY_ENHANCE.get(energy, ""),
            ENERGY_SUPPRESS.get(energy, "")
        )


# ============================================================
# Testing
# ============================================================

def test_energy_core():
    """Run test cases for the Energy Core module."""
    logger.debug("=" * 60)
    logger.debug("Testing Energy Core Module")
    logger.debug("=" * 60)

    ec = EnergyCore()
    tests_passed = 0
    tests_failed = 0

    # Test 1: Enhancement relations
    logger.debug("\n[TEST 1] Enhancement Relations")
    test_cases = [
        ("semantic", "causal", True),
        ("causal", "semantic", False),
        ("causal", "spacetime", True),
        ("spacetime", "generative", True),
        ("generative", "trust", True),
        ("trust", "semantic", True),
    ]

    for e1, e2, expected in test_cases:
        result = ec.get_enhance_relation(e1, e2)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            tests_passed += 1
        else:
            tests_failed += 1
        logger.debug(f"  {e1} -> {e2}: {result} (expected {expected}) [{status}]")

    # Test 2: Suppression relations (bidirectional per task requirements)
    logger.debug("\n[TEST 2] Suppression Relations")
    test_cases = [
        ("wood", "earth", True),     # 木克土
        ("earth", "wood", True),     # 土克木 (bidirectional)
        ("earth", "water", True),    # 土克水
        ("water", "earth", True),    # 水克土 (bidirectional)
        ("water", "fire", True),     # 水克火
        ("fire", "water", True),     # 火克水 (bidirectional)
    ]

    for e1, e2, expected in test_cases:
        result = ec.get_suppress_relation(e1, e2)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            tests_passed += 1
        else:
            tests_failed += 1
        logger.debug(f"  {e1} -> {e2}: {result} (expected {expected}) [{status}]")

    # Test 3: Energy states (旺相休囚死)
    logger.debug("\n[TEST 3] Energy States by Month (旺相休囚死)")
    test_cases = [
        ("wood", 2, StrengthState.WANG),   # 寅月木旺
        ("wood", 3, StrengthState.WANG),   # 卯月木旺
        ("fire", 5, StrengthState.WANG),   # 巳月火旺
        ("fire", 6, StrengthState.WANG),   # 午月火旺
        ("earth", 4, StrengthState.WANG),  # 辰月土旺
        ("metal", 8, StrengthState.WANG),  # 申月金旺
        ("metal", 9, StrengthState.WANG),  # 酉月金旺
        ("water", 0, StrengthState.WANG),  # 子月水旺
        ("water", 11, StrengthState.WANG), # 亥月水旺
    ]

    for energy, branch, expected_strength in test_cases:
        state = ec.get_energy_state(energy, branch)
        status = "PASS" if state.strength == expected_strength else "FAIL"
        if status == "PASS":
            tests_passed += 1
        else:
            tests_failed += 1
        logger.debug(f"  {energy} @ branch {branch}: {state.strength.name} (expected {expected_strength.name}) [{status}]")

    # Test 4: Intensity calculation
    logger.debug("\n[TEST 4] Intensity Calculation")
    state = ec.get_energy_state("wood", 2)  # 寅月
    expected_intensity = ec.STRENGTH_MULTIPLIER[StrengthState.WANG]
    status = "PASS" if abs(state.intensity - expected_intensity) < 0.01 else "FAIL"
    if status == "PASS":
        tests_passed += 1
    else:
        tests_failed += 1
    logger.debug(f"  wood @ branch 2 intensity: {state.intensity} (expected {expected_intensity}) [{status}]")

    # Test 5: Balance analysis
    logger.debug("\n[TEST 5] Balance Analysis")
    energies = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
    result = ec.analyze_balance(energies)
    logger.debug(f"  Status: {result.status}")
    logger.debug(f"  Pattern: {result.pattern.name}")
    logger.debug(f"  Dominant: {result.dominant}")
    logger.debug(f"  Ratios: {result.ratios}")
    logger.debug(f"  Suggestions: {result.suggestions}")

    # Test 6: Energy attributes
    logger.debug("\n[TEST 6] Energy Attributes")
    attrs = ec.get_energy_attributes("wood")
    logger.debug("  Wood attributes:")
    for k, v in attrs.items():
        logger.debug(f"    {k}: {v}")

    # Test 7: Compatibility calculation
    logger.debug("\n[TEST 7] Compatibility Calculation")
    e1 = {"wood": 0.4, "fire": 0.2, "earth": 0.2, "metal": 0.1, "water": 0.1}
    e2 = {"wood": 0.3, "fire": 0.3, "earth": 0.2, "metal": 0.1, "water": 0.1}
    compat = ec.calculate_compatibility(e1, e2)
    logger.debug(f"  Compatibility score: {compat:.4f}")

    # Test 8: Interaction analysis
    logger.debug("\n[TEST 8] Interaction Analysis")
    interactions = ec.analyze_interaction("wood", "fire")
    logger.debug(f"  wood <-> fire: {[r.name for r in interactions]}")
    interactions = ec.analyze_interaction("wood", "earth")
    logger.debug(f"  wood <-> earth: {[r.name for r in interactions]}")

    # Test 9: Energy flow simulation
    logger.debug("\n[TEST 9] Energy Flow Simulation")
    initial = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
    history = ec.simulate_energy_flow(initial, steps=3)
    logger.debug(f"  Initial: {initial}")
    for i, state in enumerate(history[1:], 1):
        logger.debug(f"  Step {i}: {state}")

    # Test 10: Strength from branch
    logger.debug("\n[TEST 10] Strength from Branch")
    strengths = ec.get_strength_from_branch(2)  # 寅月
    logger.debug("  Branch 2 (寅月) strengths:")
    for energy, strength in strengths.items():
        logger.debug(f"    {energy}: {strength.name}")

    # Summary
    logger.debug("\n" + "=" * 60)
    logger.debug(f"Tests Passed: {tests_passed}")
    logger.debug(f"Tests Failed: {tests_failed}")
    logger.debug(f"Total: {tests_passed + tests_failed}")
    logger.debug("=" * 60)

    return tests_failed == 0


if __name__ == "__main__":
    success = test_energy_core()
    exit(0 if success else 1)
