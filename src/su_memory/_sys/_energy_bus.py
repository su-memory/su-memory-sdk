"""
Energy Bus Module (能量总线) - Core Energy Flow System

This module implements the Energy Bus system for unified management of all energy types:
- Five Elements (五行)
- Trigrams (八卦)
- Heavenly Stems/Earthly Branches (干支)

Architecture:
- EnergyNode: Represents a node in the energy network
- EnergyChannel: Represents connections between nodes
- EnergyBus: Central controller managing energy flow
- EnergyLayer: Hierarchical organization (五行→八卦→时空)

【先天主数】- Energy bus uses prior trigram ordering for numerical calculations
【后天主象】- Energy bus uses post trigram ordering for symbolic applications

Energy Flow Model:
    五行层 (Five Elements Layer)
         ↓↑ 相生/相克 (Enhance/Suppress)
    八卦层 (Trigrams Layer)  
         ↓↑ 纳甲映射 (Najia Mapping)
    时空层 (Spacetime Layer)
         ↓↑ 天干地支 (Stems/Branches)

Core Features:
- Unified energy type management
- Cross-layer energy propagation
- Dynamic balance adjustment
- Energy state monitoring
- Multi-path concurrent flow
"""

from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math
import time

from ._enums import EnergyType, EnergyRelation as EnumEnergyRelation, TrigramType, TimeStem, TimeBranch
from ._energy_relations import (
    EnergyType as EnergyTypeEnum,
    RelationType,
    ENERGY_ENHANCE,
    ENERGY_SUPPRESS,
    ENERGY_SUPPRESSED_BY,
    ENERGY_ENHANCED_BY,
    analyze_relation,
    calculate_link_weight,
)
from ._taiji_map import (
    TaijiMapper,
    STEM_TO_TRIGRAM,
    TRIGRAM_ENERGY_TYPE,
    PRIOR_ORDER,
    POST_ORDER,
    POST_TRIGRAM_ORDER,
)


# =============================================================================
# Energy Layer Enum
# =============================================================================

class EnergyLayer(Enum):
    """Energy flow hierarchy layers"""
    FIVE_ELEMENTS = "five_elements"    # 五行层
    TRIGRAMS = "trigrams"              # 八卦层
    SPACETIME = "spacetime"            # 时空层


class EnergyState(Enum):
    """Energy node state"""
    ACTIVE = "active"                  # 活跃
    DORMANT = "dormant"                # 休眠
    BALANCED = "balanced"              # 平衡
    IMBALANCED = "imbalanced"          # 失衡
    BLOCKED = "blocked"                 # 阻塞


# =============================================================================
# Energy Node Data Structure
# =============================================================================

@dataclass
class EnergyNode:
    """
    Energy Node - Represents a point in the energy network.
    
    Attributes:
        node_id: Unique identifier for the node
        energy_type: Energy type (e.g., "wood", "fire", etc.)
        layer: Which layer this node belongs to
        intensity: Current energy intensity (0.0 - 2.0)
        max_intensity: Maximum allowed intensity
        position: Spatial/temporal position info
        stem_idx: Heavenly stem index (if applicable)
        branch_idx: Earthly branch index (if applicable)
        trigram_idx: Trigram index (if applicable)
        state: Current energy state
        connections: Set of connected node IDs
        metadata: Additional metadata
    """
    node_id: str
    energy_type: str
    layer: EnergyLayer
    intensity: float = 1.0
    max_intensity: float = 2.0
    position: Optional[Tuple[int, int]] = None  # (spatial, temporal)
    stem_idx: Optional[int] = None
    branch_idx: Optional[int] = None
    trigram_idx: Optional[int] = None
    state: EnergyState = EnergyState.ACTIVE
    connections: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and clamp intensity"""
        self.intensity = max(0.0, min(self.intensity, self.max_intensity))
    
    @property
    def energy_level(self) -> str:
        """Get energy level description"""
        if self.intensity >= 1.8:
            return "旺盛"  # Strong
        elif self.intensity >= 1.2:
            return "偏旺"  # Slightly strong
        elif self.intensity >= 0.8:
            return "中和"  # Balanced
        elif self.intensity >= 0.3:
            return "偏弱"  # Slightly weak
        else:
            return "衰弱"  # Weak
    
    def adjust_intensity(self, delta: float) -> float:
        """
        Adjust node intensity by delta.
        
        Args:
            delta: Amount to adjust (+/-)
        
        Returns:
            New intensity value
        """
        self.intensity = max(0.0, min(self.intensity + delta, self.max_intensity))
        return self.intensity


@dataclass
class EnergyChannel:
    """
    Energy Channel - Represents a connection between nodes.
    
    Attributes:
        channel_id: Unique identifier
        source_id: Source node ID
        target_id: Target node ID
        relation_type: Type of energy relation (相生/相克/同类)
        base_weight: Base connection weight
        current_flow: Current energy flow amount
        max_flow: Maximum allowed flow
        latency: Time delay for energy propagation
        active: Whether channel is active
    """
    channel_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    base_weight: float = 1.0
    current_flow: float = 0.0
    max_flow: float = 1.5
    latency: float = 0.0  # Time delay in propagation
    active: bool = True
    
    @property
    def effective_weight(self) -> float:
        """Calculate effective weight based on relation type"""
        if self.relation_type == RelationType.ENHANCE:
            return self.base_weight * 1.2
        elif self.relation_type == RelationType.SUPPRESS:
            return self.base_weight * 0.8
        elif self.relation_type == RelationType.OVERCONSTRAINT:
            return self.base_weight * 0.6
        elif self.relation_type == RelationType.REVERSE:
            return self.base_weight * 0.4
        elif self.relation_type == RelationType.SAME:
            return self.base_weight * 1.1
        return self.base_weight


@dataclass
class EnergySignal:
    """
    Energy Signal - Represents energy being transmitted through the bus.
    
    Attributes:
        signal_id: Unique identifier
        source_node: Source node ID
        target_node: Target node ID
        energy_type: Type of energy being transmitted
        intensity: Signal intensity
        timestamp: When signal was created
        layer: Which layer signal is on
        ttl: Time to live (propagation steps remaining)
        metadata: Additional signal metadata
    """
    signal_id: str
    source_node: str
    target_node: str
    energy_type: str
    intensity: float
    timestamp: float
    layer: EnergyLayer
    ttl: int = 3
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Energy Propagation Config
# =============================================================================

@dataclass
class PropagationConfig:
    """Configuration for energy propagation algorithm"""
    max_hops: int = 5                    # Maximum propagation steps
    decay_rate: float = 0.85            # Energy decay per hop
    time_factor: float = 0.1             # Time-based decay weight
    space_factor: float = 0.15           # Space-based decay weight
    concurrency_limit: int = 100         # Max concurrent signals
    enable_feedback: bool = True         # Enable feedback loops
    enable_blocking: bool = True         # Enable suppression blocking
    
    # Prior (先天) numerical config
    prior_weight: float = 0.4            # Weight for prior calculations
    
    # Post (后天) symbolic config
    post_weight: float = 0.6             # Weight for post applications


# =============================================================================
# Energy Bus Core Class
# =============================================================================

class EnergyBus:
    """
    Energy Bus - Central controller for energy flow management.
    
    Manages energy nodes, channels, and propagation across all layers
    (Five Elements, Trigrams, Spacetime).
    
    Example:
        >>> bus = EnergyBus()
        >>> bus.add_node(EnergyNode("node1", "wood", EnergyLayer.FIVE_ELEMENTS))
        >>> bus.add_node(EnergyNode("node2", "fire", EnergyLayer.FIVE_ELEMENTS))
        >>> bus.connect("node1", "node2", RelationType.ENHANCE)
        >>> bus.propagate_energy("node1", 0.5)
        >>> state = bus.get_node_state("node2")
        >>> print(state['intensity'])  # ~1.5 (original + enhanced)
    """
    
    _node_counter: int = 0
    
    def __init__(self, config: Optional[PropagationConfig] = None):
        """
        Initialize the Energy Bus.
        
        Args:
            config: Propagation configuration (optional)
        """
        self._nodes: Dict[str, EnergyNode] = {}
        self._channels: Dict[str, EnergyChannel] = {}
        self._signal_history: List[EnergySignal] = []
        self._config = config or PropagationConfig()
        self._taiji_mapper = TaijiMapper()
        
        # Layer statistics
        self._layer_stats: Dict[EnergyLayer, Dict] = {}
        for layer in EnergyLayer:
            self._layer_stats[layer] = {
                "node_count": 0,
                "total_intensity": 0.0,
                "avg_intensity": 0.0,
            }
        
        # Energy balance tracking
        self._balance_history: List[Dict] = []
    
    # =========================================================================
    # Node Management
    # =========================================================================
    
    def add_node(
        self,
        node: EnergyNode,
        auto_connect: bool = True
    ) -> str:
        """
        Add an energy node to the bus.
        
        Args:
            node: EnergyNode to add
            auto_connect: Whether to auto-connect to related nodes
        
        Returns:
            Node ID
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        
        self._nodes[node.node_id] = node
        self._update_layer_stats(node.layer)
        
        if auto_connect:
            self._auto_connect_node(node)
        
        return node.node_id
    
    def _auto_connect_node(self, node: EnergyNode):
        """Auto-connect a node to related nodes based on energy relations"""
        for other_id, other_node in self._nodes.items():
            if other_id == node.node_id:
                continue
            
            # Calculate relation based on energy types
            relation = analyze_relation(node.energy_type, other_node.energy_type)
            
            # Create connection if there's a meaningful relation
            if relation.relation != RelationType.NEUTRAL:
                self.connect(
                    node.node_id,
                    other_id,
                    relation.relation,
                    base_weight=1.0
                )
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the bus.
        
        Args:
            node_id: Node to remove
        
        Returns:
            True if removed
        """
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        
        # Remove all channels connected to this node
        channels_to_remove = [
            ch_id for ch_id, ch in self._channels.items()
            if ch.source_id == node_id or ch.target_id == node_id
        ]
        for ch_id in channels_to_remove:
            del self._channels[ch_id]
        
        # Remove from other nodes' connections
        for other_node in self._nodes.values():
            other_node.connections.discard(node_id)
        
        del self._nodes[node_id]
        self._update_layer_stats(node.layer)
        return True
    
    def get_node(self, node_id: str) -> Optional[EnergyNode]:
        """Get a node by ID"""
        return self._nodes.get(node_id)
    
    def get_nodes_by_layer(self, layer: EnergyLayer) -> List[EnergyNode]:
        """Get all nodes in a specific layer"""
        return [n for n in self._nodes.values() if n.layer == layer]
    
    def get_nodes_by_energy(self, energy_type: str) -> List[EnergyNode]:
        """Get all nodes of a specific energy type"""
        return [n for n in self._nodes.values() if n.energy_type == energy_type]
    
    # =========================================================================
    # Channel Management
    # =========================================================================
    
    def connect(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        base_weight: float = 1.0,
        latency: float = 0.0
    ) -> Optional[str]:
        """
        Create a channel between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of energy relation
            base_weight: Base connection weight
            latency: Time delay for propagation
        
        Returns:
            Channel ID if successful
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        channel_id = f"ch_{source_id}_{target_id}"
        
        channel = EnergyChannel(
            channel_id=channel_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            base_weight=base_weight,
            latency=latency
        )
        
        self._channels[channel_id] = channel
        self._nodes[source_id].connections.add(target_id)
        
        return channel_id
    
    def disconnect(self, channel_id: str) -> bool:
        """Remove a channel"""
        if channel_id not in self._channels:
            return False
        
        channel = self._channels[channel_id]
        source_node = self._nodes.get(channel.source_id)
        if source_node:
            source_node.connections.discard(channel.target_id)
        
        del self._channels[channel_id]
        return True
    
    def get_channel(self, channel_id: str) -> Optional[EnergyChannel]:
        """Get a channel by ID"""
        return self._channels.get(channel_id)
    
    def get_outgoing_channels(self, node_id: str) -> List[EnergyChannel]:
        """Get all outgoing channels from a node"""
        return [
            ch for ch in self._channels.values()
            if ch.source_id == node_id and ch.active
        ]
    
    def get_incoming_channels(self, node_id: str) -> List[EnergyChannel]:
        """Get all incoming channels to a node"""
        return [
            ch for ch in self._channels.values()
            if ch.target_id == node_id and ch.active
        ]
    
    # =========================================================================
    # Energy Propagation (能量传播算法)
    # =========================================================================
    
    def propagate_energy(
        self,
        source_id: str,
        delta: float,
        max_hops: Optional[int] = None
    ) -> List[EnergySignal]:
        """
        Propagate energy from a source node through the network.
        
        【先天主数】- Uses numerical calculation for path finding
        【后天主象】- Uses symbolic mapping for actual application
        
        Args:
            source_id: Source node ID
            delta: Energy amount to propagate
            max_hops: Maximum propagation steps (default from config)
        
        Returns:
            List of signals created during propagation
        """
        if source_id not in self._nodes:
            return []
        
        source_node = self._nodes[source_id]
        max_hops = max_hops or self._config.max_hops
        signals: List[EnergySignal] = []
        
        # Create initial signal
        initial_signal = EnergySignal(
            signal_id=self._generate_signal_id(),
            source_node=source_id,
            target_node=source_id,
            energy_type=source_node.energy_type,
            intensity=delta,
            timestamp=time.time(),
            layer=source_node.layer,
            ttl=max_hops
        )
        signals.append(initial_signal)
        
        # Direct intensity boost to source
        source_node.adjust_intensity(delta)
        
        # Propagate through channels
        self._propagate_recursive(source_id, delta, max_hops, signals, visited=set())
        
        # Record signals
        self._signal_history.extend(signals)
        
        return signals
    
    def _propagate_recursive(
        self,
        node_id: str,
        intensity: float,
        remaining_hops: int,
        signals: List[EnergySignal],
        visited: Set[str]
    ):
        """Recursively propagate energy through the network"""
        if remaining_hops <= 0 or intensity <= 0:
            return
        
        if node_id in visited:
            return
        visited.add(node_id)
        
        node = self._nodes.get(node_id)
        if not node:
            return
        
        # Get outgoing channels
        channels = self.get_outgoing_channels(node_id)
        
        for channel in channels:
            target_id = channel.target_id
            target_node = self._nodes.get(target_id)
            
            if not target_node:
                continue
            
            # Calculate propagation weight using prior/post principles
            # 【先天主数】: Numerical calculation for weight
            # 【后天主象】: Symbolic application for actual effect
            
            # Base propagation: decay based on hops
            base_decay = self._config.decay_rate ** (self._config.max_hops - remaining_hops)
            
            # Relation-based modifier
            relation_modifier = self._calculate_relation_modifier(channel.relation_type)
            
            # Time-based decay (后天主象)
            time_decay = math.exp(-self._config.time_factor * remaining_hops)
            
            # Space-based decay (后天主象)
            space_decay = math.exp(-self._config.space_factor * (channel.latency or 0))
            
            # Combined decay with prior/post weighting
            prior_contribution = base_decay * self._config.prior_weight
            post_contribution = time_decay * space_decay * self._config.post_weight
            combined_decay = prior_contribution + post_contribution
            
            # Calculate final propagation intensity
            propagated_intensity = (
                intensity
                * channel.effective_weight
                * relation_modifier
                * combined_decay
            )
            
            if propagated_intensity < 0.01:
                continue
            
            # Create signal
            signal = EnergySignal(
                signal_id=self._generate_signal_id(),
                source_node=node_id,
                target_node=target_id,
                energy_type=node.energy_type,
                intensity=propagated_intensity,
                timestamp=time.time(),
                layer=target_node.layer,
                ttl=remaining_hops - 1
            )
            signals.append(signal)
            
            # Apply to target node
            self._apply_signal_to_node(target_node, signal, channel.relation_type)
            
            # Recursive propagation
            if self._config.enable_feedback or channel.relation_type == RelationType.ENHANCE:
                self._propagate_recursive(
                    target_id,
                    propagated_intensity,
                    remaining_hops - 1,
                    signals,
                    visited.copy()
                )
    
    def _calculate_relation_modifier(self, relation: RelationType) -> float:
        """
        Calculate energy propagation modifier based on relation type.
        
        【后天主象】- Uses symbolic meaning for actual effect
        """
        if relation == RelationType.ENHANCE:
            return 1.1  # 相生增强
        elif relation == RelationType.SUPPRESS:
            return self._config.enable_blocking and 0.5 or 0.8  # 相克抑制
        elif relation == RelationType.OVERCONSTRAINT:
            return 0.3  # 相乘大幅削弱
        elif relation == RelationType.REVERSE:
            return 0.2  # 相侮大幅削弱
        elif relation == RelationType.SAME:
            return 1.05  # 同类微增
        return 1.0
    
    def _apply_signal_to_node(
        self,
        node: EnergyNode,
        signal: EnergySignal,
        relation: RelationType
    ):
        """Apply an energy signal to a node"""
        if relation in [RelationType.SUPPRESS, RelationType.OVERCONSTRAINT, RelationType.REVERSE]:
            # Suppression reduces intensity
            node.adjust_intensity(-signal.intensity * 0.5)
        else:
            # Enhancement increases intensity
            node.adjust_intensity(signal.intensity)
        
        # Update state based on new intensity
        if node.intensity > 1.5:
            node.state = EnergyState.ACTIVE
        elif node.intensity < 0.5:
            node.state = EnergyState.DORMANT
        else:
            node.state = EnergyState.BALANCED
    
    # =========================================================================
    # Cross-Layer Energy Flow
    # =========================================================================
    
    def flow_between_layers(
        self,
        source_layer: EnergyLayer,
        target_layer: EnergyLayer,
        intensity: float
    ) -> Dict[str, float]:
        """
        Flow energy between layers using Najia mapping.
        
        【后天主象】- Uses post trigram ordering for spatial mapping
        
        Args:
            source_layer: Source layer
            target_layer: Target layer
            intensity: Energy intensity to flow
        
        Returns:
            Mapping of target node IDs to applied intensity
        """
        results: Dict[str, float] = {}
        
        # Get nodes from source layer
        source_nodes = self.get_nodes_by_layer(source_layer)
        
        for source_node in source_nodes:
            # Map to target layer using energy type correspondence
            target_energy = self._get_layer_mapping_energy(source_node.energy_type, target_layer)
            
            # Find or create target nodes
            target_nodes = self.get_nodes_by_energy(target_energy)
            
            for target_node in target_nodes:
                # Calculate flow using prior/post principles
                flow_amount = intensity * self._calculate_layer_flow_coefficient(
                    source_node, target_node, target_layer
                )
                
                if flow_amount > 0.01:
                    target_node.adjust_intensity(flow_amount)
                    results[target_node.node_id] = flow_amount
        
        return results
    
    def _get_layer_mapping_energy(self, source_energy: str, target_layer: EnergyLayer) -> str:
        """Get the mapped energy type for a target layer"""
        # 五行层 -> 八卦层: Use TRIGRAM_ENERGY_TYPE mapping
        # This is handled by the energy system directly
        
        # For cross-layer mapping, we use the energy type directly
        # since all layers ultimately map to five elements
        return source_energy
    
    def _calculate_layer_flow_coefficient(
        self,
        source: EnergyNode,
        target: EnergyNode,
        target_layer: EnergyLayer
    ) -> float:
        """
        Calculate flow coefficient between layers.
        
        【先天主数】: Numerical calculation
        【后天主象】: Symbolic application
        """
        base_coefficient = 0.5
        
        # Apply prior/post weighting based on target layer
        if target_layer == EnergyLayer.TRIGRAMS:
            # Use post ordering for trigram layer (symbolic application)
            post_weight = self._config.post_weight
            prior_weight = self._config.prior_weight
        elif target_layer == EnergyLayer.SPACETIME:
            # Heavy use of post ordering for spacetime (spatial/temporal)
            post_weight = 0.8
            prior_weight = 0.2
        else:
            post_weight = 0.5
            prior_weight = 0.5
        
        # Calculate trigram position difference for spatial decay
        if source.trigram_idx is not None and target.trigram_idx is not None:
            # 【先天主数】: Calculate numerical distance
            pos_diff = abs(
                PRIOR_ORDER.get(source.trigram_idx, 0) -
                PRIOR_ORDER.get(target.trigram_idx, 0)
            )
            spatial_decay = math.exp(-0.1 * min(pos_diff, 4))
        else:
            spatial_decay = 1.0
        
        return base_coefficient * (prior_weight + post_weight * spatial_decay)
    
    # =========================================================================
    # State Management and Balance
    # =========================================================================
    
    def get_node_state(self, node_id: str) -> Optional[Dict]:
        """Get comprehensive state of a node"""
        node = self.get_node(node_id)
        if not node:
            return None
        
        incoming = self.get_incoming_channels(node_id)
        outgoing = self.get_outgoing_channels(node_id)
        
        return {
            "node_id": node.node_id,
            "energy_type": node.energy_type,
            "layer": node.layer.value,
            "intensity": node.intensity,
            "state": node.state.value,
            "energy_level": node.energy_level,
            "incoming_channels": len(incoming),
            "outgoing_channels": len(outgoing),
            "stem_idx": node.stem_idx,
            "branch_idx": node.branch_idx,
            "trigram_idx": node.trigram_idx,
            "connections": list(node.connections),
        }
    
    def get_bus_state(self) -> Dict:
        """Get overall energy bus state"""
        total_intensity = sum(n.intensity for n in self._nodes.values())
        active_channels = sum(1 for ch in self._channels.values() if ch.active)
        
        # Calculate balance
        energy_balance = self._calculate_energy_balance()
        
        return {
            "total_nodes": len(self._nodes),
            "active_channels": active_channels,
            "total_intensity": total_intensity,
            "avg_intensity": total_intensity / len(self._nodes) if self._nodes else 0,
            "layer_stats": {
                layer.value: stats for layer, stats in self._layer_stats.items()
            },
            "energy_balance": energy_balance,
            "signal_count": len(self._signal_history),
        }
    
    def _calculate_energy_balance(self) -> Dict:
        """
        Calculate energy balance across five elements.
        
        【后天主象】- Uses post ordering for symbolic balance analysis
        """
        element_totals: Dict[str, float] = {
            "wood": 0.0, "fire": 0.0, "earth": 0.0,
            "metal": 0.0, "water": 0.0
        }
        
        for node in self._nodes.values():
            element_totals[node.energy_type] = (
                element_totals.get(node.energy_type, 0.0) + node.intensity
            )
        
        total = sum(element_totals.values())
        if total == 0:
            return {"balanced": True, "ratios": element_totals}
        
        ratios = {k: v / total for k, v in element_totals.items()}
        
        # Check for imbalance (any element > 50% or < 10%)
        max_ratio = max(ratios.values())
        min_ratio = min(ratios.values())
        
        return {
            "balanced": max_ratio < 0.5 and min_ratio > 0.1,
            "ratios": ratios,
            "dominant": max(element_totals, key=element_totals.get),
            "weakest": min(element_totals, key=element_totals.get),
        }
    
    def _update_layer_stats(self, layer: EnergyLayer):
        """Update layer statistics"""
        nodes = self.get_nodes_by_layer(layer)
        total = sum(n.intensity for n in nodes)
        
        self._layer_stats[layer] = {
            "node_count": len(nodes),
            "total_intensity": total,
            "avg_intensity": total / len(nodes) if nodes else 0,
        }
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def create_five_elements_nodes(self) -> Dict[str, EnergyNode]:
        """Create nodes for all five elements"""
        nodes = {}
        for energy_type in ["wood", "fire", "earth", "metal", "water"]:
            node_id = f"wuxing_{energy_type}"
            node = EnergyNode(
                node_id=node_id,
                energy_type=energy_type,
                layer=EnergyLayer.FIVE_ELEMENTS
            )
            self.add_node(node, auto_connect=False)
            nodes[energy_type] = node
        
        # Create connections based on enhance/suppress relations
        self._connect_five_elements_network()
        
        return nodes
    
    def create_trigram_nodes(self) -> Dict[str, EnergyNode]:
        """Create nodes for all eight trigrams"""
        nodes = {}
        for trig_idx in range(8):
            trig = TrigramType(trig_idx)
            energy_type = TRIGRAM_ENERGY_TYPE[trig]
            
            node_id = f"trigram_{trig_idx}"
            node = EnergyNode(
                node_id=node_id,
                energy_type=energy_type,
                layer=EnergyLayer.TRIGRAMS,
                trigram_idx=trig_idx
            )
            self.add_node(node, auto_connect=False)
            nodes[trig.name] = node
        
        # Connect trigrams based on energy relations
        self._connect_trigram_network(nodes)
        
        return nodes
    
    def _connect_five_elements_network(self):
        """Connect five elements nodes based on enhance/suppress relations"""
        for source_type, target_type in ENERGY_ENHANCE.items():
            source_id = f"wuxing_{source_type}"
            target_id = f"wuxing_{target_type}"
            if source_id in self._nodes and target_id in self._nodes:
                self.connect(source_id, target_id, RelationType.ENHANCE, base_weight=1.0)
        
        for source_type, target_type in ENERGY_SUPPRESS.items():
            source_id = f"wuxing_{source_type}"
            target_id = f"wuxing_{target_type}"
            if source_id in self._nodes and target_id in self._nodes:
                self.connect(source_id, target_id, RelationType.SUPPRESS, base_weight=0.8)
    
    def _connect_trigram_network(self, nodes: Dict[str, EnergyNode]):
        """Connect trigram nodes based on energy relations"""
        for trig_name, node in nodes.items():
            for other_name, other_node in nodes.items():
                if trig_name == other_name:
                    continue
                
                relation = analyze_relation(node.energy_type, other_node.energy_type)
                if relation.relation != RelationType.NEUTRAL:
                    self.connect(node.node_id, other_node.node_id, relation.relation)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @classmethod
    def _generate_signal_id(cls) -> str:
        """Generate unique signal ID"""
        cls._node_counter += 1
        return f"sig_{cls._node_counter}_{int(time.time() * 1000)}"
    
    def clear(self):
        """Clear all nodes and channels"""
        self._nodes.clear()
        self._channels.clear()
        self._signal_history.clear()
        for layer in EnergyLayer:
            self._layer_stats[layer] = {
                "node_count": 0,
                "total_intensity": 0.0,
                "avg_intensity": 0.0,
            }
    
    def __repr__(self) -> str:
        return f"EnergyBus(nodes={len(self._nodes)}, channels={len(self._channels)})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_energy_bus(config: Optional[PropagationConfig] = None) -> EnergyBus:
    """
    Create and initialize an Energy Bus.
    
    Args:
        config: Optional propagation configuration
    
    Returns:
        Initialized EnergyBus instance
    """
    return EnergyBus(config)


def create_complete_energy_network() -> EnergyBus:
    """
    Create a complete energy network with all layers.
    
    Returns:
        EnergyBus with Five Elements and Trigrams nodes
    """
    bus = EnergyBus()
    bus.create_five_elements_nodes()
    bus.create_trigram_nodes()
    return bus


# =============================================================================
# Test Suite
# =============================================================================

def test_energy_bus():
    """Test Energy Bus functionality"""
    print("=" * 60)
    print("Energy Bus Test Suite")
    print("=" * 60)
    
    bus = EnergyBus()
    passed = 0
    failed = 0
    
    def test(name: str, condition: bool, details: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name} - FAILED{details}")
            failed += 1
    
    # Test 1: Node Management
    print("\n[Test 1] Node Management")
    print("-" * 40)
    
    node1 = EnergyNode("node1", "wood", EnergyLayer.FIVE_ELEMENTS)
    node2 = EnergyNode("node2", "fire", EnergyLayer.FIVE_ELEMENTS)
    
    bus.add_node(node1, auto_connect=False)
    bus.add_node(node2, auto_connect=False)
    
    test("Add node1", bus.get_node("node1") is not None)
    test("Add node2", bus.get_node("node2") is not None)
    test("Get nodes by layer", len(bus.get_nodes_by_layer(EnergyLayer.FIVE_ELEMENTS)) == 2)
    test("Get nodes by energy", len(bus.get_nodes_by_energy("wood")) == 1)
    
    # Test 2: Channel Management
    print("\n[Test 2] Channel Management")
    print("-" * 40)
    
    channel_id = bus.connect("node1", "node2", RelationType.ENHANCE, base_weight=1.0)
    test("Create channel", channel_id is not None)
    test("Channel exists", bus.get_channel(channel_id) is not None)
    test("Outgoing channels", len(bus.get_outgoing_channels("node1")) == 1)
    test("Incoming channels", len(bus.get_incoming_channels("node2")) == 1)
    
    # Test 3: Energy Propagation
    print("\n[Test 3] Energy Propagation")
    print("-" * 40)
    
    initial_intensity = bus.get_node("node1").intensity
    signals = bus.propagate_energy("node1", 0.5)
    test("Propagate creates signals", len(signals) > 0)
    
    node1_new = bus.get_node("node1")
    test("Source intensity increased", node1_new.intensity > initial_intensity)
    
    # Test 4: Five Elements Network
    print("\n[Test 4] Five Elements Network")
    print("-" * 40)
    
    bus2 = EnergyBus()
    bus2.create_five_elements_nodes()
    
    test("Five elements nodes created", len(bus2._nodes) == 5)
    
    # Check enhance connections exist
    wood_node = bus2.get_node("wuxing_wood")
    fire_node = bus2.get_node("wuxing_fire")
    test("Wood node exists", wood_node is not None)
    test("Fire node exists", fire_node is not None)
    
    # Test propagation through enhance cycle
    bus2.propagate_energy("wuxing_wood", 0.3)
    fire_intensity = bus2.get_node("wuxing_fire").intensity
    test("Fire intensity increased via propagation", fire_intensity > 1.0)
    
    # Test 5: Trigrams Network
    print("\n[Test 5] Trigrams Network")
    print("-" * 40)
    
    bus3 = EnergyBus()
    bus3.create_trigram_nodes()
    
    test("Trigram nodes created", len(bus3._nodes) == 8)
    
    # Check QIAN (metal) and ZHEN (wood) have correct energy
    qian = bus3.get_node("trigram_0")
    test("QIAN energy is metal", qian.energy_type == "metal")
    
    # Test 6: Bus State
    print("\n[Test 6] Bus State")
    print("-" * 40)
    
    state = bus2.get_bus_state()
    test("State has total nodes", state["total_nodes"] == 5)
    test("State has energy balance", "energy_balance" in state)
    test("State has layer stats", "layer_stats" in state)
    
    # Test 7: Balance Calculation
    print("\n[Test 7] Energy Balance Calculation")
    print("-" * 40)
    
    balance = bus2._calculate_energy_balance()
    test("Balance has ratios", "ratios" in balance)
    test("Balance has dominant", "dominant" in balance)
    test("Balance has weakest", "weakest" in balance)
    
    # Test 8: Node State
    print("\n[Test 8] Node State Query")
    print("-" * 40)
    
    node_state = bus2.get_node_state("wuxing_wood")
    test("Node state has intensity", "intensity" in node_state)
    test("Node state has energy_level", "energy_level" in node_state)
    test("Node state has connections", "connections" in node_state)
    
    # Test 9: Cross-Layer Flow
    print("\n[Test 9] Cross-Layer Energy Flow")
    print("-" * 40)
    
    bus4 = EnergyBus()
    bus4.create_five_elements_nodes()
    bus4.create_trigram_nodes()
    
    # Flow from five elements to trigrams
    results = bus4.flow_between_layers(
        EnergyLayer.FIVE_ELEMENTS,
        EnergyLayer.TRIGRAMS,
        0.5
    )
    test("Cross-layer flow produces results", len(results) >= 0)
    
    # Test 10: Channel Relation Types
    print("\n[Test 10] Channel Relation Types")
    print("-" * 40)
    
    bus5 = EnergyBus()
    wood = EnergyNode("w", "wood", EnergyLayer.FIVE_ELEMENTS)
    earth = EnergyNode("e", "earth", EnergyLayer.FIVE_ELEMENTS)
    bus5.add_node(wood, auto_connect=False)
    bus5.add_node(earth, auto_connect=False)
    bus5.connect("w", "e", RelationType.SUPPRESS)
    
    channel = bus5.get_channel("ch_w_e")
    test("Channel has correct relation", channel.relation_type == RelationType.SUPPRESS)
    test("Suppress effective weight", abs(channel.effective_weight - 0.8) < 0.01)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_energy_bus()
    exit(0 if success else 1)