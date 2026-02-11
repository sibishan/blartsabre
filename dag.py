import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class GateNode:
    """
    Attributes:
        gate_id: Unique identifier for the gate
        gate_type: Type of gate (e.g., 'H', 'CNOT', 'RZ', 'MEASURE')
        qubits: List of qubit indices this gate operates on
        parameters: Optional gate parameters (e.g., rotation angles)
        layer: Depth layer in the circuit (computed via topological ordering)
        metadata: Additional gate-specific information
    """
    gate_id: str
    gate_type: str
    qubits: Tuple[int, ...]
    clbits: Tuple[int, ...] = ()
    parameters: Optional[Dict[str, float]] = None
    layer: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Gate({self.gate_id}: {self.gate_type} on qubits {self.qubits})"


class QuantumDAG:
    
    def __init__(self, num_qubits: int = 0, num_clbits: int = 0):
        self.dag = nx.DiGraph()
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.gates: Dict[str, GateNode] = {}
        self._gate_counter = 0
        self._last_gate_on_qubit: Dict[int, str] = {}

    def __deepcopy__(self, memo):
        new_dag = QuantumDAG()
        new_dag.dag = deepcopy(self.dag)
        new_dag.num_qubits = deepcopy(self.num_qubits)
        new_dag.num_clbits = deepcopy(self.num_clbits)
        new_dag.gates = deepcopy(self.gates)
        new_dag._gate_counter = deepcopy(self._gate_counter)
        new_dag._last_gate_on_qubit = deepcopy(self._last_gate_on_qubit)
        return new_dag

    def add_gate(self, 
                 gate_type: str, 
                 qubits: List[int],
                 gate_id: Optional[str] = None,
                 parameters: Optional[Dict[str, float]] = None,
                 auto_dependencies: bool = True,
                 clbits = None) -> str:
        
        if gate_id is None:
            gate_id = f"g{self._gate_counter}"
            self._gate_counter += 1
            
        gate_node = GateNode(
            gate_id=gate_id,
            gate_type=gate_type,
            qubits=tuple(qubits),
            clbits=tuple(clbits) if clbits is not None else (),
            parameters=parameters or {},
        )
        
        self.gates[gate_id] = gate_node
        self.dag.add_node(gate_id, **gate_node.__dict__)
        
        if auto_dependencies:
            for qubit in qubits:
                if qubit in self._last_gate_on_qubit:
                    prev_gate = self._last_gate_on_qubit[qubit]
                    self.add_dependency(prev_gate, gate_id)
                self._last_gate_on_qubit[qubit] = gate_id
        
        return gate_id
    
    def add_dependency(self, from_gate: str, to_gate: str):
        """Add a dependency edge between two gates."""
        if from_gate not in self.gates or to_gate not in self.gates:
            raise ValueError("Both gates must exist in the DAG")
        
        self.dag.add_edge(from_gate, to_gate)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.dag):
            self.dag.remove_edge(from_gate, to_gate)
            raise ValueError(f"Adding edge {from_gate} -> {to_gate} would create a cycle")
    
    def remove_gate(self, gate_id: str):
        """Remove a gate from the DAG."""
        if gate_id in self.gates:
            self.dag.remove_node(gate_id)
            del self.gates[gate_id]
        
    def topological_sort(self) -> List[str]:
        """Return gates in topological order (respecting dependencies)."""
        try:
            return list(nx.topological_sort(self.dag))
        except nx.NetworkXError:
            raise ValueError("DAG contains a cycle!")
    
    def compute_layers(self) -> Dict[str, int]:
        """
        Compute the depth layer for each gate.
        Layer 0 contains gates with no dependencies, layer k contains gates
        whose predecessors are all in layers < k.
        
        Returns:
            Dictionary mapping gate_id to layer number
        """
        layers = {}
        for gate_id in self.topological_sort():
            predecessors = list(self.dag.predecessors(gate_id))
            if not predecessors:
                layers[gate_id] = 0
            else:
                layers[gate_id] = max(layers[p] for p in predecessors) + 1
            
            self.gates[gate_id].layer = layers[gate_id]
            self.dag.nodes[gate_id]['layer'] = layers[gate_id]
        
        return layers
    
    def get_depth(self) -> int:
        """Get the depth (critical path length) of the circuit."""
        if not self.gates:
            return 0
        layers = self.compute_layers()
        return max(layers.values()) + 1
    
    def get_gate_count(self) -> int:
        """Get total number of gates in the circuit."""
        return len(self.gates)
    
    def get_two_qubit_gate_count(self) -> int:
        """Get number of two-qubit gates."""
        return sum(1 for gate in self.gates.values() if len(gate.qubits) == 2)
    
    def get_predecessors(self, gate_id: str) -> List[str]:
        """Get all gates that must execute before this gate."""
        return list(self.dag.predecessors(gate_id))
    
    def get_successors(self, gate_id: str) -> List[str]:
        """Get all gates that must execute after this gate."""
        return list(self.dag.successors(gate_id))
    
    def get_ancestors(self, gate_id: str) -> Set[str]:
        """Get all gates in the dependency cone before this gate."""
        return nx.ancestors(self.dag, gate_id)
    
    def get_descendants(self, gate_id: str) -> Set[str]:
        """Get all gates in the dependency cone after this gate."""
        return nx.descendants(self.dag, gate_id)
    
    def get_front_layer(self) -> List[str]:
        """Get all GateNodes at the front layer (no predecessors/dependencies)."""
        return [gate_id for gate_id, layer in self.compute_layers().items() if layer == 0]
    
    def get_back_layer(self) -> List[str]:
        """Get all gates at the back layer (no successors)."""
        return [node for node, degree in self.dag.out_degree() if degree == 0]
    
    def get_extended_layer(self, distance: int = 1) -> List[str]:
        """Get all gates at a specific depth layer from the front."""
        return [gate_id for gate_id, layer in self.compute_layers().items() if (layer <= distance and layer > 0)]
    
    def get_gate_from_node(self, node):
        return self.gates[node]
    
    def get_gates_from_nodes(self, nodes):
        return [self.gates[node] for node in nodes]