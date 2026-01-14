import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class GateNode:
    """
    Represents a quantum gate in the circuit DAG.
    
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
    """Directed Acyclic Graph representation for quantum circuits with DQC support."""
    
    def __init__(self, num_qubits: int = 0, num_clbits: int = 0):
        """
        Initialize a quantum circuit DAG.
        
        Args:
            num_qubits: Total number of qubits in the circuit
        """
        self.dag = nx.DiGraph()
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.gates: Dict[str, GateNode] = {}
        self._gate_counter = 0
        
        # Track the last gate on each qubit (for automatic dependency addition)
        self._last_gate_on_qubit: Dict[int, str] = {}
        
        # Track qubit-to-processor mapping (for DQC)
        self.qubit_to_processor: Dict[int, str] = {}

    def __deepcopy__(self, memo):
        new_dag = QuantumDAG()
        new_dag.dag = deepcopy(self.dag)
        new_dag.num_qubits = deepcopy(self.num_qubits)
        new_dag.num_clbits = deepcopy(self.num_clbits)
        new_dag.gates = deepcopy(self.gates)
        new_dag._gate_counter = deepcopy(self._gate_counter)
        new_dag._last_gate_on_qubit = deepcopy(self._last_gate_on_qubit)
        new_dag.qubit_to_processor = deepcopy(self.qubit_to_processor)
        return new_dag

    def add_gate(self, 
                 gate_type: str, 
                 qubits: List[int],
                 gate_id: Optional[str] = None,
                 parameters: Optional[Dict[str, float]] = None,
                 auto_dependencies: bool = True,
                 clbits = None) -> str:
        """
        Add a quantum gate to the DAG.
        
        Args:
            gate_type: Type of quantum gate
            qubits: List of qubit indices
            gate_id: Optional custom gate ID (auto-generated if None)
            parameters: Optional gate parameters
            auto_dependencies: If True, automatically add dependencies based on qubit usage
            
        Returns:
            gate_id: The ID of the added gate
        """
        if gate_id is None:
            gate_id = f"g{self._gate_counter}"
            self._gate_counter += 1
            
        # Create gate node
        gate_node = GateNode(
            gate_id=gate_id,
            gate_type=gate_type,
            qubits=tuple(qubits),
            clbits=tuple(clbits) if clbits is not None else (),
            parameters=parameters or {},
        )
        
        self.gates[gate_id] = gate_node
        self.dag.add_node(gate_id, **gate_node.__dict__)
        
        # Automatically add dependencies based on qubit usage
        if auto_dependencies:
            for qubit in qubits:
                if qubit in self._last_gate_on_qubit:
                    prev_gate = self._last_gate_on_qubit[qubit]
                    self.add_dependency(prev_gate, gate_id)
                self._last_gate_on_qubit[qubit] = gate_id
        
        return gate_id
    
    def add_dependency(self, from_gate: str, to_gate: str):
        """
        Add a dependency edge between two gates.
        
        Args:
            from_gate: Source gate ID (must execute before to_gate)
            to_gate: Target gate ID (must execute after from_gate)
        """
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
        """
        Return gates in topological order (respecting dependencies).
        
        Returns:
            List of gate IDs in topological order
        """
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
        """
        Get the depth (critical path length) of the circuit.
        
        Returns:
            Maximum layer number + 1
        """
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

    # ==================== DQC-Specific Methods ====================
    
    # def get_cross_partition_gates(self) -> List[str]:
    #     """
    #     Find gates that span multiple processors (require communication).
    #     These are multi-qubit gates where the qubits belong to different processors.
        
    #     Returns:
    #         List of gate IDs that require inter-processor communication
    #     """
    #     cross_gates = []
    #     for gate_id, gate in self.gates.items():
    #         if len(gate.qubits) < 2:
    #             continue
            
    #         # Get processors for all qubits in this gate
    #         processors = set()
    #         for q in gate.qubits:
    #             if q in self.qubit_to_processor:
    #                 processors.add(self.qubit_to_processor[q])
            
    #         # If qubits belong to more than one processor, it's a cross-partition gate
    #         if len(processors) > 1:
    #             cross_gates.append(gate_id)
        
    #     return cross_gates
    
    # def get_communication_edges(self) -> List[Tuple[str, str]]:
    #     """
    #     Find edges that cross processor boundaries (require quantum communication).
    #     This includes both:
    #     1. Edges between gates on different processors
    #     2. Cross-partition gates (marked with predecessor/successor edges)
        
    #     Returns:
    #         List of (from_gate, to_gate) tuples requiring communication
    #     """
    #     comm_edges = []
    #     for u, v in self.dag.edges():
    #         proc_u = self.gates[u].processor
    #         proc_v = self.gates[v].processor
    #         if proc_u is not None and proc_v is not None and proc_u != proc_v:
    #             comm_edges.append((u, v))
    #     return comm_edges
    
    # def get_communication_cost(self) -> int:
    #     """
    #     Calculate total quantum communication cost.
    #     This is the number of cross-partition gates (each requires teleportation).
        
    #     Returns:
    #         Number of gates requiring inter-processor communication
    #     """
    #     return len(self.get_cross_partition_gates())
    
    # def get_gates_by_processor(self, processor: str) -> List[str]:
    #     """
    #     Get all gates assigned to a specific processor.
        
    #     Args:
    #         processor: Processor identifier
            
    #     Returns:
    #         List of gate IDs assigned to this processor
    #     """
    #     return [gid for gid, gate in self.gates.items() 
    #             if gate.processor == processor]
    
    # def get_processor_stats(self) -> Dict[str, Dict[str, int]]:
    #     """
    #     Get statistics for each processor.
        
    #     Returns:
    #         Dictionary with processor stats (gate_count, depth, etc.)
    #     """
    #     stats = {}
    #     processors = set(g.processor for g in self.gates.values() if g.processor)
        
    #     for proc in processors:
    #         gate_ids = self.get_gates_by_processor(proc)
    #         subdag = self.dag.subgraph(gate_ids)
            
    #         stats[proc] = {
    #             'gate_count': len(gate_ids),
    #             'two_qubit_gates': sum(1 for gid in gate_ids 
    #                                   if len(self.gates[gid].qubits) == 2),
    #             'depth': len(list(nx.dag_longest_path(subdag))) if gate_ids else 0
    #         }
        
    #     return stats
    
    # def partition_by_qubits(self, 
    #                        qubit_groups: List[List[int]]) -> Dict[str, str]:
    #     """
    #     Partition circuit based on qubit groups (simple processor assignment).
        
    #     Args:
    #         qubit_groups: List of qubit lists, e.g., [[0,1], [2,3]] for 2 processors
            
    #     Returns:
    #         Dictionary mapping gate_id to processor_id
    #     """
    #     # First, record qubit-to-processor mapping
    #     self.qubit_to_processor.clear()
    #     for proc_idx, qubit_group in enumerate(qubit_groups):
    #         proc_id = f"QP{proc_idx}"
    #         for q in qubit_group:
    #             self.qubit_to_processor[q] = proc_id
        
    #     # Now assign gates to processors
    #     assignment = {}
    #     for gate_id, gate in self.gates.items():
    #         # Get all processors involved in this gate's qubits
    #         processors = set()
    #         for q in gate.qubits:
    #             if q in self.qubit_to_processor:
    #                 processors.add(self.qubit_to_processor[q])
            
    #         if len(processors) == 1:
    #             # All qubits on same processor - assign gate there
    #             proc_id = processors.pop()
    #             self.assign_processor(gate_id, proc_id)
    #             assignment[gate_id] = proc_id
    #         elif len(processors) > 1:
    #             # Cross-partition gate - mark in metadata but don't assign processor
    #             self.gates[gate_id].metadata['cross_partition'] = True
    #             self.gates[gate_id].metadata['spans_processors'] = list(processors)
    #             # Assign to the processor of the first qubit (control qubit convention)
    #             # This helps with visualization, but the gate still needs communication
    #             primary_proc = self.qubit_to_processor[gate.qubits[0]]
    #             self.assign_processor(gate_id, primary_proc)
    #             assignment[gate_id] = primary_proc
        
    #     return assignment
    
    # ==================== Visualization ====================
    
    def draw(self, 
            layout: str = 'layered',
            show_processors: bool = True,
            show_qubits: bool = True,
            figsize: Tuple[int, int] = (12, 8),
            save_path: Optional[str] = None):
        """
        Visualize the DAG similar to Qiskit's DAG visualization.
        
        Args:
            layout: Layout algorithm ('layered', 'spring', 'kamada_kawai')
            show_processors: Color nodes by processor assignment
            show_qubits: Show qubit labels on edges
            figsize: Figure size
            save_path: Path to save figure (None to display)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute layers for layered layout
        if layout == 'layered':
            layers = self.compute_layers()
            pos = self._layered_layout(layers)
        elif layout == 'spring':
            pos = nx.spring_layout(self.dag, k=2, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.dag)
        else:
            pos = nx.spring_layout(self.dag)
        
        # Identify cross-partition gates
        cross_partition_gates = set(self.get_cross_partition_gates())
        
        # Color nodes by processor if assigned
        if show_processors:
            processors = list(set(g.processor for g in self.gates.values() 
                                if g.processor is not None))
            
            # Updated colormap code (Matplotlib 3.7+ compatible)
            try:
                # New way (Matplotlib 3.7+)
                import matplotlib as mpl
                color_map = mpl.colormaps.get_cmap('Set3')
            except AttributeError:
                # Fallback for older Matplotlib versions
                color_map = plt.cm.get_cmap('Set3')
            
            # Generate colors for each processor
            processor_colors = {proc: color_map(i / max(len(processors), 1)) 
                            for i, proc in enumerate(processors)}
            
            node_colors = []
            for gate_id in self.dag.nodes():
                proc = self.gates[gate_id].processor
                if proc is None:
                    node_colors.append('lightgray')
                else:
                    node_colors.append(processor_colors[proc])
        else:
            node_colors = 'lightblue'
        
        # Draw nodes (regular gates)
        regular_nodes = [g for g in self.dag.nodes() if g not in cross_partition_gates]
        cross_nodes = [g for g in self.dag.nodes() if g in cross_partition_gates]
        
        if regular_nodes:
            regular_colors = [node_colors[list(self.dag.nodes()).index(n)] 
                            for n in regular_nodes] if isinstance(node_colors, list) else node_colors
            nx.draw_networkx_nodes(self.dag, pos, 
                                nodelist=regular_nodes,
                                node_color=regular_colors,
                                node_size=800,
                                alpha=0.9,
                                ax=ax)
        
        # Draw cross-partition gates with red border
        if cross_nodes:
            cross_colors = [node_colors[list(self.dag.nodes()).index(n)] 
                          for n in cross_nodes] if isinstance(node_colors, list) else node_colors
            nx.draw_networkx_nodes(self.dag, pos,
                                nodelist=cross_nodes,
                                node_color=cross_colors,
                                node_size=800,
                                alpha=0.9,
                                edgecolors='red',
                                linewidths=3,
                                ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.dag, pos,
                            edge_color='gray',
                            arrows=True,
                            arrowsize=20,
                            width=2,
                            alpha=0.6,
                            ax=ax,
                            connectionstyle='arc3,rad=0.1')
        
        # Highlight communication edges in red
        comm_edges = self.get_communication_edges()
        if comm_edges:
            nx.draw_networkx_edges(self.dag, pos,
                                edgelist=comm_edges,
                                edge_color='red',
                                arrows=True,
                                arrowsize=20,
                                width=3,
                                alpha=0.8,
                                ax=ax,
                                connectionstyle='arc3,rad=0.1')
        
        # Draw labels
        labels = {}
        for gate_id, gate in self.gates.items():
            if show_qubits:
                qubit_str = ','.join(map(str, gate.qubits))
                labels[gate_id] = f"{gate.gate_type}\nq{qubit_str}"
            else:
                labels[gate_id] = gate.gate_type
        
        nx.draw_networkx_labels(self.dag, pos, labels, 
                            font_size=8,
                            font_weight='bold',
                            ax=ax)
        
        # Add legend for processors
        if show_processors and processors:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=processor_colors[proc],
                                        markersize=10, label=proc)
                            for proc in processors]
            # Add cross-partition indicator to legend
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor='white',
                                            markeredgecolor='red',
                                            markeredgewidth=2,
                                            markersize=10, 
                                            label='Cross-partition'))
            ax.legend(handles=legend_elements, loc='upper right', 
                    title='Processors')
        
        cross_count = len(cross_partition_gates)
        ax.set_title(f'Quantum Circuit DAG\n'
                    f'Gates: {self.get_gate_count()}, '
                    f'Depth: {self.get_depth()}, '
                    f'Cross-partition Gates: {cross_count}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    def _layered_layout(self, layers: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
        """
        Create a layered layout where gates are positioned by their depth layer.
        
        Args:
            layers: Dictionary mapping gate_id to layer number
            
        Returns:
            Position dictionary for networkx drawing
        """
        pos = {}
        layer_gates = {}
        
        # Group gates by layer
        for gate_id, layer in layers.items():
            if layer not in layer_gates:
                layer_gates[layer] = []
            layer_gates[layer].append(gate_id)
        
        # Position gates
        max_layer = max(layers.values())
        for layer, gates in layer_gates.items():
            x = layer / (max_layer + 1) if max_layer > 0 else 0.5
            num_gates = len(gates)
            
            for i, gate_id in enumerate(gates):
                y = (i + 1) / (num_gates + 1)
                pos[gate_id] = (x, y)
        
        return pos
    
    def print_stats(self):
        """Print comprehensive statistics about the DAG."""
        cross_partition = self.get_cross_partition_gates()
        
        print("=" * 60)
        print("Quantum Circuit DAG Statistics")
        print("=" * 60)
        print(f"Total Gates: {self.get_gate_count()}")
        print(f"Two-Qubit Gates: {self.get_two_qubit_gate_count()}")
        print(f"Circuit Depth: {self.get_depth()}")
        print(f"Number of Qubits: {self.num_qubits}")
        print(f"Cross-partition Gates: {len(cross_partition)}")
        
        if cross_partition:
            print("\nCross-partition Gates (require teleportation):")
            for gate_id in cross_partition:
                gate = self.gates[gate_id]
                spans = gate.metadata.get('spans_processors', [])
                print(f"  {gate} spans {spans}")
        
        proc_stats = self.get_processor_stats()
        if proc_stats:
            print("\nProcessor Statistics:")
            for proc, stats in proc_stats.items():
                print(f"  {proc}:")
                print(f"    Gates: {stats['gate_count']}")
                print(f"    Two-Qubit Gates: {stats['two_qubit_gates']}")
                print(f"    Local Depth: {stats['depth']}")
        
        print("=" * 60)