from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from mapping import Mapping

COMM_EDGE_WEIGHT = 10
    
class QubitNetworkGraph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(QubitNetworkGraph, self).__init__(*args, **kwargs)
        nx.set_edge_attributes(self, 'data', 'type')
        self.distance_matrix = nx.floyd_warshall(self)

    def get_distance_matrix(self):
        return self.distance_matrix
    
    def check_gate_executable(self, gate, mapping: Mapping):
        if len(gate.qubits) < 2:
            return True
        
        q1, q2 = gate.qubits
        return self.has_edge(mapping.l_to_p(q1),mapping.l_to_p(q2))
    
    def draw(self):
        nx.draw(self)
        plt.show()

class SingleCoreDQGraph(QubitNetworkGraph):
    def __init__(self, *args, comm_edges=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.comm_edges = list(comm_edges or [])
        self.comm_edge_set = {tuple(sorted(e)) for e in self.comm_edges}
        self.data_edges = [e for e in self.edges() if tuple(sorted(e)) not in self.comm_edge_set]

        for u, v in self.edges():
            self[u][v]["weight"] = 1.0

        for u, v in self.comm_edges:
            if self.has_edge(u, v):
                self[u][v]["weight"] = COMM_EDGE_WEIGHT

        self.distance_matrix = dict(nx.floyd_warshall(self, weight="weight"))

    def is_comm_edge(self, u, v):
        return tuple(sorted((u, v))) in self.comm_edge_set

    
    def draw(self):
        """Draw graph with communication edges highlighted"""
        pos = nx.spring_layout(self)
        
        # Identify communication qubits
        comm_subgraph = nx.Graph(self.comm_edges)
        comm_qubits = list(comm_subgraph.nodes())
        non_comm_qubits = [n for n in self.nodes() if n not in comm_qubits]
        
        # Draw nodes
        nx.draw_networkx_nodes(self, pos, nodelist=comm_qubits, node_shape="h", 
                              linewidths=1, edgecolors="black", node_color="white")
        nx.draw_networkx_nodes(self, pos, nodelist=non_comm_qubits, node_shape="o", 
                              linewidths=1, edgecolors="black", node_color="white")
        
        # Draw edges
        nx.draw_networkx_edges(self, pos, edgelist=self.comm_edges, edge_color="red")
        nx.draw_networkx_edges(self, pos, edgelist=self.data_edges, edge_color="black")
        
        # Draw labels
        nx.draw_networkx_labels(self, pos)
        plt.show()
        
class DistributedQubitNetworkGraph(QubitNetworkGraph):

    def __init__(self, *args, core_node_groups=[], **kwargs):
        super(DistributedQubitNetworkGraph, self).__init__(*args, **kwargs)
        self.core_subgraphs = []

        self.core_node_groups = core_node_groups
        self.qubit_core_map = [i for i, sublist in enumerate(core_node_groups) for _ in sublist]
        
        self.num_cores = (max(self.qubit_core_map) + 1) if self.number_of_nodes() > 0 else 0

        self.separated_core_graph = nx.Graph()
        for core_node_group in core_node_groups:
            core_subgraph = self.subgraph(core_node_group)
            self.core_subgraphs.append(core_subgraph)
            self.separated_core_graph = nx.union(self.separated_core_graph,core_subgraph)
        self.data_edges = self.separated_core_graph.edges()
        self.comm_edges = self.edges() - self.data_edges
        comm_subgraph = nx.Graph(self.comm_edges)
        self.comm_qubits = comm_subgraph.nodes()
        self.non_comm_qubits = self.nodes() - self.comm_qubits
        
    def check_gate_executable(self, gate, mapping):
        if len(gate.qubits) < 2:
            return True
        
        q1, q2 = gate.qubits
        return self.separated_core_graph.has_edge(mapping.l_to_p(q1),mapping.l_to_p(q2))

    def draw(self):
        pos = nx.spring_layout(self)
        nx.draw_networkx_nodes(self, pos, nodelist=self.comm_qubits, node_shape="h", linewidths=1, edgecolors="black",node_color="white")
        nx.draw_networkx_nodes(self, pos, nodelist=self.non_comm_qubits, node_shape="o", linewidths=1, edgecolors="black",node_color="white")
        nx.draw_networkx_edges(self, pos, edgelist=self.comm_edges, edge_color="red")
        nx.draw_networkx_edges(self, pos, edgelist=self.data_edges, edge_color="black")
        nx.draw_networkx_labels(self, pos)

        plt.show()

def tokyo_arch():
    edges = []

    # 4 horizontal chains: 0-1-2-3-4, 5-6-7-8-9, 10-11-12-13-14, 15-16-17-18-19
    for start in (0, 5, 10, 15):
        for i in range(start, start + 4):
            edges.append((i, i + 1))

    # vertical links between rows: i <-> i+5 for i = 0..14
    for i in range(0, 15):
        edges.append((i, i + 5))

    # diagonals +6 for these i
    for i in (1, 3, 5, 7, 11, 13):
        edges.append((i, i + 6))

    # diagonals +4 for these i
    for i in (2, 4, 6, 8, 12, 14):
        edges.append((i, i + 4))

    return QubitNetworkGraph(edges, name="IBM Q Tokyo (20 qubits)")


if __name__ == '__main__':

    distributed_graph = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(1,6),(3,4),(1,4),(3,6)],
                                                     core_node_groups=[[0,1,2,3],[4,5,6,7]])
    distributed_graph.draw()
