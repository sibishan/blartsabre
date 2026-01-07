from typing import List, Set

class Graph:
    """
    (V,E) representation of a graph network

    Attributes:
        num_vertices: Number of vertices in graph (immutable from initialisation)
        edges: List of edges, represented as paired Sets of indices
    """
    num_vertices: int
    edges: List[Set[int, int]]

    def __init__(self, num_vertices: int = 0, edges: List[Set[int, int]] = []):
        self.num_vertices = num_vertices
        self.edges = edges

    def add_edge(self, edge: Set[int, int]):
        if edge[0] >= self.num_vertices or edge[1] >= self.num_vertices:
            raise ValueError("Edge vertex does not exist")
        self.edges.append(edge)
    
    def remove_edge(self, edge: Set[int, int]):
        if edge in self.edges:
            self.edges.remove(edge)
        else:
            raise ValueError("Edge does not exist")
    
    def get_distance_matrix(self):
        """
        Applies Floyd-Warshall algorithm to get distance matrix between any two qubit indices

        TODO
        """
        return []
    
class QubitNetworkGraph(Graph):
    """
    (V,E) representation of a qubit connectivity graph network

    Attributes:
        num_qubits: Number of qubits in network (immutable from initialisation)
        edges: List of edges, represented as paired Sets of indices
        name: Network architecture name
    """
    name: str

    def __init__(self, num_qubits: int = 0, edges: List[Set[int, int]] = [], name: str = ""):
        super().__init__(num_qubits, edges)
        self.name = name

class DistributedQubitNetworkGraph(Graph):
    """
    (V,E) representation of a distributed qubit connectivity graph network

    Attributes:
        num_qubits: Number of qubits in network (immutable from initialisation)
        data_edges: List of intra-core edges, represented as paired Sets of indices
        comm_edges: List of inter-core edges, represented as paired Sets of indices
        name: Network architecture name
    """
    comm_edges: List[Set[int, int]]
    def __init__(self, num_qubits: int = 0, data_edges: List[Set[int, int]] = [], comm_edges: List[Set[int, int]] = [], name: str = ""):
        super().__init__(num_qubits, data_edges, name)
        self.comm_edges = comm_edges