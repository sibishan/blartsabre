from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt

COMM_EDGE_WEIGHT = 3
    
class QubitNetworkGraph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(QubitNetworkGraph, self).__init__(*args, **kwargs)
        nx.set_edge_attributes(self, 'data', 'type')

    def get_distance_matrix(self):
        return nx.floyd_warshall(self)
    
    def check_gate_executable(self, gate, mapping):
        q1, q2 = gate.qubits
        return self.has_edge(mapping[q1],mapping[q2])
 
class DistributedQubitNetworkGraph(QubitNetworkGraph):
    def __init__(self, *args, comm_edges=[], **kwargs):
        super(DistributedQubitNetworkGraph, self).__init__(*args, **kwargs)
        self.add_edges_from(comm_edges, type='comm', weight = COMM_EDGE_WEIGHT)

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


# if __name__ == '__main__':
#     graph = QubitNetworkGraph([(0,1),(1,2),(1,3),(3,4)], name="Ourense")
#     print(graph)
#     print(graph.get_distance_matrix())
#     subax1 = plt.subplot(121)
#     nx.draw(graph)

#     distributed_graph = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)],comm_edges=[(1,6)])
#     subax1 = plt.subplot(122)
#     nx.draw(distributed_graph)
#     print(distributed_graph)
#     plt.show() 
