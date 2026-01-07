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
 
class DistributedQubitNetworkGraph(QubitNetworkGraph):
    def __init__(self, *args, comm_edges=[], **kwargs):
        super(DistributedQubitNetworkGraph, self).__init__(*args, **kwargs)
        self.add_edges_from(comm_edges, type='comm', weight = COMM_EDGE_WEIGHT)

if __name__ == '__main__':
    graph = QubitNetworkGraph([(0,1),(1,2),(1,3),(3,4)], name="Ourense")
    print(graph)
    print(graph.get_distance_matrix())
    subax1 = plt.subplot(121)
    nx.draw(graph)

    distributed_graph = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)],comm_edges=[(1,6)])
    subax1 = plt.subplot(122)
    nx.draw(distributed_graph)
    print(distributed_graph)
    # print(distributed_graph.get_distance_matrix())
    plt.show() 
