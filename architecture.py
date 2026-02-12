from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from mapping import Mapping

COMM_EDGE_WEIGHT = 2.0
    
class QubitNetworkGraph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(QubitNetworkGraph, self).__init__(*args, **kwargs)
        nx.set_edge_attributes(self, 'data', 'type')
        self.distance_matrix = nx.floyd_warshall(self, weight="weight")
        self.pos = nx.spring_layout(self, iterations=200)


    def get_distance_matrix(self):
        return self.distance_matrix
    
    def check_gate_executable(self, gate, mapping: Mapping):
        if len(gate.qubits) < 2:
            return True
        
        q1, q2 = gate.qubits
        return self.has_edge(mapping.l_to_p(q1),mapping.l_to_p(q2))
    
    def draw(self):
        nx.draw(self, pos = self.pos)
        plt.show()
        
class DistributedQubitNetworkGraph(QubitNetworkGraph):

    def __init__(self, *args, core_node_groups=[], **kwargs):
        super(DistributedQubitNetworkGraph, self).__init__(*args, **kwargs)
        self.core_subgraphs = []

        self.core_node_groups = core_node_groups
        self.qubit_core_map = [i for i, sublist in enumerate(core_node_groups) for _ in sublist]
        
        self.num_cores = (max(self.qubit_core_map) + 1) if self.number_of_nodes() > 0 else 0

        self.active_telegate_nodes = ()

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

        for u, v in self.comm_edges:
            self.edges[u, v]['weight'] = 2

        self.distance_matrix = nx.floyd_warshall(self, weight="weight")
        self.separated_core_distance_matrix = nx.floyd_warshall(self.separated_core_graph)

        
    def check_gate_executable(self, gate, mapping):
        if len(gate.qubits) < 2:
            return True
        
        q1, q2 = gate.qubits

        if mapping.l_to_p(q1) in self.active_telegate_nodes and mapping.l_to_p(q2) in self.active_telegate_nodes:
            self.clear_active_telegate_qubits()
            return True
        
        return self.separated_core_graph.has_edge(mapping.l_to_p(q1),mapping.l_to_p(q2))
    
    def get_separated_core_distance_matrix(self):
        return self.separated_core_distance_matrix

    def draw(self):
        nx.draw_networkx_nodes(self, self.pos, nodelist=self.comm_qubits, node_shape="h",
                                linewidths=1, edgecolors="black",node_color="white")
        nx.draw_networkx_nodes(self, self.pos, nodelist=self.non_comm_qubits, node_shape="o",
                                linewidths=1, edgecolors="black",node_color="white")
        nx.draw_networkx_edges(self, self.pos, edgelist=self.comm_edges, edge_color="red")
        nx.draw_networkx_edges(self, self.pos, edgelist=self.data_edges, edge_color="black")
        nx.draw_networkx_labels(self, self.pos)

        plt.show()
    
    def draw_mapping(self, mapping: Mapping):
        free_nodes_set = set(mapping.get_free_p_nodes())
        nx.draw_networkx_nodes(self, self.pos, nodelist=set(self.comm_qubits) & free_nodes_set, node_shape="h",
                                linewidths=1, edgecolors="black",node_color="white")
        nx.draw_networkx_nodes(self, self.pos, nodelist=set(self.comm_qubits) - free_nodes_set, node_shape="h",
                                linewidths=1, edgecolors="black",node_color="grey")
        nx.draw_networkx_nodes(self, self.pos, nodelist=set(self.non_comm_qubits) & free_nodes_set, node_shape="o",
                                linewidths=1, edgecolors="black",node_color="white")
        nx.draw_networkx_nodes(self, self.pos, nodelist=set(self.non_comm_qubits) - free_nodes_set, node_shape="o",
                                linewidths=1, edgecolors="black",node_color="grey")
        nx.draw_networkx_edges(self, self.pos, edgelist=self.comm_edges, edge_color="red")
        nx.draw_networkx_edges(self, self.pos, edgelist=self.data_edges, edge_color="black")
        nx.draw_networkx_labels(self, self.pos)

        shifted_pos = {}
        for node, coordinates in self.pos.items():
            shifted_pos[node] = (coordinates[0] + 0.05, coordinates[1] + 0.08)
        nx.draw_networkx_labels(self, shifted_pos, labels=mapping.inv, font_color="Red")

        plt.show()
    
    def  get_nth_nearest_intercore_free_qubit(self, mapping: Mapping, node, n = 0):
        free_nodes = mapping.get_free_p_nodes()
        core = self.qubit_core_map[node]
        core_free_nodes = [free_node for free_node in free_nodes if self.qubit_core_map[free_node] == core]

        if len(core_free_nodes) <= n:
            return None

        free_node_distance_map = {free_node: self.get_distance_matrix()[node][free_node]
                                for free_node in core_free_nodes}
        sorted_map = sorted(free_node_distance_map.items(), key=lambda item: item[1])
        nearest_free = sorted_map[n][0]
        return nearest_free

    def get_nth_nearest_free_qubit_map(self, mapping: Mapping, n = 0):
        free_nodes = mapping.get_free_p_nodes()
        core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in self.core_node_groups]

        free_qubit_map = dict()

        for comm_node in self.comm_qubits:
            core = self.qubit_core_map[comm_node]
            core_free_nodes = core_free_nodes_map[core]

            if len(core_free_nodes) <= n:
                continue

            free_node_distance_map = {free_node: self.get_separated_core_distance_matrix()[comm_node][free_node]
                                    for free_node in core_free_nodes}
            sorted_map = sorted(free_node_distance_map.items(), key=lambda item: item[1])
            nearest_free = sorted_map[n][0]
            # nearest_free = min(free_node_distance_map, key=free_node_distance_map.get)
            free_qubit_map[comm_node] = nearest_free
        
        return free_qubit_map

    def get_full_cores(self, mapping: Mapping):
        free_nodes = mapping.get_free_p_nodes()
        core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in self.core_node_groups]

        return [len(core_free_nodes_map) < 2 for core_free_nodes_map in core_free_nodes_map]

    def get_core_capacity(self, mapping: Mapping):
        free_nodes = mapping.get_free_p_nodes()
        core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in self.core_node_groups]

        return [len(core_free_nodes_map) for core_free_nodes_map in core_free_nodes_map]
    
    def register_active_telegate_qubits(self, p_q1, p_q2):
        self.active_telegate_nodes = (p_q1, p_q2)

    def clear_active_telegate_qubits(self):
        self.active_telegate_nodes = ()



def tokyo(offset=0):
    edges = []

    for start in (0, 5, 10, 15):
        for i in range(start, start + 4):
            edges.append((i + offset, i + 1 + offset))

    for i in range(0, 15):
        edges.append((i + offset, i + 5 + offset))

    for i in (1, 3, 5, 7, 11, 13):
        edges.append((i + offset, i + 6 + offset))

    for i in (2, 4, 6, 8, 12, 14):
        edges.append((i + offset, i + 4 + offset))

    return QubitNetworkGraph(edges, name=f"IBM Q Tokyo (20 qubits, offset {offset})")

def two_tokyo():
    # base edges for the two 20-qubit tokyo graphs
    edges = []
    edges += list(tokyo(offset=0).edges())
    edges += list(tokyo(offset=20).edges())

    # single inter-core link (communication edge)
    edges += [(4, 20), (19,35)]

    core_node_groups = [
        list(range(0, 20)),   # core 0
        list(range(20, 40)),  # core 1
    ]

    arch = DistributedQubitNetworkGraph(
        edges,
        core_node_groups=core_node_groups,
        name="Two connected IBM Q Tokyo (40 qubits, 2 cores)"
    )

    return arch

def four_tokyo():
    edges = []
    edges += list(tokyo(offset=0).edges())
    edges += list(tokyo(offset=20).edges())
    edges += list(tokyo(offset=40).edges())
    edges += list(tokyo(offset=60).edges())

    # inter-core communication edges (aligned: right column ↔ left column)
    # Core 0 ↔ Core 1
    edges += [(4, 20), (19, 35)]
    # Core 1 ↔ Core 2
    edges += [(24, 40), (39, 55)]
    # Core 2 ↔ Core 3
    edges += [(44, 60), (59, 75)]

    core_node_groups = [
        list(range(0, 20)),    # core 0
        list(range(20, 40)),   # core 1
        list(range(40, 60)),   # core 2
        list(range(60, 80)),   # core 3
    ]

    arch = DistributedQubitNetworkGraph(
        edges,
        core_node_groups=core_node_groups,
        name="Four connected IBM Q Tokyo (80 qubits, 4 cores)"
    )
    return arch

def sycamore(offset=0):
    edges = set()

    # start from the 54-qubit Sycamore pattern (0..53), then remove node 3
    nodes = list(range(54))
    I = (
        list(range(6, 12)) +
        list(range(18, 24)) +
        list(range(30, 36)) +
        list(range(42, 48))
    )
    Iset = set(I)

    # build edges using the same rule as your nx version
    for i in I:
        for j in nodes:
            if j in Iset:
                continue
            if (i - j) in (5, 6) or (j - i) in (6, 7):
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))

    # remove node 3 and relabel nodes >3 down by 1 to get 53 qubits (0..52)
    rem = 3
    relabel = {n: (n if n < rem else n - 1) for n in nodes if n != rem}

    out_edges = []
    for u, v in edges:
        if u == rem or v == rem:
            continue
        out_edges.append((relabel[u] + offset, relabel[v] + offset))

    return QubitNetworkGraph(out_edges, name=f"Sycamore-53 (offset {offset})")


def multi_core_grid(core_height, core_width, core_rows, core_cols):
    edges = []
    core_node_groups = []
    num_nodes = 0
    core_area = core_height * core_width

    for core_row in range(core_rows):
        for core_col in range(core_cols):
            for row in range(core_height):
                for col in range(core_width - 1):
                    edges.append((row * core_width + col + num_nodes, row * core_width + col + 1 + num_nodes))
            for col in range(core_width):
                for row in range(core_height - 1):
                    edges.append((row * core_width + col + num_nodes, row * core_width + col + core_width + num_nodes))
            core_node_groups.append(list(range(num_nodes, core_area + num_nodes)))
            num_nodes += core_area
    
    for core_row in range(core_rows):
        for core_col in range(core_cols - 1):
            left_parity =  0 if core_row/(core_rows-1) < 0.5 or core_height%2==1 else -1
            right_parity = 0 if core_row/(core_rows-1) < 0.5 or core_height%2==1 else -1
            left_core  = core_row*core_area*core_cols +  core_col     *core_area
            right_core = core_row*core_area*core_cols + (core_col + 1)*core_area
            left_node  = left_core  + (int((core_height - 1)/2) + left_parity) * core_width + core_width - 1
            right_node = right_core + (int((core_height - 1)/2) + right_parity) * core_width
            edges.append((left_node, right_node))
    for core_col in range(core_cols):
        for core_row in range(core_rows - 1):
            up_parity =   0 if core_col/core_cols < 0.5 or core_width%2==1 else -1
            down_parity = 0 if core_col/core_cols < 0.5 or core_width%2==1 else -1
            up_core   =  core_row     *core_area*core_cols + core_col*core_area
            down_core = (core_row + 1)*core_area*core_cols + core_col*core_area
            up_node   = up_core   + (int((core_width - 1)/2) + up_parity) + (core_height - 1) * core_width
            down_node = down_core + (int((core_width - 1)/2) + down_parity)
            edges.append((up_node, down_node))


    return DistributedQubitNetworkGraph(edges, core_node_groups=core_node_groups)

