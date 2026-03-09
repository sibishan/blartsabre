from architecture import QubitNetworkGraph
import networkx as nx
import matplotlib.pyplot as plt

BLART_EDGE_WEIGHT = 3

class BLARTNetworkGraph(QubitNetworkGraph):
    def __init__(self, *args, blart_edge_groups=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.core_node_groups = [list(i) for i in nx.connected_components(self.graph)]
        self.num_cores = len(self.core_node_groups)
        self.qubit_core_map = [i for i, sublist in enumerate(self.core_node_groups) for _ in sublist]

        self.data_edges = list(self.graph.edges())
        self.blart_edges = []
        self.blart_edge_groups = blart_edge_groups

        for blart_edge in blart_edge_groups:
            core1_qubits, core2_qubits = blart_edge
            for q1 in core1_qubits:
                for q2 in core2_qubits:
                    self.blart_edges.append((q1, q2))

        self.graph.add_edges_from(self.blart_edges)

        for u, v in self.data_edges:
            self.graph[u][v]["weight"] = 1.0
            self.graph[u][v]["type"] = "data"

        for u, v in self.blart_edges:
            self.graph[u][v]["weight"] = BLART_EDGE_WEIGHT
            self.graph[u][v]["type"] = "blart"

        self.distance_matrix = dict(nx.floyd_warshall(self.graph, weight="weight"))

    def draw(self):
        abstracted_graph = nx.Graph(self.data_edges)
        abstract_edges = []
        abstract_nodes = []
        for i in range(len(self.blart_edge_groups)):
            abstract_nodes.append(f"{i}a")
            abstract_nodes.append(f"{i}b")
            abstract_edges.append((f"{i}a", f"{i}b"))
            for q1 in self.blart_edge_groups[i][0]:
                abstract_edges.append((q1, f"{i}a"))
            for q2 in self.blart_edge_groups[i][1]:
                abstract_edges.append((q2, f"{i}b"))

        abstracted_graph.add_nodes_from(abstract_nodes)
        abstracted_graph.add_edges_from(abstract_edges)

        pos = nx.spring_layout(abstracted_graph, iterations=500)
        nx.draw_networkx_nodes(abstracted_graph, pos, nodelist=self.nodes(), node_shape="o", 
                              linewidths=1, edgecolors="black", node_color="white")
        nx.draw_networkx_nodes(abstracted_graph, pos, nodelist=abstract_nodes, node_shape="h", 
                              linewidths=1, edgecolors="red", node_color="white", node_size=500)
        nx.draw_networkx_edges(abstracted_graph, pos, edgelist=abstract_edges, edge_color="red")
        nx.draw_networkx_edges(abstracted_graph, pos, edgelist=self.data_edges, edge_color="black")
        nx.draw_networkx_labels(abstracted_graph, pos)
        plt.show()
    
    def __len__(self):
        return len(self.graph)

    def get_p_qubit_core(self, p):
        return self.qubit_core_map[p]

    def check_execute_remote_gate_simul(self, remote_gates):
        """
        Input: list of tuples of pairs of ints representing target p_qubits of remote gate

        Output: int representing how many of the first n gates can be run simultaneously
        """
        blart_edge_capacity = [0 for edge_group in self.blart_edge_groups]
        num_simul_remote_gates = 0
        used_qubits = []
        for remote_gate in remote_gates:
            q1, q2 = remote_gate
            if q1 in used_qubits or q2 in used_qubits:
                # re-used qubit
                return num_simul_remote_gates
            for i in range(len(self.blart_edge_groups)):
                core1_qubits, core2_qubits = self.blart_edge_groups[i]
                if (q1 in core1_qubits and q2 in core2_qubits) or (q2 in core1_qubits and q1 in core2_qubits):
                    if blart_edge_capacity[i] < 2:
                        blart_edge_capacity[i] += 1
                        num_simul_remote_gates += 1
                        used_qubits.extend([q1, q2])
                        break
            else:
                # No blart edges with <2 remote gates can facilitate remote gate
                return num_simul_remote_gates
        return num_simul_remote_gates


def blart_grid(core_height, core_width, core_rows, core_cols):
    edges = []
    blart_edge_groups=[]
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
            num_nodes += core_area
    
    for core_row in range(core_rows):
        for core_col in range(core_cols - 1):
            left_parity =  0 if core_row/(core_rows-1) < 0.5 or core_height%2==0 or core_height<2 else -1
            right_parity = 0 if core_row/(core_rows-1) < 0.5 or core_height%2==0 or core_height<2 else -1
            left_core  = core_row*core_area*core_cols +  core_col     *core_area
            right_core = core_row*core_area*core_cols + (core_col + 1)*core_area
            left_node  = left_core  + (int((core_height - 1)/2) + left_parity) * core_width + core_width - 1
            right_node = right_core + (int((core_height - 1)/2) + right_parity) * core_width
            blart_edge_groups.append((range(left_node,min(left_node+core_width+1,left_core + core_area),core_width), range(right_node,min(right_node+core_width+1,right_core + core_area - core_width + 1),core_width)))
    for core_col in range(core_cols):
        for core_row in range(core_rows - 1):
            up_parity =   0 if core_col/core_cols < 0.5 or core_width%2==0 or core_width<2 else -1
            down_parity = 0 if core_col/core_cols < 0.5 or core_width%2==0 or core_width<2 else -1
            up_core   =  core_row     *core_area*core_cols + core_col*core_area
            down_core = (core_row + 1)*core_area*core_cols + core_col*core_area
            up_node   = up_core   + (int((core_width - 1)/2) + up_parity) + (core_height - 1) * core_width
            down_node = down_core + (int((core_width - 1)/2) + down_parity)
            blart_edge_groups.append((range(up_node,min(up_node+2,up_core + core_area)), range(down_node,min(down_node+2,down_core + core_width))))

    return BLARTNetworkGraph(edges, blart_edge_groups = blart_edge_groups)


def tokyo_edges(offset=0):
    edges = []

    # 4 horizontal chains: 0-1-2-3-4, 5-6-7-8-9, 10-11-12-13-14, 15-16-17-18-19
    for start in (0, 5, 10, 15):
        for i in range(start, start + 4):
            edges.append((i + offset, i + 1 + offset))

    # vertical links i <-> i+5 for i = 0..14
    for i in range(0, 15):
        edges.append((i + offset, i + 5 + offset))

    # diagonals
    for i in (1, 3, 5, 7, 11, 13):
        edges.append((i + offset, i + 6 + offset))

    for i in (2, 4, 6, 8, 12, 14):
        edges.append((i + offset, i + 4 + offset))

    return edges

@staticmethod
def blart_two_tokyo():
    data_edges = []
    data_edges += tokyo_edges(offset=0)
    data_edges += tokyo_edges(offset=20)

    blart_edge_groups = [
        ([9, 4],  [20, 21]),   # group 0: 0a side, 0b side
        ([14, 19], [24, 23]),  # group 1: 1a side, 1b side
    ]

    return BLARTNetworkGraph(
        data_edges,
        blart_edge_groups=blart_edge_groups,
        name="Two connected IBM Q Tokyo BLART (40 qubits, 2 corees)"
    )


@staticmethod
def blart_four_tokyo():
    data_edges = []
    data_edges += tokyo_edges(offset=0)
    data_edges += tokyo_edges(offset=20)
    data_edges += tokyo_edges(offset=40)
    data_edges += tokyo_edges(offset=60)

    blart_edge_groups = [
        ([4, 9], [35, 36]),
        ([14, 19], [37, 38]),
        
        ([24, 29], [50, 55]),
        ([34, 39], [40, 45]),
        
        ([41, 42], [60, 65]),
        ([43, 44], [70, 75]),

        ([17,18], [61, 62]),
        ([15,16], [63, 64])
    ]

    return BLARTNetworkGraph(
        data_edges,
        blart_edge_groups=blart_edge_groups,
        name="Four connected IBM Q Tokyo BLART (80 qubits, 4 cores)"
    )
