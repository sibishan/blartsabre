from architecture import QubitNetworkGraph
import networkx as nx
import matplotlib.pyplot as plt

COMM_EDGE_WEIGHT = 4.0

class BLARTNetworkGraph(QubitNetworkGraph):
    def __init__(self, *args, blart_edge_groups=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.data_edges = list(self.edges())
        self.blart_edges = []
        self.blart_edge_groups = blart_edge_groups

        for blart_edge in blart_edge_groups:
            core1_qubits, core2_qubits = blart_edge
            for q1 in core1_qubits:
                for q2 in core2_qubits:
                    self.blart_edges.append((q1, q2))

        self.add_edges_from(self.blart_edges)

        for u, v in self.data_edges:
            self[u][v]["weight"] = 1.0
            self[u][v]["type"] = "data"

        for u, v in self.blart_edges:
            self[u][v]["weight"] = COMM_EDGE_WEIGHT
            self[u][v]["type"] = "blart"

        self.distance_matrix = dict(nx.floyd_warshall(self, weight="weight"))

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
        nx.draw_networkx_nodes(abstracted_graph, pos, nodelist=self.nodes, node_shape="o", 
                              linewidths=1, edgecolors="black", node_color="white")
        nx.draw_networkx_nodes(abstracted_graph, pos, nodelist=abstract_nodes, node_shape="h", 
                              linewidths=1, edgecolors="red", node_color="white", node_size=500)
        nx.draw_networkx_edges(abstracted_graph, pos, edgelist=abstract_edges, edge_color="red")
        nx.draw_networkx_edges(abstracted_graph, pos, edgelist=self.data_edges, edge_color="black")
        nx.draw_networkx_labels(abstracted_graph, pos)
        plt.show()

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
            # blart_edge_groups.append((range(left_node,min(left_node+2,left_core + core_area)), range(right_node,min(right_node+2,right_core + core_height))))
            # print(left_node,left_node+core_width+1,left_core + core_area)
            # print(right_node,right_node+core_width+1,right_core + core_area - core_width + 1)
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
def blart_tokyo(offset=0):
    edges = tokyo_edges(offset=offset)
    return BLARTNetworkGraph(edges, blart_edge_groups=[], name=f"IBM Q Tokyo BLART (20 qubits, offset {offset})")

@staticmethod
def blart_two_tokyo():
    data_edges = []
    data_edges += tokyo_edges(offset=0)
    data_edges += tokyo_edges(offset=20)

    # inter-core links become BLART groups, use singletons to avoid all-to-all expansion
    blart_edge_groups = [
        ([4], [20]),
        ([19], [25]),
    ]

    return BLARTNetworkGraph(
        data_edges,
        blart_edge_groups=blart_edge_groups,
        name="Two connected IBM Q Tokyo BLART (40 qubits, 2 cores)"
    )

@staticmethod
def blart_five_tokyo():
    data_edges = []
    for k in range(5):
        data_edges += tokyo_edges(offset=20 * k)

    comm_edges = [
        (4, 20), (24, 40), (44, 60), (64, 80), (15, 99),
        (19, 25), (39, 45), (59, 65), (79, 85), (0, 4)
    ]

    blart_edge_groups = [([u], [v]) for (u, v) in comm_edges]

    return BLARTNetworkGraph(
        data_edges,
        blart_edge_groups=blart_edge_groups,
        name="Five connected IBM Q Tokyo BLART (100 qubits, 5 cores)"
    )

