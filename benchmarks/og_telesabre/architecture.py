import networkx as nx


class Edge:
    # physical qubits, undirected edge
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

class TeleportEdge:
    # physical qubits, directed triadic hyperedge
    def __init__(self, p_source, p_mediator, p_target):
        self.p_source = p_source
        self.p_mediator = p_mediator
        self.p_target = p_target

class Architecture:
    def __init__(self, num_qubits, qubit_to_core, intra_core_edges, inter_core_edges=None, name="arch"):
        self.name = name

        self.num_qubits = int(num_qubits)
        self.qubit_to_core = list(qubit_to_core)
        if len(self.qubit_to_core) != self.num_qubits:
            raise ValueError(f"Invalid qubit_to_core mapping: expected length {num_qubits}")
        
        self.num_cores = (max(self.qubit_to_core) + 1) if self.num_qubits > 0 else 0

        self.edges = list(intra_core_edges)
        self.inter_core_edges = list(inter_core_edges or [])

        self.qubit_to_edges = []
        self.teleport_edges = []
        self.communication_qubits = []
        self.qubit_to_teleport_edges_as_source = []
        self.qubit_to_teleport_edges_as_mediator = []
        self.qubit_to_teleport_edges_as_target = []
        self.qubit_to_teleport_edges = []

        self.swap_duration = 3

        self.tp_source_busy_offset = 1
        self.tp_source_busy_duration = 3

        self.tp_mediator_busy_offset = 0
        self.tp_mediator_busy_duration = 3

        self.tp_target_busy_offset = 0
        self.tp_target_busy_duration = 5

        self.teleport_duration = max(
            self.tp_source_busy_offset + self.tp_source_busy_duration,
            self.tp_mediator_busy_offset + self.tp_mediator_busy_duration,
            self.tp_target_busy_offset + self.tp_target_busy_duration
        )

        self.update_qubit_to_edges()
        self.build_teleport_edges()

        self.num_edges = len(self.edges)
        self.num_tp_edges = len(self.teleport_edges)

        self.communication_qubits = list(set(self.communication_qubits))

        self.core_comm_qubits = [[] for _ in range(self.num_cores)]
        for p in self.communication_qubits:
            self.core_comm_qubits[self.qubit_to_core[p]].append(p)
        
        self.core_qubits = [[] for _ in range(self.num_cores)]
        for p in range(self.num_qubits):
            self.core_qubits[self.qubit_to_core[p]].append(p)

    def update_qubit_to_edges(self):
        self.qubit_to_edges = [[] for _ in range(self.num_qubits)]
        for i, edge in enumerate(self.edges):
            self.qubit_to_edges[edge.p1].append(i)
            self.qubit_to_edges[edge.p2].append(i)
    
    def build_teleport_edges(self):
        self.teleport_edges = []
        self.communication_qubits = []

        for edge in self.inter_core_edges:
            p1, p2 = edge.p1, edge.p2

            self.communication_qubits.append(p1)
            self.communication_qubits.append(p2)

            # forward: neighbours of p1 teleport via mediator p1 to p2
            for e_idx in self.qubit_to_edges[p1]:
                le = self.edges[e_idx]
                p1_neighbour = le.p1 if le.p1 != p1 else le.p2
                self.teleport_edges.append(TeleportEdge(p_source=p1_neighbour, p_mediator=p1, p_target=p2))
            
            # reverse: neighbours of p2 teleport via mediator p2 to p1
            for e_idx in self.qubit_to_edges[p2]:
                le = self.edges[e_idx]
                p2_neighbour = le.p1 if le.p1 != p2 else le.p2
                self.teleport_edges.append(TeleportEdge(p_source=p2_neighbour, p_mediator=p2, p_target=p1))
        
        # build indices for teleport edges
        self.qubit_to_teleport_edges_as_source = [[] for _ in range(self.num_qubits)]
        self.qubit_to_teleport_edges_as_mediator = [[] for _ in range(self.num_qubits)]
        self.qubit_to_teleport_edges_as_target = [[] for _ in range(self.num_qubits)]
        self.qubit_to_teleport_edges = [[] for _ in range(self.num_qubits)]

        for e, te in enumerate(self.teleport_edges):
            self.qubit_to_teleport_edges_as_source[te.p_source].append(e)
            self.qubit_to_teleport_edges_as_mediator[te.p_mediator].append(e)
            self.qubit_to_teleport_edges_as_target[te.p_target].append(e)

            self.qubit_to_teleport_edges[te.p_source].append(e)
            self.qubit_to_teleport_edges[te.p_mediator].append(e)
            self.qubit_to_teleport_edges[te.p_target].append(e)

    # APIs

    def is_comm_qubit(self, qubit):
        return qubit in self.communication_qubits
    
    def get_qubit_core(self, qubit):
        return self.qubit_to_core[qubit]
    
    def get_core_distance_matrix(self):
        core_graph = nx.Graph()
        core_graph.add_edges_from([(self.get_qubit_core(e.p1), self.get_qubit_core(e.p2)) for e in self.inter_core_edges])
        return nx.floyd_warshall_numpy(core_graph, nodelist=range(self.num_cores), weight=None)
    
    def build_weighted_graph(self, local_w=1, teleport_w=None):
        if teleport_w is None:
            teleport_w = self.teleport_duration

        G = nx.Graph()
        for e in self.edges:
            G.add_edge(e.p1, e.p2, weight=local_w)
        for e in self.inter_core_edges:
            G.add_edge(e.p1, e.p2, weight=teleport_w)
        return G
    
    def distance(self, q1, q2, local_w=1, teleport_w=None):
        G = self.build_weighted_graph(local_w=local_w, teleport_w=teleport_w)
        return nx.dijkstra_path_length(G, q1, q2, weight="weight")

@staticmethod
def STAR_LINE_RING():
    num_qubits = 15
    qubit_to_core = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]

    # core 0: line 0-1-2-3-4
    edges0 = [Edge(0,1), Edge(1,2), Edge(2,3), Edge(3,4)]
    # core 1: star centred at 5
    edges1 = [Edge(5,6), Edge(5,7), Edge(5,8), Edge(5,9)]
    # core 2: ring
    edges2 = [Edge(10,11), Edge(11,12), Edge(12,13), Edge(13,14), Edge(14,10)]

    intra_core_edges = edges0 + edges1 + edges2

    # inter core links between communication qubits
    inter_core_edges = [
        Edge(2,5), # core0 comm qubit 2 <-> core1 comm qubit 5
        Edge(3,10) # core0 comm qubit 3 <-> core1 comm qubit 10
    ]

    arch = Architecture(
    num_qubits=num_qubits,
    qubit_to_core=qubit_to_core,
    intra_core_edges=intra_core_edges,
    inter_core_edges=inter_core_edges,
    name="star-line-ring"
    )

    return arch

