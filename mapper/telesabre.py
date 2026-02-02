from convert import from_qiskit
from qiskit import QuantumCircuit
from bidict import bidict
import random
from copy import deepcopy
import networkx as nx

EXTENDED_LAYER_SIZE = 10
EXTENDED_HEURISTIC_WEIGHT = 0.5
DECAY_VALUE = 0.001
FULL_CORE_PENALTY = 10
COMM_EDGE_WEIGHT = 1.0

RESET_TIMER_START = 100

NUM_ITERATIONS = 100

class DeadlockError(RuntimeError): pass

def reduce_2_qubit_gates(circuit):
    for i in range(len(circuit.data)-1,-1,-1):
        if circuit.data[i].operation.num_qubits != 2:
            del circuit.data[i]
    return circuit

def DQC_contracted_graph(arch, temp_mapping, q1, q2):
    node1 = temp_mapping[q1]
    node2 = temp_mapping[q2]
    # Add comm nodes, comm edges & target nodes
    contracted_graph = nx.Graph(arch.comm_edges)
    nx.set_edge_attributes(contracted_graph,COMM_EDGE_WEIGHT,'weight')
    contracted_graph.add_node(node1)
    contracted_graph.add_node(node2)
    # Add edges between target and comm qubits within core
    core_node_map = [[node for node in core_group if node in contracted_graph.nodes()] for core_group in arch.core_node_groups]
    for core in core_node_map:
        for i in range(len(core)-1):
            for j in range(i+1,len(core)):
                if core[i] != core[j]:
                    contracted_graph.add_edge(core[i], core[j], weight = arch.get_distance_matrix()[core[i]][core[j]])
    
    # Add free node and full core scores
    free_nodes = [temp_mapping[i] for i in temp_mapping if i<0]
    if len(free_nodes) == 0:
        raise ValueError("No available free qubits for inter-core routing")
    core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in arch.core_node_groups]

    full_cores = [len(core_free_nodes_map) < 2 for core_free_nodes_map in core_free_nodes_map]

    for comm_node in arch.comm_qubits:
        core = arch.qubit_core_map[comm_node]

        if len(core_free_nodes_map[core]) == 0:
            raise DeadlockError()

        free_node_distances = []
        for free_node in core_free_nodes_map[core]:
            free_node_distances.append(arch.get_distance_matrix()[comm_node][free_node])

        free_node_score = min(free_node_distances)/2
        core_score = FULL_CORE_PENALTY if full_cores[core] else 0
        for u,v in contracted_graph.edges(comm_node):
            contracted_graph.edges[u, v]['weight'] = contracted_graph.edges[u, v]['weight'] + free_node_score + core_score

    # pos = nx.spring_layout(contracted_graph)
    # nx.draw_networkx_nodes(contracted_graph, pos)
    # nx.draw_networkx_edges(contracted_graph, pos)
    # nx.draw_networkx_labels(contracted_graph, pos)
    # nx.draw_networkx_edge_labels(contracted_graph, pos)

    return contracted_graph

def DQC_gate_routing_energy(arch, temp_mapping, q1, q2):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2)
    # shortest_path = nx.shortest_path(contracted_graph, source=temp_mapping[q1], target=temp_mapping[q2], weight="weight")
    shortest_path_length = nx.shortest_path_length(contracted_graph, source=temp_mapping[q1], target=temp_mapping[q2], weight="weight")

    return shortest_path_length

def DQC_gate_routing_path(arch, temp_mapping, q1, q2):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2)
    shortest_path = nx.shortest_path(contracted_graph, source=temp_mapping[q1], target=temp_mapping[q2], weight="weight")

    return shortest_path

def get_SWAP_candidates(graph, mapping, front_layer):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    front_nodes = set()
    for gate in front_layer:
        q1, q2 = gate.qubits
        if graph.qubit_core_map[mapping[q1]] == graph.qubit_core_map[mapping[q2]]:
            front_nodes.add(mapping[q1])
            front_nodes.add(mapping[q2])
    edges = graph.core_subgraph_union.edges(front_nodes)
    return edges

def get_teleport_candidates(arch, mapping, front_layer):
    # Current implementation: get all path routes of inter-core front layer qubits
    teleportations = []
    for gate in front_layer:
        q1, q2 = gate.qubits
        if arch.qubit_core_map[mapping[q1]] != arch.qubit_core_map[mapping[q2]]:
            path = DQC_gate_routing_path(arch, mapping, q1, q2)
            if path[0] not in arch.comm_qubits:
                teleportations.append(tuple(path[:3]))
            else:
                # for neighbour in arch.core_subgraph_union[path[0]]:
                    teleportations.append(tuple([path[0]]+path[:2]))
            if path[-1] not in arch.comm_qubits:
                teleportations.append(tuple(path[:-4:-1]))
            else:
                # for neighbour in arch.core_subgraph_union[path[-1]]:
                    teleportations.append(tuple([path[-1]]+path[:-3:-1]))
    return teleportations

def mapping_energy(arch, circuit_dag, temp_mapping, dist_matrix, SWAP_candidate, decay_array):
    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())

    H_basic = 0
    for gate in front_layer_gates:
        q1, q2 = gate.qubits
        if arch.qubit_core_map[temp_mapping[q1]] == arch.qubit_core_map[temp_mapping[q2]]:
            gate_energy = dist_matrix[temp_mapping[q1]][temp_mapping[q2]]
        else:
            gate_energy = DQC_gate_routing_energy(arch, temp_mapping, q1, q2)
        H_basic += gate_energy

    decay_factor = 1 + max(decay_array[SWAP_candidate[0]],decay_array[SWAP_candidate[1]])

    extended_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_extended_layer(EXTENDED_LAYER_SIZE))

    if len(extended_layer_gates) == 0:
        H = decay_factor / len(front_layer_gates) * H_basic
        return H

    H_extended = 0
    for gate in extended_layer_gates:
        q1, q2 = gate.qubits
        H_extended += dist_matrix[temp_mapping[q1]][temp_mapping[q2]]
        
    H = decay_factor / len(front_layer_gates) * H_basic + EXTENDED_HEURISTIC_WEIGHT / len(extended_layer_gates) * H_extended
    return H

def update_mapping_SWAP(mapping, p_q1, p_q2):
    temp1 = mapping.inv[p_q1]
    temp2 = mapping.inv[p_q2]
    mapping.inv[p_q1] = None
    mapping.inv[p_q2] = temp1
    mapping.inv[p_q1] = temp2
    return mapping

def swap_to_target(mapping, start, target, arch, early_stop=0):
    path = nx.shortest_path(arch,start,target)
    for i in range(len(path)-1-early_stop):
        mapping = update_mapping_SWAP(mapping, path[i], path[i+1])
    return mapping

def update_mapping_teleport(mapping, p_qstart, p_qcomm1, p_qcomm2, arch):
    l_qstart = mapping.inv[p_qstart]
    # Swap nearest free qubit to comm 1
    free_nodes = [mapping[i] for i in mapping if i<0]
    core_1 = arch.qubit_core_map[p_qcomm1]
    core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in arch.core_node_groups]
    free_node_distance_map = dict()
    for free_node in core_free_nodes_map[core_1]:
        free_node_distance_map[free_node] = arch.get_distance_matrix()[p_qcomm1][free_node]
    nearest_free_1 = min(free_node_distance_map, key=free_node_distance_map.get)
    mapping = swap_to_target(mapping, nearest_free_1, p_qcomm1, arch, early_stop=0)
    # Swap nearest free qubit to comm 2
    free_nodes = [mapping[i] for i in mapping if i<0]
    core_2 = arch.qubit_core_map[p_qcomm2]
    core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in arch.core_node_groups]
    free_node_distance_map = dict()
    for free_node in core_free_nodes_map[core_2]:
        free_node_distance_map[free_node] = arch.get_distance_matrix()[p_qcomm1][free_node]
    nearest_free_2 = min(free_node_distance_map, key=free_node_distance_map.get)
    mapping = swap_to_target(mapping, nearest_free_2, p_qcomm2, arch, early_stop=0)
    # Swap start qubit next to comm 1
    p_qstart = mapping[l_qstart]
    mapping = swap_to_target(mapping, p_qstart, p_qcomm1, arch, early_stop=1)
    # Teleport
    p_qstart = mapping[l_qstart]
    mapping = update_mapping_SWAP(mapping, p_qstart, p_qcomm2)
    return mapping
    




def sabre_forward_pass(arch, dist_matrix, initial_mapping, circuit_dag):
    circuit_dag = deepcopy(circuit_dag)
    initial_mapping = initial_mapping.copy()

    mapping = initial_mapping
    gate_execution_log = []
    decay_array = [1 for _ in range(len(arch))]
    decay_timer = [0 for _ in range(len(arch))]

    front_layer = circuit_dag.get_front_layer()
    reset_timer = RESET_TIMER_START
    while len(front_layer) != 0:
        executable_gate_nodes = []
        for gate_node in front_layer:
            gate = circuit_dag.get_gate_from_node(gate_node)
            if arch.check_gate_executable(gate, mapping):
                executable_gate_nodes.append(gate_node)
        
        if len(executable_gate_nodes) != 0:
            # There are gates ready to be executed as-is
            for gate_node in executable_gate_nodes:
                gate = circuit_dag.get_gate_from_node(gate_node)
                gate_execution_log.append((gate.gate_type + " " + str(gate.parameters),(mapping[gate.qubits[0]],mapping[gate.qubits[1]])))
                circuit_dag.remove_gate(gate_node)
                decay_array[mapping[gate.qubits[0]]] = 1
                decay_array[mapping[gate.qubits[1]]] = 1
            reset_timer = RESET_TIMER_START
        else:
            # A SWAP is required for the next gate
            score = dict()
            front_layer_gates = circuit_dag.get_gates_from_nodes(front_layer)

            SWAP_candidates = get_SWAP_candidates(arch, mapping, front_layer_gates)
            for SWAP_candidate in SWAP_candidates:
                temp_mapping = update_mapping_SWAP(mapping.copy(),*SWAP_candidate)
                score[SWAP_candidate] = mapping_energy(arch, circuit_dag, temp_mapping, dist_matrix, SWAP_candidate, decay_array)

            teleport_candidates = get_teleport_candidates(arch, mapping, front_layer_gates)
            for teleport_candidate in teleport_candidates:
                temp_mapping = update_mapping_teleport(mapping.copy(),*teleport_candidate,arch)
                score[teleport_candidate] = mapping_energy(arch, circuit_dag, temp_mapping, dist_matrix, teleport_candidate, decay_array)

            best_action = min(score, key=score.get)
            if len(best_action) == 2:
                # SWAP
                update_mapping_SWAP(mapping,*best_action)
                gate_execution_log.append(("SWAP", best_action))
            else:
                # Teleport
                update_mapping_teleport(mapping,*best_action,arch)
                gate_execution_log.append(("Teleport", best_action))
       
            decay_array[best_action[0]] = 1 + DECAY_VALUE
            decay_array[best_action[1]] = 1 + DECAY_VALUE
            for i in range(len(arch)):
                if decay_array[i]:
                    decay_timer[i]+=1
                if decay_timer[i] > 5:
                    decay_array[i] = 1
                    decay_timer[i] = 0

        front_layer = circuit_dag.get_front_layer()
        reset_timer -= 1
        if reset_timer < 0:
            raise DeadlockError

    return mapping, gate_execution_log


def telesabre(arch, quantum_circuit, verbose = False, return_log = False):
    """
    return values:
        mapping: Bidict mapping where logical qubits are keys, Physical qubits are values
    """
    num_physical_qubits = 0
    num_logical_qubits = 0
    circuit_dag = None
    reverse_circuit_dag = None

    if isinstance(quantum_circuit, QuantumCircuit):
        quantum_circuit = quantum_circuit.decompose()
        if verbose:
            print(quantum_circuit)
        num_logical_qubits = quantum_circuit.num_qubits
        num_physical_qubits = len(arch)

        reduced_quantum_circuit = reduce_2_qubit_gates(quantum_circuit)
        reverse_quantum_circuit = reduced_quantum_circuit.inverse()
        circuit_dag = from_qiskit(reduced_quantum_circuit)

        reverse_circuit_dag = from_qiskit(reverse_quantum_circuit)

        if num_logical_qubits > num_physical_qubits:
            raise ValueError("Too many circuit qubits to perform algorithm on hardware network")
    else:
        raise ValueError("SABRE Layout only accepts Qiskit QuantumCircuit")
    

    dist_matrix = arch.get_distance_matrix()

    gate_execution_log_iterations = dict()

    for iteration in range(NUM_ITERATIONS):

        try:
            random_mapping = list(range(num_logical_qubits-num_physical_qubits,num_logical_qubits))
            random.shuffle(random_mapping)
            initial_mapping = bidict([(i,j) for j,i in enumerate(random_mapping)])

            final_mapping, _ = sabre_forward_pass(arch, dist_matrix, initial_mapping, circuit_dag)
            initial_mapping, _ = sabre_forward_pass(arch, dist_matrix, final_mapping, reverse_circuit_dag)
            _, gate_execution_log = sabre_forward_pass(arch, dist_matrix, initial_mapping, circuit_dag)

            gate_execution_log_iterations[iteration] = (initial_mapping,gate_execution_log)
            print(iteration)

        except DeadlockError:
            pass

    if len(gate_execution_log_iterations)==0:
        raise DeadlockError("No valid runs found")
    
    best_iteration = min(gate_execution_log_iterations, key=lambda k: len(gate_execution_log_iterations[k][1]))
    best_initial_mapping, best_gate_execution_log = gate_execution_log_iterations[best_iteration]

    if verbose:
        best_swap_log = [k for k in best_gate_execution_log if (k[0] == "SWAP")]
        best_teleport_log = [k for k in best_gate_execution_log if (k[0] == "Teleport")]
        print("Best Iteration:")
        print(f"#{len(best_gate_execution_log)} total gates")
        print(f"#{len(best_swap_log)} inserted SWAPS")
        print(f"#{len(best_teleport_log)} inserted teleports")
        print()

        print("Initial Mapping:")
        for i in range(num_logical_qubits):
            print(f"Logical Qubit {i}: Physical Qubit {best_initial_mapping[i]}")
        print()

        print("Physical Gate Log:")
        for gate_log in best_gate_execution_log:
            print(f"{gate_log[0]} -> {gate_log[1][0]} {gate_log[1][1]}")

    if return_log:
        return best_initial_mapping, best_gate_execution_log
    return best_initial_mapping
