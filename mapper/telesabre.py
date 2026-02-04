from convert import from_qiskit
from qiskit import QuantumCircuit
from mapping import Mapping
from architecture import DistributedQubitNetworkGraph
from dag import QuantumDAG
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

def DQC_contracted_graph(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2):
    node1 = temp_mapping.l_to_p(q1)
    node2 = temp_mapping.l_to_p(q2)
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
    free_nodes = temp_mapping.get_free_p_nodes()
    if len(free_nodes) == 0:
        raise ValueError("No available free qubits for inter-core routing")
    core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in arch.core_node_groups]

    full_cores = arch.get_full_cores(temp_mapping)

    # Remove for telegate
    if len(core_free_nodes_map[arch.qubit_core_map[node1]]) == 0:
        raise DeadlockError(f"Starting core {arch.qubit_core_map[node1]} has 0 free nodes, free_nodes={free_nodes}")
    if len(core_free_nodes_map[arch.qubit_core_map[node2]]) == 0:
        raise DeadlockError(f"Target core {arch.qubit_core_map[node2]} has 0 free nodes, free_nodes={free_nodes}")

    for comm_node in arch.comm_qubits:
        core = arch.qubit_core_map[comm_node]

        if len(core_free_nodes_map[core]) == 0:
            contracted_graph.remove_node(comm_node)
            continue

        free_node_distances = []
        for free_node in core_free_nodes_map[core]:
            free_node_distances.append(arch.get_distance_matrix()[comm_node][free_node])

        free_node_score = min(free_node_distances)/2
        core_score = FULL_CORE_PENALTY if full_cores[core] else 0
        for u,v in contracted_graph.edges(comm_node):
            contracted_graph.edges[u, v]['weight'] = contracted_graph.edges[u, v]['weight'] + free_node_score/2 + core_score/2

    return contracted_graph

def DQC_gate_routing_energy(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.qubit_core_map[temp_mapping.l_to_p(q1)]} to core {arch.qubit_core_map[temp_mapping.l_to_p(q2)]}")
    shortest_path_length = nx.shortest_path_length(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")

    return shortest_path_length

def DQC_gate_routing_path(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.qubit_core_map[temp_mapping.l_to_p(q1)]} to core {arch.qubit_core_map[temp_mapping.l_to_p(q2)]}")
    shortest_path = nx.shortest_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")

    return shortest_path

def get_traversed_comm_nodes(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer):
    traversed_comm_nodes = set()
    for gate in front_layer:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.qubit_core_map[mapping.l_to_p(q1)] != arch.qubit_core_map[mapping.l_to_p(q2)]:
                path = DQC_gate_routing_path(arch, mapping, q1, q2)
                for node in path:
                    if node in arch.comm_qubits:
                        traversed_comm_nodes.add(node)
    return traversed_comm_nodes

def get_SWAP_candidates(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    #                         get all adjacent SWAPs of free qubits nearest comm qubits requiring teleportation
    swappable_nodes = set()
    free_qubit_map = arch.get_nearest_free_qubit_map(mapping)
    traversed_comm_nodes = get_traversed_comm_nodes(arch, mapping, front_layer)
    for gate in front_layer:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            swappable_nodes.add(mapping.l_to_p(q1))
            swappable_nodes.add(mapping.l_to_p(q2))
    for comm_node in traversed_comm_nodes:
        swappable_nodes.add(free_qubit_map[comm_node])
    edges = arch.separated_core_graph.edges(swappable_nodes)
    return [[edge, None] for edge in edges]

def get_teleport_candidates(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer, is_forward):
    # Old implementation: get all path routes of inter-core front layer qubits
    # Current implementation: Add valid 4-length paths as a telegate
    #                         Add 3+ length path if 1st 3 qubits make a valid teleportation [or reversed]
    #                         To prevent deadlocks, add teleports targeting a full core in reverse
    teleportations = []
    free_p_nodes = mapping.get_free_p_nodes()
    core_capacity = arch.get_core_capacity(mapping)
    for gate in front_layer:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.qubit_core_map[mapping.l_to_p(q1)] != arch.qubit_core_map[mapping.l_to_p(q2)]:
                path = DQC_gate_routing_path(arch, mapping, q1, q2)
                if len(path) == 4:
                    # get telegates
                    if  (path[1] in arch.comm_qubits and path[1] in free_p_nodes and 
                         path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                         arch.separated_core_graph.has_edge(path[0],path[1]) and
                         arch.separated_core_graph.has_edge(path[2],path[3])):
                        teleportations.append([list(path),gate])
                if len(path) > 2:
                    # get teleports
                    # if is_forward:
                        if  (path[1] in arch.comm_qubits and path[1] in free_p_nodes and 
                             path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                             arch.separated_core_graph.has_edge(path[0],path[1]) and
                             core_capacity[arch.qubit_core_map[path[2]]] > 1):
                            teleportations.append([[path[0],path[1],path[2]],gate])

                        if  (path[-2] in arch.comm_qubits and path[-2] in free_p_nodes and 
                             path[-3] in arch.comm_qubits and path[-3] in free_p_nodes and
                             arch.separated_core_graph.has_edge(path[-1],path[-2]) and
                             core_capacity[arch.qubit_core_map[path[-3]]] > 1):
                            teleportations.append([[path[-1],path[-2],path[-3]],gate])
                        
                        if  (path[1] in arch.comm_qubits and path[1] in free_p_nodes and 
                             path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                             core_capacity[arch.qubit_core_map[path[1]]] > 1 and
                             core_capacity[arch.qubit_core_map[path[2]]] <= 1):
                            for target_qubit in arch.separated_core_graph.edges(path[2]):
                                if target_qubit not in free_p_nodes:
                                    teleportations.append([[path[1],path[2],target_qubit],None])

                        if  (path[-2] in arch.comm_qubits and path[-2] in free_p_nodes and 
                             path[-3] in arch.comm_qubits and path[-3] in free_p_nodes and
                             core_capacity[arch.qubit_core_map[path[-2]]] > 1 and
                             core_capacity[arch.qubit_core_map[path[-3]]] <= 1):
                            for target_qubit in arch.separated_core_graph.edges(path[-3]):
                                if target_qubit not in free_p_nodes:
                                    teleportations.append([[path[-2],path[-3],target_qubit],None])
    return teleportations

def mapping_energy(arch: DistributedQubitNetworkGraph, circuit_dag: QuantumDAG, temp_mapping: Mapping, operation_candidate, decay_array):
    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())

    H_basic = 0
    for gate in front_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.qubit_core_map[temp_mapping.l_to_p(q1)] == arch.qubit_core_map[temp_mapping.l_to_p(q2)]:
                gate_energy = arch.get_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
            else:
                gate_energy = DQC_gate_routing_energy(arch, temp_mapping, q1, q2)
            H_basic += gate_energy

    decay_factor = 1 + max(decay_array[operation_candidate[0][0]],decay_array[operation_candidate[0][-1]])

    extended_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_extended_layer(EXTENDED_LAYER_SIZE))

    if len(extended_layer_gates) == 0:
        H = decay_factor / len(front_layer_gates) * H_basic
        return H

    H_extended = 0
    for gate in extended_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            H_extended += arch.get_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
        
    H = decay_factor / len(front_layer_gates) * H_basic + EXTENDED_HEURISTIC_WEIGHT / len(extended_layer_gates) * H_extended
    return H

# def swap_sequence_to_target(mapping: Mapping, start, target, arch: DistributedQubitNetworkGraph, early_stop=0):
#     path = nx.shortest_path(arch,start,target)
#     swap_sequence = []
#     for i in range(len(path)-1-early_stop):
#         mapping.swap_p_qubits(path[i], path[i+1])
#         swap_sequence.append([path[i],path[i+1]])
#     return mapping, swap_sequence

def update_mapping_operation(mapping: Mapping, operation):
    qubits = operation[0]
    if len(qubits) == 2:        
        p_q1, p_q2 = qubits
        mapping.swap_p_qubits(p_q1, p_q2)
    if len(qubits) == 3:        
        p_qstart, p_qcomm1, p_qcomm2 = qubits
        mapping.swap_p_qubits(p_qstart, p_qcomm2)
    if len(qubits) == 4:        
        p_qstart, p_qcomm1, p_qcomm2, p_target = qubits
        pass
        # APPLY TELEGATE

def sabre_pass(arch: DistributedQubitNetworkGraph, initial_mapping: Mapping, circuit_dag: QuantumDAG, is_forward):
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
                gate_execution_log.append((gate.gate_type + " " + str(gate.parameters),[mapping.l_to_p(qubit) for qubit in gate.qubits]))
                circuit_dag.remove_gate(gate_node)
                for qubit in gate.qubits:
                    decay_array[mapping.l_to_p(qubit)] = 1
            reset_timer = RESET_TIMER_START
        else:
            # A SWAP is required for the next gate
            score = dict()
            front_layer_gates = circuit_dag.get_gates_from_nodes(front_layer)

            SWAP_candidates = get_SWAP_candidates(arch, mapping, front_layer_gates)
            teleport_candidates = get_teleport_candidates(arch, mapping, front_layer_gates, is_forward)

            operation_candidates = SWAP_candidates + teleport_candidates
            for operation_idx in range(len(operation_candidates)):
                operation_candidate = operation_candidates[operation_idx]
                temp_mapping = mapping.copy()
                update_mapping_operation(temp_mapping,operation_candidate)
                score[operation_idx] = mapping_energy(arch, circuit_dag, temp_mapping, operation_candidate, decay_array)

            best_operation_idx = min(score, key=score.get)
            best_operation = operation_candidates[best_operation_idx]
            update_mapping_operation(mapping,best_operation)
            if len(best_operation[0]) == 2:
                gate_execution_log.append(("SWAP", best_operation[0]))
            elif len(best_operation[0]) == 3:
                gate_execution_log.append(("Teleport", best_operation[0]))
            else:
                gate_execution_log.append(("Telegate", best_operation[0], best_operation[1].gate_type + " " + str(best_operation[1].parameters),[mapping.l_to_p(qubit) for qubit in best_operation[1].qubits]))    

            decay_array[best_operation[0][0]] = 1 + DECAY_VALUE
            decay_array[best_operation[0][-1]] = 1 + DECAY_VALUE
            for i in range(len(arch)):
                if decay_array[i]:
                    decay_timer[i]+=1
                if decay_timer[i] > 5:
                    decay_array[i] = 1
                    decay_timer[i] = 0

        front_layer = circuit_dag.get_front_layer()
        reset_timer -= 1
        if reset_timer < 0:
            raise DeadlockError("reset_timer expired in sabre_pass")

    return mapping, gate_execution_log


def telesabre(arch: DistributedQubitNetworkGraph, quantum_circuit, verbose = False, return_log = False):
    """
    return values:
        mapping: Mapping of logical qubits to physical qubits
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

        reverse_quantum_circuit = quantum_circuit.inverse()
        circuit_dag = from_qiskit(quantum_circuit)
        reverse_circuit_dag = from_qiskit(reverse_quantum_circuit)

        if num_logical_qubits > num_physical_qubits:
            raise ValueError("Too many circuit qubits to perform algorithm on hardware network")
    else:
        raise ValueError("SABRE Layout only accepts Qiskit QuantumCircuit")
    
    num_free = num_physical_qubits - num_logical_qubits
    num_cores = len(arch.core_node_groups)
    print("num_free", num_free, "num_cores", num_cores)

    gate_execution_log_iterations = dict()
    deadlocks = 0

    for iteration in range(NUM_ITERATIONS):

        try:
            random_mapping = list(range(num_logical_qubits-num_physical_qubits,num_logical_qubits))
            random.shuffle(random_mapping)
            initial_mapping = Mapping([(i,j) for j,i in enumerate(random_mapping)])

            free_nodes = initial_mapping.get_free_p_nodes()
            num_cores = len(arch.core_node_groups)

            free_per_core = [0] * num_cores
            for p in free_nodes:
                free_per_core[arch.qubit_core_map[p]] += 1

            if 0 in free_per_core:
                continue

            final_mapping, _ = sabre_pass(arch, initial_mapping, circuit_dag, True)
            initial_mapping, _ = sabre_pass(arch, final_mapping, reverse_circuit_dag, False)
            _, gate_execution_log = sabre_pass(arch, initial_mapping, circuit_dag, True)

            gate_execution_log_iterations[iteration] = (initial_mapping,gate_execution_log)
            print(iteration)

        except DeadlockError as e:
            deadlocks += 1
            print("deadlock", iteration, str(e))
            continue
    
    if not gate_execution_log_iterations:
        raise RuntimeError(f"No successful iterations, deadlocks={deadlocks}/{NUM_ITERATIONS}")

    if len(gate_execution_log_iterations)==0:
        raise DeadlockError("No valid runs found")
    
    best_iteration = min(gate_execution_log_iterations, key=lambda k: len(gate_execution_log_iterations[k][1]))
    best_initial_mapping, best_gate_execution_log = gate_execution_log_iterations[best_iteration]

    if verbose:
        best_swap_log = [k for k in best_gate_execution_log if ("SWAP" in k[0])]
        best_teleport_log = [k for k in best_gate_execution_log if (k[0] == "Teleport")]
        print("Best Iteration:")
        print(f"#{len(best_gate_execution_log)} total gates")
        print(f"#{len(best_swap_log)} inserted SWAPS")
        print(f"#{len(best_teleport_log)} inserted teleports")
        print()

        print("Initial Mapping:")
        for i in range(num_logical_qubits):
            print(f"Logical Qubit {i}: Physical Qubit {best_initial_mapping.l_to_p(i)}")
        print()

        print("Physical Gate Log:")
        for gate_log in best_gate_execution_log:
            print(f"{gate_log[0]} -> {gate_log[1]}")

    if return_log:
        return best_initial_mapping, best_gate_execution_log
    return best_initial_mapping
