from convert import from_qiskit
from qiskit import QuantumCircuit
from mapping import Mapping
from architecture import DistributedQubitNetworkGraph, COMM_EDGE_WEIGHT
from dag import QuantumDAG
import random
from copy import deepcopy
import networkx as nx
import numpy as np

EXTENDED_LAYER_SIZE = 20
EXTENDED_HEURISTIC_WEIGHT = 0.05
DECAY_VALUE = 0.002
FULL_CORE_PENALTY = 10
TELE_BONUS = -100
CONTRACTED_GRAPH_FREE_NODE_WEIGHT = 1

RESET_TIMER_START = 150
BREAK_DEADLOCK_TIMER_START = 80

class DeadlockError(RuntimeError): pass

def DQC_contracted_graph(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward, traffic = None):
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
                    contracted_graph.add_edge(core[i], core[j], weight = arch.get_separated_core_distance_matrix()[core[i]][core[j]])
    if is_forward:
        # If start/end is comm, increment all adj edge weights (twice for comm edges)
        if node1 in arch.comm_qubits:
            for u,v in contracted_graph.edges(node1):
                if arch.get_p_qubit_core(u) != arch.get_p_qubit_core(v):
                    contracted_graph.edges[u, v]['weight'] += 2
                else:
                    contracted_graph.edges[u, v]['weight'] += 1
        if node2 in arch.comm_qubits:
            for u,v in contracted_graph.edges(node2):
                if arch.get_p_qubit_core(u) != arch.get_p_qubit_core(v):
                    contracted_graph.edges[u, v]['weight'] += 2
                else:
                    contracted_graph.edges[u, v]['weight'] += 1
    else:
        if node1 in arch.comm_qubits and node2 in arch.comm_qubits:
            for u,v in contracted_graph.edges(node1):
                if arch.get_p_qubit_core(u) != arch.get_p_qubit_core(v):
                    contracted_graph.edges[u, v]['weight'] += 2
                else:
                    contracted_graph.edges[u, v]['weight'] += 1
            for u,v in contracted_graph.edges(node2):
                if arch.get_p_qubit_core(u) != arch.get_p_qubit_core(v):
                    contracted_graph.edges[u, v]['weight'] += 2
                else:
                    contracted_graph.edges[u, v]['weight'] += 1


    # Add free node and full core scores
    free_nodes = temp_mapping.get_free_p_nodes()
    if len(free_nodes) == 0:
        raise ValueError("No available free qubits for inter-core routing")
    core_free_nodes_map = [[node for node in core_group if node in free_nodes] for core_group in arch.core_node_groups]

    full_cores = arch.get_full_cores(temp_mapping, 3)

    if len(core_free_nodes_map[arch.get_p_qubit_core(node1)]) == 0:
        raise DeadlockError(f"Starting core {arch.get_p_qubit_core(node1)} has 0 free nodes, free_nodes={free_nodes}")
    if len(core_free_nodes_map[arch.get_p_qubit_core(node2)]) == 0:
        raise DeadlockError(f"Target core {arch.get_p_qubit_core(node2)} has 0 free nodes, free_nodes={free_nodes}")
    
    nearest_free_qubit_map = arch.get_nth_nearest_free_qubit_map(temp_mapping, 0)
    second_nearest_free_qubit_map = arch.get_nth_nearest_free_qubit_map(temp_mapping, 1)

    # Add free node penalties
    if is_forward:
        for comm_node in arch.comm_qubits:
            if comm_node not in nearest_free_qubit_map.keys():
                contracted_graph.remove_node(comm_node)
                continue

            nearest_free_node = nearest_free_qubit_map[comm_node]
            free_node_score = arch.get_separated_core_distance_matrix()[comm_node][nearest_free_node] * CONTRACTED_GRAPH_FREE_NODE_WEIGHT
            for edge in contracted_graph.edges(comm_node):
                contracted_graph.edges[edge]['weight'] += free_node_score/2
    else:
        for comm_edge in arch.comm_edges:
            u, v = comm_edge
            
            nearest_free_node_u = nearest_free_qubit_map[u]
            nearest_free_node_v = nearest_free_qubit_map[v]

            teleport_free_score_1 = np.inf
            teleport_free_score_2 = np.inf

            telegate_free_score = arch.get_separated_core_distance_matrix()[u][nearest_free_node_u] + arch.get_separated_core_distance_matrix()[v][nearest_free_node_v]
            
            if u in second_nearest_free_qubit_map.keys():
                second_nearest_free_node_u = second_nearest_free_qubit_map[u]
                teleport_free_score_1 = arch.get_separated_core_distance_matrix()[u][nearest_free_node_u] + arch.get_separated_core_distance_matrix()[u][second_nearest_free_node_u]
            if v in second_nearest_free_qubit_map.keys():
                second_nearest_free_node_v = second_nearest_free_qubit_map[v]
                teleport_free_score_2 = arch.get_separated_core_distance_matrix()[v][nearest_free_node_v] + arch.get_separated_core_distance_matrix()[v][second_nearest_free_node_v]
            
            free_node_score = min(telegate_free_score, teleport_free_score_1, teleport_free_score_2) * CONTRACTED_GRAPH_FREE_NODE_WEIGHT
            contracted_graph.edges[comm_edge]['weight'] += free_node_score

    # Add full core penalties
    for comm_node in arch.comm_qubits:
        core = arch.get_p_qubit_core(comm_node)
        core_score = FULL_CORE_PENALTY if full_cores[core] else 0
        for u,v in contracted_graph.edges(comm_node):
            if u in arch.comm_qubits and v in arch.comm_qubits:
                contracted_graph.edges[u, v]['weight'] += core_score
    
    if traffic is not None:
        for edge, weight in traffic.items():
            contracted_graph.edges[edge]['weight'] += weight

    return contracted_graph

def DQC_gate_routing_energy(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2, is_forward)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.get_p_qubit_core(temp_mapping.l_to_p(q1))} to core {arch.get_p_qubit_core(temp_mapping.l_to_p(q2))}")
    shortest_path_length = nx.shortest_path_length(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")

    return shortest_path_length

def DQC_gate_routing_path(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2, is_forward)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.get_p_qubit_core(temp_mapping.l_to_p(q1))} to core {arch.get_p_qubit_core(temp_mapping.l_to_p(q2))}")
    shortest_path = nx.shortest_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")

    return shortest_path

def DQC_gate_routing_path_and_energy(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward, traffic = None):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2, is_forward, traffic = traffic)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.get_p_qubit_core(temp_mapping.l_to_p(q1))} to core {arch.get_p_qubit_core(temp_mapping.l_to_p(q2))}")
    shortest_path = nx.shortest_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")
    shortest_path_length = sum(contracted_graph.edges[edge]['weight'] for edge in zip(shortest_path[:-1], shortest_path[1:]))

    return shortest_path, shortest_path_length

def get_traversed_comm_nodes(arch: DistributedQubitNetworkGraph, gate_paths):
    traversed_comm_nodes = set()
    for path in gate_paths:
        for node in path:
            if node in arch.comm_qubits:
                traversed_comm_nodes.add(node)
    return traversed_comm_nodes

def get_SWAP_candidates(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer_gates, gate_paths, is_forward):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    #                         get all adjacent SWAPs of free qubits nearest comm qubits requiring teleportation    
    swappable_nodes = set()
    free_qubits = mapping.get_free_p_nodes()
    free_qubit_map = arch.get_nth_nearest_free_qubit_map(mapping, 0)
    second_free_qubit_map = arch.get_nth_nearest_free_qubit_map(mapping, 1)
    traversed_comm_nodes = get_traversed_comm_nodes(arch, gate_paths)
    for gate in front_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            swappable_nodes.add(mapping.l_to_p(q1))
            swappable_nodes.add(mapping.l_to_p(q2))
    for comm_node in traversed_comm_nodes:
        swappable_nodes.add(free_qubit_map[comm_node])
        if not is_forward and comm_node in second_free_qubit_map.keys():
            swappable_nodes.add(second_free_qubit_map[comm_node])
    edges = arch.separated_core_graph.edges(swappable_nodes)
    return [edge for edge in edges if not (edge[0] in free_qubits and edge[1] in free_qubits)]

def get_teleport_candidates(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer_gates, gate_paths, is_forward):
    # Current implementation: Add valid 4-length paths as a telegate
    #                         Add 3+ length path if first or last 3 qubits make a valid teleportation [or reversed]
    #                         To prevent deadlocks, add teleports targeting a full core in reverse
    teleportations = set()
    deadlock_teleportations = set()
    free_p_nodes = mapping.get_free_p_nodes()
    core_capacities = arch.get_core_capacities(mapping)
    for path in gate_paths:
        if len(path) == 4:
            # get telegates
            if  (path[1] in arch.comm_qubits and path[1] in free_p_nodes and 
                    path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                    arch.separated_core_graph.has_edge(path[0],path[1]) and
                    arch.separated_core_graph.has_edge(path[2],path[3])):
                teleportations.add(tuple(path))
        if len(path) >= 3:
            # get teleports
            if is_forward:
                if  (path[1] in arch.comm_qubits and path[1] in free_p_nodes and 
                        path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                        arch.separated_core_graph.has_edge(path[0],path[1]) and
                        not arch.separated_core_graph.has_edge(path[1],path[2]) and
                        core_capacities[arch.get_p_qubit_core(path[2])] > 1):
                    teleportations.add(tuple([path[0],path[1],path[2]]))

                if  (path[-2] in arch.comm_qubits and path[-2] in free_p_nodes and 
                        path[-3] in arch.comm_qubits and path[-3] in free_p_nodes and
                        arch.separated_core_graph.has_edge(path[-1],path[-2]) and
                        not arch.separated_core_graph.has_edge(path[-2],path[-3]) and
                        core_capacities[arch.get_p_qubit_core(path[-3])] > 1):
                    teleportations.add(tuple([path[-1],path[-2],path[-3]]))

            else:
                if  (path[0] in arch.comm_qubits and path[1] in free_p_nodes and 
                        path[1] in arch.comm_qubits):
                    for target_qubit in arch.separated_core_graph.neighbors(path[1]):
                        if target_qubit in free_p_nodes:
                            teleportations.add(tuple([path[0],path[1],target_qubit]))

                if  (path[-1] in arch.comm_qubits and path[-2] in free_p_nodes and 
                        path[-2] in arch.comm_qubits):
                    for target_qubit in arch.separated_core_graph.neighbors(path[-2]):
                        if target_qubit in free_p_nodes:
                            teleportations.add(tuple([path[-1],path[-2],target_qubit]))

    if is_forward:
        for comm_edge in arch.comm_edges:
            q1, q2 = comm_edge
            if  (q1 in free_p_nodes and q2 in free_p_nodes and
                    core_capacities[arch.get_p_qubit_core(q1)] > 2 and
                    core_capacities[arch.get_p_qubit_core(q2)] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q2):
                    if target_qubit not in free_p_nodes:
                        deadlock_teleportations.add(tuple([target_qubit,q2,q1]))

            if  (q2 in free_p_nodes and q1 in free_p_nodes and
                    core_capacities[arch.get_p_qubit_core(q2)] > 2 and
                    core_capacities[arch.get_p_qubit_core(q1)] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q1):
                    if target_qubit not in free_p_nodes:
                        deadlock_teleportations.add(tuple([target_qubit,q1,q2]))
    else:
        for comm_edge in arch.comm_edges:
            q1, q2 = comm_edge
            if  (q1 not in free_p_nodes and q2 in free_p_nodes and
                    core_capacities[arch.get_p_qubit_core(q2)] > 2 and
                    core_capacities[arch.get_p_qubit_core(q1)] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q2):
                    if target_qubit in free_p_nodes:
                        deadlock_teleportations.add(tuple([target_qubit,q2,q1]))

            if  (q2 not in free_p_nodes and q1 in free_p_nodes and
                    core_capacities[arch.get_p_qubit_core(q1)] > 2 and
                    core_capacities[arch.get_p_qubit_core(q2)] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q1):
                    if target_qubit in free_p_nodes:
                        deadlock_teleportations.add(tuple([target_qubit,q1,q2]))

    return list(teleportations), list(deadlock_teleportations)

def mapping_energy(arch: DistributedQubitNetworkGraph, circuit_dag: QuantumDAG, mapping: Mapping, operation_candidate, decay_array, is_forward, break_deadlock):
    temp_mapping = mapping.copy()
    update_mapping_operation(temp_mapping,operation_candidate,arch)

    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())
    front_size = max(1, sum((len(gate.qubits) == 2) for gate in front_layer_gates))

    traffic = dict()

    H_basic = 0

    for gate in front_layer_gates:
        
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.get_p_qubit_core(temp_mapping.l_to_p(q1)) == arch.get_p_qubit_core(temp_mapping.l_to_p(q2)):
                gate_energy = arch.get_separated_core_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
            else:
                path, gate_energy = DQC_gate_routing_path_and_energy(arch, temp_mapping, q1, q2, is_forward, traffic=traffic)
                for i in range(len(path)-1):
                    if path[i] in arch.comm_qubits and path[i+1] in arch.comm_qubits:
                        comm_edge_A = (path[i], path[i+1])
                        comm_edge_B = (path[i+1], path[i])
                        if comm_edge_A in traffic:
                            traffic[comm_edge_A] += 1
                        elif comm_edge_B in traffic:
                            traffic[comm_edge_B] += 1
                        else:
                            traffic[comm_edge_A] = 1
            H_basic += gate_energy
            if break_deadlock:
                break
        

    if len(operation_candidate) == 4:
        decay_factor = 1
    else:
        decay_factor = max(decay_array[operation_candidate[0]],decay_array[operation_candidate[-1]])

    extended_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_extended_layer(EXTENDED_LAYER_SIZE))
    extended_size = sum((len(gate.qubits) == 2) for gate in extended_layer_gates)

    if extended_size == 0 or break_deadlock:
        H = H_basic / front_size * decay_factor
        return H

    traffic = dict()

    H_extended = 0
    for gate in extended_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.get_p_qubit_core(temp_mapping.l_to_p(q1)) == arch.get_p_qubit_core(temp_mapping.l_to_p(q2)):
                gate_energy = arch.get_separated_core_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
            else:
                path, gate_energy = DQC_gate_routing_path_and_energy(arch, temp_mapping, q1, q2, is_forward, traffic=traffic)
                for i in range(len(path)-1):
                    if path[i] in arch.comm_qubits and path[i+1] in arch.comm_qubits:
                        comm_edge_A = (path[i], path[i+1])
                        comm_edge_B = (path[i+1], path[i])
                        if comm_edge_A in traffic:
                            traffic[comm_edge_A] += 1
                        elif comm_edge_B in traffic:
                            traffic[comm_edge_B] += 1
                        else:
                            traffic[comm_edge_A] = 1
            H_extended += gate_energy
        
    H = decay_factor * (H_basic / front_size + EXTENDED_HEURISTIC_WEIGHT * H_extended / extended_size)
    return H

def get_gate_paths(arch: DistributedQubitNetworkGraph, mapping: Mapping, gates, is_forward):
    paths = []
    for gate in gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.get_p_qubit_core(mapping.l_to_p(q1)) != arch.get_p_qubit_core(mapping.l_to_p(q2)):
                paths.append(DQC_gate_routing_path(arch, mapping, q1, q2, is_forward))
    return paths

def update_mapping_operation(mapping: Mapping, operation, arch: DistributedQubitNetworkGraph):
    if len(operation) == 2:        
        p_q1, p_q2 = operation
        mapping.swap_p_qubits(p_q1, p_q2)
    if len(operation) == 3:        
        p_qstart, p_qcomm1, p_qcomm2 = operation
        mapping.swap_p_qubits(p_qstart, p_qcomm2)
    if len(operation) == 4:        
        p_qstart, p_qcomm1, p_qcomm2, p_target = operation
        arch.register_active_telegate_qubits(p_qstart, p_target)

def initialise_mapping(arch: DistributedQubitNetworkGraph, num_logical_qubits, two_qubit_circuit_dag: QuantumDAG):
    l_qubit_to_core = np.full(len(arch), -1)
    capacities = [len(core_group) for core_group in arch.core_node_groups]

    for node in two_qubit_circuit_dag.get_front_layer():
        gate = two_qubit_circuit_dag.get_gate_from_node(node)
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if l_qubit_to_core[q1] == -1 and l_qubit_to_core[q2] == -1:
                for core_idx in np.random.permutation(arch.num_cores):
                    if capacities[core_idx] > 4:
                        l_qubit_to_core[q1] = core_idx
                        l_qubit_to_core[q2] = core_idx
                        capacities[core_idx] -= 2
                        break
    for l_qubit_idx in np.random.permutation(num_logical_qubits):
        if l_qubit_to_core[l_qubit_idx] == -1:
            most_free_core = capacities.index(max(capacities))
            if capacities[most_free_core] > 0:
                l_qubit_to_core[l_qubit_idx] = most_free_core
                capacities[most_free_core] -= 1
            else:
                raise ValueError("Cannot create layout circuit qubits to perform algorithm on hardware network")
                
    core_to_l_qubit = [[] for _ in range(arch.num_cores)] 
    for core_idx in range(arch.num_cores):
        for l_qubit_idx in range(len(arch)):
            if l_qubit_to_core[l_qubit_idx] == core_idx:
                core_to_l_qubit[core_idx].append(l_qubit_idx)
    
    permutation = np.random.permutation(len(arch))
    free_p_qubits = -1
    p_to_l_map = [0] * len(arch)
    for p in permutation:
        core = arch.get_p_qubit_core(p)
        if len(core_to_l_qubit[core]) > 0:
            p_to_l_map[p] = core_to_l_qubit[core][-1]
            core_to_l_qubit[core].remove(core_to_l_qubit[core][-1])
        else:
            p_to_l_map[p] = free_p_qubits
            free_p_qubits -= 1

    return Mapping([(i,j) for j,i in enumerate(p_to_l_map)])

def sabre_pass(arch: DistributedQubitNetworkGraph, initial_mapping: Mapping, circuit_dag: QuantumDAG, is_forward):
    circuit_dag = deepcopy(circuit_dag)
    initial_mapping = initial_mapping.copy()

    mapping = initial_mapping
    gate_execution_log = []
    decay_array = [1 for _ in range(len(arch))]
    decay_timer = [0 for _ in range(len(arch))]

    front_layer = circuit_dag.get_front_layer()
    reset_timer = RESET_TIMER_START
    break_deadlock_timer = BREAK_DEADLOCK_TIMER_START
    break_deadlock = False

    previous_mapping = deepcopy(mapping)
    previous_gate_execution_log = []

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
            break_deadlock_timer = BREAK_DEADLOCK_TIMER_START
            break_deadlock = False
            previous_mapping = deepcopy(mapping)
            previous_gate_execution_log = deepcopy(gate_execution_log)

        else:
            # A SWAP is required for the next gate
            score = []
            front_layer_gates = circuit_dag.get_gates_from_nodes(front_layer)

            gate_paths = get_gate_paths(arch, mapping, front_layer_gates, is_forward)

            SWAP_candidates = get_SWAP_candidates(arch, mapping, front_layer_gates, gate_paths, is_forward)
            teleport_candidates, deadlock_teleport_candidates = get_teleport_candidates(arch, mapping, front_layer_gates, gate_paths, is_forward)

            operation_candidates = SWAP_candidates + teleport_candidates
            for operation_idx in range(len(operation_candidates)):
                operation_candidate = operation_candidates[operation_idx]
                bonus = TELE_BONUS if len(operation_candidate) >= 3 else 0
                score.append(mapping_energy(arch, circuit_dag, mapping, operation_candidate, decay_array, is_forward, break_deadlock) + bonus)
            for operation_idx in range(len(deadlock_teleport_candidates)):
                operation_candidate = deadlock_teleport_candidates[operation_idx]
                score.append(mapping_energy(arch, circuit_dag, mapping, operation_candidate, decay_array, is_forward, break_deadlock))
            operation_candidates.extend(deadlock_teleport_candidates)

            best_operation_indices = np.where(np.isclose(score, np.min(score)))[0]
            best_operation_idx = np.random.choice(best_operation_indices)
            best_operation = operation_candidates[best_operation_idx]

            arch.clear_active_telegate_qubits()

            update_mapping_operation(mapping,best_operation,arch)
            if len(best_operation) == 2:
                gate_execution_log.append(("SWAP", best_operation))
            elif len(best_operation) == 3:
                gate_execution_log.append(("Teleport", best_operation))
            elif len(best_operation) == 4:
                gate_execution_log.append(("Telegate", best_operation))    

            for i in best_operation:
                decay_array[i] += DECAY_VALUE

        for i in range(len(arch)):
            if decay_array[i] > 1:
                decay_timer[i]+=1
            if decay_timer[i] > 5:
                decay_array[i] = 1
                decay_timer[i] = 0

        front_layer = circuit_dag.get_front_layer()
        break_deadlock_timer -= 1
        if break_deadlock_timer < 0 and not break_deadlock:
            break_deadlock = True
            mapping = previous_mapping
            gate_execution_log = previous_gate_execution_log
        if break_deadlock:
            reset_timer -= 1
            if reset_timer < 0:
                raise DeadlockError("reset_timer expired in sabre_pass")

    return mapping, gate_execution_log


def telesabre_layout(arch: DistributedQubitNetworkGraph, quantum_circuit, verbose = False, return_log = False, seed = None, num_iterations = 10, back_opt=True):
    """
    return values:
        mapping: Mapping of logical qubits to physical qubits
    """
    num_physical_qubits = 0
    num_logical_qubits = 0
    circuit_dag = None
    reverse_circuit_dag = None
    two_qubit_circuit_dag = None

    if seed is not None:
        random.seed(int(seed))

    if isinstance(quantum_circuit, QuantumCircuit):
        num_logical_qubits = quantum_circuit.num_qubits
        num_physical_qubits = len(arch)

        reverse_quantum_circuit = quantum_circuit.inverse()
        circuit_dag = from_qiskit(quantum_circuit)
        reverse_circuit_dag = from_qiskit(reverse_quantum_circuit)
        two_qubit_quantum_circuit = quantum_circuit.copy_empty_like()
        for gate in quantum_circuit.data:
            if len(gate.qubits) == 2:
                two_qubit_quantum_circuit.append(gate.operation, gate.qubits)
        two_qubit_circuit_dag = from_qiskit(two_qubit_quantum_circuit)

        if num_logical_qubits > num_physical_qubits:
            raise ValueError("Too many circuit qubits to perform algorithm on hardware network")
    else:
        raise ValueError("SABRE Layout only accepts Qiskit QuantumCircuit")
    
    if verbose:
        num_free = num_physical_qubits - num_logical_qubits
        num_cores = len(arch.core_node_groups)
        print("num_free", num_free, "num_cores", num_cores)
        if num_free < 2:
            print("Architecture configuration is incompatible with inter-core communication")
        elif num_free * 2 < num_cores:
            print("Architecture configuration is suboptimal for circuit routing")
        elif num_free * 4 < num_cores:
            print("Architecture configuration is standard for circuit routing")
        else:
            print("Architecture configuration is flexible for circuit routing")

    gate_execution_log_iterations = dict()
    deadlocks = 0

    for iteration in range(num_iterations):

        try:
            initial_mapping = initialise_mapping(arch, num_logical_qubits, two_qubit_circuit_dag)
            
            free_nodes = initial_mapping.get_free_p_nodes()     
            num_cores = len(arch.core_node_groups)

            free_per_core = [0] * num_cores
            for p in free_nodes:
                free_per_core[arch.get_p_qubit_core(p)] += 1

            if 0 in free_per_core:
                continue

            final_mapping, _ = sabre_pass(arch, initial_mapping, circuit_dag, True)
            initial_mapping, _ = sabre_pass(arch, final_mapping, reverse_circuit_dag, True)
            _, gate_execution_log = sabre_pass(arch, initial_mapping, circuit_dag, True)

            gate_execution_log_iterations[iteration] = (initial_mapping,gate_execution_log)
            print(f"Iteration {iteration} ran successfully")

        except DeadlockError as e:
            deadlocks += 1
            print("deadlock", iteration, str(e))
            continue
    
    if not gate_execution_log_iterations:
        raise RuntimeError(f"No successful iterations, deadlocks={deadlocks}/{num_iterations}")

    if len(gate_execution_log_iterations)==0:
        raise DeadlockError("No valid runs found")
    
    best_iteration = min(gate_execution_log_iterations, key=lambda k: len(gate_execution_log_iterations[k][1]))
    best_initial_mapping, best_gate_execution_log = gate_execution_log_iterations[best_iteration]

    if verbose:
        best_swap_log = [k for k in best_gate_execution_log if ("SWAP" in k[0])]
        best_teleport_log = [k for k in best_gate_execution_log if (k[0] == "Teleport")]
        best_telegate_log = [k for k in best_gate_execution_log if (k[0] == "Telegate")]
        print("Best Iteration:")
        print(f"#{len(best_gate_execution_log)} total gates")
        print(f"#{len(best_swap_log)} inserted SWAPS")
        print(f"#{len(best_teleport_log)} inserted teleports")
        print(f"#{len(best_telegate_log)} telegates executed")
        print()

        print("Initial Mapping:")
        for i in range(num_logical_qubits):
            print(f"Logical Qubit {i}: Physical Qubit {best_initial_mapping.l_to_p(i)}")
        print()

        # print("Physical Gate Log:")
        # for gate_log in best_gate_execution_log:
        #     print(f"{gate_log[0]} -> {gate_log[1]}")

    if return_log:
        return best_initial_mapping, best_gate_execution_log, len(best_gate_execution_log), len(best_swap_log), len(best_teleport_log), len(best_telegate_log)
    return best_initial_mapping