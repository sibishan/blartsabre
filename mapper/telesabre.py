from convert import from_qiskit
from qiskit import QuantumCircuit
from mapping import Mapping
from architecture import DistributedQubitNetworkGraph, COMM_EDGE_WEIGHT
from dag import QuantumDAG
import random
from copy import deepcopy
import networkx as nx
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    RemoveDiagonalGatesBeforeMeasure,
    Unroll3qOrMore,
    RemoveResetInZeroState,
    OptimizeSwapBeforeMeasure,
    RemoveFinalMeasurements,
    RemoveBarriers
)

EXTENDED_LAYER_SIZE = 10
EXTENDED_HEURISTIC_WEIGHT = 0.25
DECAY_VALUE = 0.001
FULL_CORE_PENALTY = 10
TELE_BONUS = -5
CONTRACTED_GRAPH_FREE_NODE_WEIGHT = 2

RESET_TIMER_START = 50

NUM_ITERATIONS = 10

class DeadlockError(RuntimeError): pass

def DQC_contracted_graph(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward):
    node1 = temp_mapping.l_to_p(q1)
    node2 = temp_mapping.l_to_p(q2)
    # Add comm nodes, comm edges & target nodes
    contracted_graph = nx.Graph(arch.comm_edges)
    nx.set_edge_attributes(contracted_graph,COMM_EDGE_WEIGHT,'weight')
    contracted_graph.add_node(node1)
    contracted_graph.add_node(node2)

    comm_data_direction_bias = 1 if is_forward else 0
    # Add edges between target and comm qubits within core
    core_node_map = [[node for node in core_group if node in contracted_graph.nodes()] for core_group in arch.core_node_groups]
    for core in core_node_map:
        for i in range(len(core)-1):
            for j in range(i+1,len(core)):
                if core[i] != core[j]:
                    contracted_graph.add_edge(core[i], core[j], weight = abs(arch.get_separated_core_distance_matrix()[core[i]][core[j]] - comm_data_direction_bias))
    if is_forward:
        if node1 in arch.comm_qubits:
            for u,v in contracted_graph.edges(node1):
                if u in arch.comm_qubits and v in arch.comm_qubits:
                    contracted_graph.edges[u, v]['weight'] = contracted_graph.edges[u, v]['weight'] + 1
        if node2 in arch.comm_qubits:
            for u,v in contracted_graph.edges(node2):
                if u in arch.comm_qubits and v in arch.comm_qubits:
                    contracted_graph.edges[u, v]['weight'] = contracted_graph.edges[u, v]['weight'] + 1
                
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
    
    nearest_free_qubit_map = arch.get_nth_nearest_free_qubit_map(temp_mapping, 0)

    for comm_node in arch.comm_qubits:
        core = arch.qubit_core_map[comm_node]

        if comm_node not in nearest_free_qubit_map.keys():
            contracted_graph.remove_node(comm_node)
            continue


        if is_forward:
            nearest_free_node = nearest_free_qubit_map[comm_node]
            free_node_score = arch.get_separated_core_distance_matrix()[comm_node][nearest_free_node] * CONTRACTED_GRAPH_FREE_NODE_WEIGHT
        else:
            nearest_free_node = arch.get_nth_nearest_intercore_free_qubit(temp_mapping, comm_node, 0)
            second_nearest_free_node = arch.get_nth_nearest_intercore_free_qubit(temp_mapping, comm_node, 1)
            free_node_score = (arch.get_distance_matrix()[comm_node][nearest_free_node] + arch.get_distance_matrix()[comm_node][second_nearest_free_node])
        core_score = FULL_CORE_PENALTY if full_cores[core] else 0
        for u,v in contracted_graph.edges(comm_node):
            # print(u,v,free_node_score)
            contracted_graph.edges[u, v]['weight'] = contracted_graph.edges[u, v]['weight'] + free_node_score/2
            if u in arch.comm_qubits and v in arch.comm_qubits:
                contracted_graph.edges[u, v]['weight'] = contracted_graph.edges[u, v]['weight'] + core_score

    return contracted_graph

def DQC_gate_routing_energy(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2, is_forward)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.qubit_core_map[temp_mapping.l_to_p(q1)]} to core {arch.qubit_core_map[temp_mapping.l_to_p(q2)]}")
    shortest_path_length = nx.shortest_path_length(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")

    return shortest_path_length

def DQC_gate_routing_path(arch: DistributedQubitNetworkGraph, temp_mapping: Mapping, q1, q2, is_forward):

    contracted_graph = DQC_contracted_graph(arch, temp_mapping, q1, q2, is_forward)
    if not nx.has_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2)):
        raise DeadlockError(f"Impossible to route from core {arch.qubit_core_map[temp_mapping.l_to_p(q1)]} to core {arch.qubit_core_map[temp_mapping.l_to_p(q2)]}")
    shortest_path = nx.shortest_path(contracted_graph, source=temp_mapping.l_to_p(q1), target=temp_mapping.l_to_p(q2), weight="weight")

    return shortest_path

def get_traversed_comm_nodes(arch: DistributedQubitNetworkGraph, gate_paths):
    traversed_comm_nodes = set()
    for path in gate_paths:
        for node in path:
            if node in arch.comm_qubits:
                traversed_comm_nodes.add(node)
    return traversed_comm_nodes

def get_SWAP_candidates(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer, gate_paths, is_forward):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    #                         get all adjacent SWAPs of free qubits nearest comm qubits requiring teleportation
    swappable_nodes = set()
    free_qubits = mapping.get_free_p_nodes()
    free_qubit_map = arch.get_nth_nearest_free_qubit_map(mapping, 0)
    second_free_qubit_map = arch.get_nth_nearest_free_qubit_map(mapping, 1)
    traversed_comm_nodes = get_traversed_comm_nodes(arch, gate_paths)
    for gate in front_layer:
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

def get_teleport_candidates(arch: DistributedQubitNetworkGraph, mapping: Mapping, front_layer, gate_paths, is_forward):
    # Current implementation: Add valid 4-length paths as a telegate
    #                         Add 3+ length path if first or last 3 qubits make a valid teleportation [or reversed]
    #                         To prevent deadlocks, add teleports targeting a full core in reverse
    teleportations = set()
    free_p_nodes = mapping.get_free_p_nodes()
    core_capacity = arch.get_core_capacity(mapping)
    for path in gate_paths:
        # print("path",path)
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
                        core_capacity[arch.qubit_core_map[path[2]]] > 1):
                    teleportations.add(tuple([path[0],path[1],path[2]]))

                if  (path[-2] in arch.comm_qubits and path[-2] in free_p_nodes and 
                        path[-3] in arch.comm_qubits and path[-3] in free_p_nodes and
                        arch.separated_core_graph.has_edge(path[-1],path[-2]) and
                        core_capacity[arch.qubit_core_map[path[-3]]] > 1):
                    teleportations.add(tuple([path[-1],path[-2],path[-3]]))

                if  (path[1] in arch.comm_qubits and path[1] in free_p_nodes and 
                        path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                        core_capacity[arch.qubit_core_map[path[1]]] > 1 and
                        core_capacity[arch.qubit_core_map[path[2]]] <= 2):
                    for target_qubit in arch.separated_core_graph.neighbors(path[2]):
                        if target_qubit not in free_p_nodes:
                            teleportations.add(tuple([target_qubit,path[2],path[1]]))

                if  (path[-2] in arch.comm_qubits and path[-2] in free_p_nodes and 
                        path[-3] in arch.comm_qubits and path[-3] in free_p_nodes and
                        core_capacity[arch.qubit_core_map[path[-2]]] > 1 and
                        core_capacity[arch.qubit_core_map[path[-3]]] <= 2):
                    for target_qubit in arch.separated_core_graph.neighbors(path[-3]):
                        if target_qubit not in free_p_nodes:
                            teleportations.add(tuple([target_qubit,path[-3],path[-2]]))
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

                if  (path[1] in arch.comm_qubits and path[1] not in free_p_nodes and
                    path[2] in arch.comm_qubits and path[2] in free_p_nodes and
                    core_capacity[arch.qubit_core_map[path[2]]] > 1 and
                    core_capacity[arch.qubit_core_map[path[1]]] <= 2):
                    for target_qubit in arch.separated_core_graph.neighbors(path[2]):
                        if target_qubit in free_p_nodes:
                            teleportations.add(tuple([target_qubit,path[2],path[1]]))

                if  (path[-2] in arch.comm_qubits and  path[-2] not in free_p_nodes and
                    path[-3] in arch.comm_qubits and path[-3] in free_p_nodes and
                    core_capacity[arch.qubit_core_map[path[-3]]] > 1 and
                    core_capacity[arch.qubit_core_map[path[-2]]] <= 2):
                    for target_qubit in arch.separated_core_graph.neighbors(path[-3]):
                        if target_qubit in free_p_nodes:
                            teleportations.add(tuple([target_qubit,path[-3],path[-2]]))

    if is_forward:
        for comm_edge in arch.comm_edges:
            q1, q2 = comm_edge
            if  (q1 in free_p_nodes and q2 in free_p_nodes and
                    core_capacity[arch.qubit_core_map[q1]] > 2 and
                    core_capacity[arch.qubit_core_map[q2]] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q2):
                    if target_qubit not in free_p_nodes:
                        teleportations.add(tuple([target_qubit,q2,q1]))

            if  (q2 in free_p_nodes and q1 in free_p_nodes and
                    core_capacity[arch.qubit_core_map[q2]] > 2 and
                    core_capacity[arch.qubit_core_map[q1]] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q1):
                    if target_qubit not in free_p_nodes:
                        teleportations.add(tuple([target_qubit,q1,q2]))
    else:
        for comm_edge in arch.comm_edges:
            q1, q2 = comm_edge
            if  (q1 not in free_p_nodes and q2 in free_p_nodes and
                    core_capacity[arch.qubit_core_map[q2]] > 2 and
                    core_capacity[arch.qubit_core_map[q1]] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q2):
                    if target_qubit in free_p_nodes:
                        teleportations.add(tuple([target_qubit,q2,q1]))

            if  (q2 not in free_p_nodes and q1 in free_p_nodes and
                    core_capacity[arch.qubit_core_map[q1]] > 2 and
                    core_capacity[arch.qubit_core_map[q2]] <= 2):
                for target_qubit in arch.separated_core_graph.neighbors(q1):
                    if target_qubit in free_p_nodes:
                        teleportations.add(tuple([target_qubit,q1,q2]))

    return list(teleportations)

def mapping_energy(arch: DistributedQubitNetworkGraph, circuit_dag: QuantumDAG, mapping: Mapping, operation_candidate, decay_array, is_forward):
    temp_mapping = mapping.copy()
    update_mapping_operation(temp_mapping,operation_candidate,arch)

    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())

    H_basic = TELE_BONUS if len(operation_candidate) >= 3 else 0
    # print(operation_candidate)
    for gate in front_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.qubit_core_map[temp_mapping.l_to_p(q1)] == arch.qubit_core_map[temp_mapping.l_to_p(q2)]:
                gate_energy = arch.get_separated_core_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)] - 1
            else:
                gate_energy = DQC_gate_routing_energy(arch, temp_mapping, q1, q2, is_forward)
            # print(gate,gate_energy)
            H_basic += gate_energy

    if len(operation_candidate) == 4:
        decay_factor = 1
    else:
        decay_factor = 1 + max(decay_array[operation_candidate[0]],decay_array[operation_candidate[-1]])

    extended_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_extended_layer(EXTENDED_LAYER_SIZE))

    if len(extended_layer_gates) == 0:
        H = decay_factor / len(front_layer_gates) * H_basic
        return H

    H_extended = 0
    for gate in extended_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.qubit_core_map[temp_mapping.l_to_p(q1)] == arch.qubit_core_map[temp_mapping.l_to_p(q2)]:
                gate_energy = arch.get_separated_core_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
            else:
                gate_energy = DQC_gate_routing_energy(arch, temp_mapping, q1, q2, is_forward)
            H_extended += gate_energy
        
    H = decay_factor / len(front_layer_gates) * H_basic + EXTENDED_HEURISTIC_WEIGHT / len(extended_layer_gates) * H_extended
    return H

def get_gate_paths(arch: DistributedQubitNetworkGraph, mapping: Mapping, gates, is_forward):
    paths = []
    for gate in gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            if arch.qubit_core_map[mapping.l_to_p(q1)] != arch.qubit_core_map[mapping.l_to_p(q2)]:
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

            gate_paths = get_gate_paths(arch, mapping, front_layer_gates, is_forward)

            SWAP_candidates = get_SWAP_candidates(arch, mapping, front_layer_gates, gate_paths, is_forward)
            teleport_candidates = get_teleport_candidates(arch, mapping, front_layer_gates, gate_paths, is_forward)

            operation_candidates = SWAP_candidates + teleport_candidates
            for operation_idx in range(len(operation_candidates)):
                operation_candidate = operation_candidates[operation_idx]
                score[operation_idx] = mapping_energy(arch, circuit_dag, mapping, operation_candidate, decay_array, is_forward)

            # if reset_timer < 5:
            #     print([(operation_candidates[i], score[i]) for i in range(len(operation_candidates))])

            #     temp_mapping = mapping.copy()
            #     # update_mapping_operation(temp_mapping,(0,3),arch)
            #     print("ahoy")
            #     print(circuit_dag.get_gate_count())
            #     contracto = DQC_contracted_graph(arch,temp_mapping,*front_layer_gates[0].qubits,is_forward)

            #     pos = nx.spring_layout(contracto, k=1)
            #     nx.draw(contracto, pos)
            #     nx.draw_networkx_labels(contracto, pos)
            #     # nx.draw_networkx_edge_labels(contracto, pos)

            #     arch.draw_mapping(temp_mapping)

            best_operation_idx = min(score, key=score.get)
            best_operation = operation_candidates[best_operation_idx]

            arch.clear_active_telegate_qubits()

            update_mapping_operation(mapping,best_operation,arch)
            if len(best_operation) == 2:
                gate_execution_log.append(("SWAP", best_operation))
            elif len(best_operation) == 3:
                gate_execution_log.append(("Teleport", best_operation))
            elif len(best_operation) == 4:
                gate_execution_log.append(("Telegate", best_operation))    

            decay_array[best_operation[0]] = 1 + DECAY_VALUE
            decay_array[best_operation[-1]] = 1 + DECAY_VALUE
            for i in range(len(arch)):
                if decay_array[i]:
                    decay_timer[i]+=1
                if decay_timer[i] > 5:
                    decay_array[i] = 1
                    decay_timer[i] = 0

            # if reset_timer < 5:
            #     print(best_operation)
            #     print([gate for gate in front_layer_gates])
            #     print([gate for gate in circuit_dag.get_gates_from_nodes(circuit_dag.get_extended_layer())])
            #     print(is_forward)
            #     arch.draw_mapping(mapping)
        
        front_layer = circuit_dag.get_front_layer()
        reset_timer -= 1
        if reset_timer < 0:
            raise DeadlockError("reset_timer expired in sabre_pass")
        # print(circuit_dag.get_gate_count())

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
        pm = PassManager()
        pm.append([
            RemoveBarriers(),
            OptimizeSwapBeforeMeasure(),
            RemoveDiagonalGatesBeforeMeasure(),
            RemoveFinalMeasurements(),
            Unroll3qOrMore(),
            RemoveResetInZeroState(),
        ])
        quantum_circuit = pm.run(quantum_circuit)

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
            print(f"Iteration {iteration} ran successfully")

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

        print("Physical Gate Log:")
        for gate_log in best_gate_execution_log:
            print(f"{gate_log[0]} -> {gate_log[1]}")

    if return_log:
        return best_initial_mapping, best_gate_execution_log
    return best_initial_mapping
