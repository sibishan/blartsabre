from convert import from_qiskit
from qiskit import QuantumCircuit
from mapping import Mapping
import random
from dag import QuantumDAG
from blart_architecture import BLARTNetworkGraph
from copy import deepcopy
from mapper.telesabre import DeadlockError
import numpy as np
import networkx as nx

EXTENDED_LAYER_SIZE = 20
EXTENDED_HEURISTIC_WEIGHT = 0.05
DECAY_VALUE = 0.001

RESET_TIMER_START = 150
BREAK_DEADLOCK_TIMER_START = 80

def get_SWAP_candidates(arch: BLARTNetworkGraph, mapping: Mapping, front_layer):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    front_nodes = set()
    for gate in front_layer:
        for i in gate.qubits:
            front_nodes.add(mapping.l_to_p(i))
    edges = arch.graph.edges(front_nodes)
    return list(edges)


def SWAP_heuristic(arch: BLARTNetworkGraph, circuit_dag: QuantumDAG, mapping: Mapping, SWAP_candidate, decay_array, break_deadlock):
    temp_mapping = mapping.copy()
    update_mapping_SWAP(temp_mapping,SWAP_candidate)

    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())
    front_size = max(1, sum((len(gate.qubits) == 2) for gate in front_layer_gates))

    H_basic = 0

    for gate in front_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            H_basic += arch.get_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
            if break_deadlock:
                break

    decay_factor = max(decay_array[SWAP_candidate[0]], decay_array[SWAP_candidate[-1]])

    extended_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_extended_layer(EXTENDED_LAYER_SIZE))
    extended_size = sum((len(gate.qubits) == 2) for gate in extended_layer_gates)

    if extended_size == 0 or break_deadlock:
        H = H_basic / front_size * decay_factor
        return H

    H_extended = 0
    for gate in extended_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            H_extended += arch.get_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
        

    H = decay_factor * (H_basic / front_size + EXTENDED_HEURISTIC_WEIGHT * H_extended / extended_size)
    return H

def update_mapping_SWAP(mapping: Mapping, SWAP_candidate):
    p_q1, p_q2 = SWAP_candidate
    mapping.swap_p_qubits(p_q1, p_q2)

def initialise_mapping(arch: BLARTNetworkGraph, num_logical_qubits, two_qubit_circuit_dag: QuantumDAG):
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

def blartsabre_pass(arch: BLARTNetworkGraph, initial_mapping: Mapping, circuit_dag: QuantumDAG):
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
        # print(circuit_dag.get_gate_count())
        executable_gate_nodes = []
        for gate_node in front_layer:
            gate = circuit_dag.get_gate_from_node(gate_node)
            if arch.check_gate_executable(gate, mapping):
                executable_gate_nodes.append(gate_node)
        
        if len(executable_gate_nodes) != 0:
            # There are gates ready to be executed as-is
            for gate_node in executable_gate_nodes:
                gate = circuit_dag.get_gate_from_node(gate_node)

                remote_gate_string = ""
                if len(gate.qubits) == 2:
                    q1, q2 = gate.qubits
                    if arch.graph[mapping.l_to_p(q1)][mapping.l_to_p(q2)]["type"] == "blart":
                        remote_gate_string = "Remote Gate "
                    
                gate_execution_log.append((remote_gate_string + gate.gate_type + " " + str(gate.parameters),[mapping.l_to_p(qubit) for qubit in gate.qubits]))
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

            SWAP_candidates = get_SWAP_candidates(arch, mapping, front_layer_gates)

            for SWAP_candidate in SWAP_candidates:
                score.append(SWAP_heuristic(arch, circuit_dag, mapping, SWAP_candidate, decay_array, break_deadlock))

            best_SWAP_indices = np.where(np.isclose(score, np.min(score)))[0]
            best_SWAP_idx = np.random.choice(best_SWAP_indices)
            best_SWAP = SWAP_candidates[best_SWAP_idx]

            update_mapping_SWAP(mapping,best_SWAP)
            if best_SWAP in arch.blart_edges or (best_SWAP[1], best_SWAP[0]) in arch.blart_edges:
                gate_execution_log.append(("Remote SWAP", best_SWAP))
            else:
                gate_execution_log.append(("SWAP", best_SWAP))
            
            for i in best_SWAP:
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


def blartsabre_layout(arch: BLARTNetworkGraph, quantum_circuit, verbose = False, return_log = False, seed = None, num_iterations = 50):
    """
    return values:
        mapping: Bidict mapping where logical qubits are keys, Physical qubits are values
    """
    num_physical_qubits = 0
    num_logical_qubits = 0
    circuit_dag = None
    reverse_circuit_dag = None
    two_qubit_circuit_dag = None

    if seed is not None:
        random.seed(int(seed))

    if isinstance(quantum_circuit, QuantumCircuit):
        reverse_quantum_circuit = quantum_circuit.inverse()

        num_logical_qubits = quantum_circuit.num_qubits
        num_physical_qubits = len(arch)

        if num_logical_qubits > num_physical_qubits:
            raise ValueError("Circuit contains too many qubits to perform algorithm on hardware architecture")

        circuit_dag = from_qiskit(quantum_circuit)
        reverse_circuit_dag = from_qiskit(reverse_quantum_circuit)

        two_qubit_quantum_circuit = quantum_circuit.copy_empty_like()
        for gate in quantum_circuit.data:
            if len(gate.qubits) == 2:
                two_qubit_quantum_circuit.append(gate.operation, gate.qubits)
        two_qubit_circuit_dag = from_qiskit(two_qubit_quantum_circuit)
    else:
        raise ValueError("SABRE Layout only accepts Qiskit QuantumCircuit")
    
    gate_execution_log_iterations = dict()

    for iteration in range(num_iterations):
        # random_mapping = list(range(num_logical_qubits-num_physical_qubits,num_logical_qubits))
        # random.shuffle(random_mapping)
        # initial_mapping = Mapping([(i,j) for j,i in enumerate(random_mapping)])
        initial_mapping = initialise_mapping(arch, num_logical_qubits, two_qubit_circuit_dag)

        final_mapping, _ = blartsabre_pass(arch, initial_mapping, circuit_dag)
        initial_mapping, _ = blartsabre_pass(arch, final_mapping, reverse_circuit_dag)
        _, gate_execution_log = blartsabre_pass(arch, initial_mapping, circuit_dag)

        gate_execution_log_iterations[iteration] = (initial_mapping,gate_execution_log)

        if verbose:
            print(f"Iteration {iteration} ran successfully")

    best_iteration = min(gate_execution_log_iterations, key=lambda k: (len(gate_execution_log_iterations[k][1]), len([i for i in gate_execution_log_iterations[k][1] if ("Remote Gate" in i[0])])))
    best_initial_mapping, best_gate_execution_log = gate_execution_log_iterations[best_iteration]

    if verbose:
        best_swap_log = [k for k in best_gate_execution_log if (k[0] == "SWAP")]
        best_remoteswap_log = [k for k in best_gate_execution_log if (k[0] == "Remote SWAP")]
        best_remotegate_log = [k for k in best_gate_execution_log if ("Remote Gate" in k[0])]
        print("Best Iteration:")
        print(f"#{len(best_gate_execution_log)} total gates")
        print(f"#{len(best_swap_log)} inserted SWAPs")
        print(f"#{len(best_remoteswap_log)} inserted Remote SWAPs")
        print(f"#{len(best_remotegate_log)} Gates executed Remotely")
        print()

        print("Initial Mapping:")
        for i in range(num_logical_qubits):
            print(f"Logical Qubit {i}: Physical Qubit {best_initial_mapping.l_to_p(i)}")
        print()

        # print("Physical Gate Log:")
        # for gate_log in best_gate_execution_log:
        #     print(f"{gate_log[0]} -> {gate_log[1]}")

    if return_log:
        return best_initial_mapping, best_gate_execution_log, len(best_gate_execution_log), len(best_swap_log), len(best_remoteswap_log), len(best_remotegate_log) 
    return best_initial_mapping
