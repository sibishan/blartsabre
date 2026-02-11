from convert import from_qiskit
from qiskit import QuantumCircuit
from mapping import Mapping
import random
from dag import QuantumDAG
from blart_architecture import BLARTNetworkGraph
from copy import deepcopy

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    RemoveDiagonalGatesBeforeMeasure,
    Unroll3qOrMore,
    RemoveResetInZeroState,
    OptimizeSwapBeforeMeasure,
    RemoveFinalMeasurements,
    RemoveBarriers
)

EXTENDED_LAYER_SIZE = 5
EXTENDED_HEURISTIC_WEIGHT = 0.5
DECAY_VALUE = 0.001

def get_SWAP_candidates(arch: BLARTNetworkGraph, mapping: Mapping, front_layer):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    front_nodes = set()
    for gate in front_layer:
        for i in gate.qubits:
            front_nodes.add(mapping.l_to_p(i))
    edges = arch.edges(front_nodes)
    return edges


def SWAP_heuristic(arch: BLARTNetworkGraph, circuit_dag: QuantumDAG, mapping: Mapping, SWAP_candidate, decay_array):
    temp_mapping = mapping.copy()
    update_mapping_SWAP(temp_mapping,SWAP_candidate)

    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())
    H_basic = 0
    for gate in front_layer_gates:
        if len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            H_basic += arch.get_distance_matrix()[temp_mapping.l_to_p(q1)][temp_mapping.l_to_p(q2)]
    
    u, v = SWAP_candidate
    decay_factor = 1 + max(decay_array[u], decay_array[v])

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

def update_mapping_SWAP(mapping: Mapping, SWAP_candidate):
    p_q1, p_q2 = SWAP_candidate
    mapping.swap_p_qubits(p_q1, p_q2)

def sabre_forward_pass(arch: BLARTNetworkGraph, initial_mapping: Mapping, circuit_dag: QuantumDAG):
    circuit_dag = deepcopy(circuit_dag)
    initial_mapping = initial_mapping.copy()

    mapping = initial_mapping
    gate_execution_log = []
    decay_array = [1 for _ in range(len(arch))]
    decay_timer = [0 for _ in range(len(arch))]

    front_layer = circuit_dag.get_front_layer()
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

                remote_gate_string = ""
                if len(gate.qubits) == 2:
                    q1, q2 = gate.qubits
                    if arch[mapping.l_to_p(q1)][mapping.l_to_p(q2)]["type"] == "blart":
                        remote_gate_string = "Remote Gate "
                    
                gate_execution_log.append((remote_gate_string + gate.gate_type + " " + str(gate.parameters),[mapping.l_to_p(qubit) for qubit in gate.qubits]))
                circuit_dag.remove_gate(gate_node)
                for qubit in gate.qubits:
                    decay_array[mapping.l_to_p(qubit)] = 1
        else:
            # A SWAP is required for the next gate
            score = dict()
            front_layer_gates = circuit_dag.get_gates_from_nodes(front_layer)

            SWAP_candidates = get_SWAP_candidates(arch, mapping, front_layer_gates)

            for SWAP_candidate in SWAP_candidates:
                score[SWAP_candidate] = SWAP_heuristic(arch, circuit_dag, mapping, SWAP_candidate, decay_array)

            best_SWAP = min(score, key=score.get)

            update_mapping_SWAP(mapping,best_SWAP)
            if best_SWAP in arch.blart_edges:
                gate_execution_log.append(("Remote SWAP", best_SWAP))
            else:
                gate_execution_log.append(("SWAP", best_SWAP))
            decay_array[best_SWAP[0]] = 1 + DECAY_VALUE
            decay_array[best_SWAP[1]] = 1 + DECAY_VALUE
            for i in range(len(arch)):
                if decay_array[i]:
                    decay_timer[i]+=1
                if decay_timer[i] > 5:
                    decay_array[i] = 1
                    decay_timer[i] = 0

        front_layer = circuit_dag.get_front_layer()

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
    else:
        raise ValueError("SABRE Layout only accepts Qiskit QuantumCircuit")
    
    gate_execution_log_iterations = dict()

    for iteration in range(num_iterations):
        
        random_mapping = list(range(num_logical_qubits-num_physical_qubits,num_logical_qubits))
        random.shuffle(random_mapping)
        initial_mapping = Mapping([(i,j) for j,i in enumerate(random_mapping)])

        final_mapping, _ = sabre_forward_pass(arch, initial_mapping, circuit_dag)
        initial_mapping, _ = sabre_forward_pass(arch, final_mapping, reverse_circuit_dag)
        _, gate_execution_log = sabre_forward_pass(arch, initial_mapping, circuit_dag)

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

        print("Physical Gate Log:")
        for gate_log in best_gate_execution_log:
            print(f"{gate_log[0]} -> {gate_log[1]}")

    if return_log:
        return best_initial_mapping, best_gate_execution_log
    return best_initial_mapping
