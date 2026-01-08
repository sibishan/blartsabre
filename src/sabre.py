from .convert import from_qiskit, from_qasm_file
from qiskit import QuantumCircuit
from bidict import bidict
import random
from copy import deepcopy

EXTENDED_LAYER_SIZE = 10
EXTENDED_HEURISTIC_WEIGHT = 0.5
DECAY_VALUE = 0.001

def reduce_2_qubit_gates(circuit):
    for i in range(len(circuit.data)-1,-1,-1):
        if circuit.data[i].operation.num_qubits != 2:
            del circuit.data[i]
    return circuit

def get_SWAP_candidates(graph, mapping, front_layer):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    front_nodes = set()
    for gate in front_layer:
        for i in gate.qubits:
            front_nodes.add(mapping[i])
    edges = graph.edges(front_nodes)
    return edges


def SWAP_heuristic(circuit_dag, temp_mapping, dist_matrix, SWAP_candidate, decay_array):
    front_layer_gates = circuit_dag.get_gates_from_nodes(circuit_dag.get_front_layer())
    H_basic = 0
    for gate in front_layer_gates:
        q1, q2 = gate.qubits
        H_basic += dist_matrix[temp_mapping[q1]][temp_mapping[q2]]

    
    decay_factor = 1 + max(decay_array[temp_mapping[SWAP_candidate[0]]],decay_array[temp_mapping[SWAP_candidate[1]]])

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

def update_mapping(mapping, p_q1, p_q2):
    temp1 = mapping.inv[p_q1]
    temp2 = mapping.inv[p_q2]
    mapping.inv[p_q1] = None
    mapping.inv[p_q2] = temp1
    mapping.inv[p_q1] = temp2
    return mapping

def sabre_forward_pass(qubit_graph, dist_matrix, initial_mapping, circuit_dag):
    circuit_dag = deepcopy(circuit_dag)
    initial_mapping = initial_mapping.copy()

    mapping = initial_mapping
    gate_execution_log = []
    decay_array = [1 for _ in range(len(qubit_graph))]
    decay_timer = [0 for _ in range(len(qubit_graph))]

    front_layer = circuit_dag.get_front_layer()
    while len(front_layer) != 0:
        executable_gate_nodes = []
        for gate_node in front_layer:
            gate = circuit_dag.get_gate_from_node(gate_node)
            if qubit_graph.check_gate_executable(gate, mapping):
                executable_gate_nodes.append(gate_node)
        
        if len(executable_gate_nodes) != 0:
            # There are gates ready to be executed as-is
            for gate_node in executable_gate_nodes:
                gate = circuit_dag.get_gate_from_node(gate_node)
                gate_execution_log.append((gate.gate_type + " " + str(gate.parameters),(mapping[gate.qubits[0]],mapping[gate.qubits[1]])))
                circuit_dag.remove_gate(gate_node)
                decay_array[mapping[gate.qubits[0]]] = 1
                decay_array[mapping[gate.qubits[1]]] = 1
        else:
            # A SWAP is required for the next gate
            score = dict()
            front_layer_gates = circuit_dag.get_gates_from_nodes(front_layer)
            SWAP_candidates = get_SWAP_candidates(qubit_graph, mapping, front_layer_gates)
            for SWAP_candidate in SWAP_candidates:
                temp_mapping = update_mapping(mapping.copy(),*SWAP_candidate)
                score[SWAP_candidate] = SWAP_heuristic(circuit_dag, temp_mapping, dist_matrix, SWAP_candidate, decay_array)
            best_SWAP = min(score, key=score.get)
            update_mapping(mapping,*best_SWAP)
            gate_execution_log.append(("SWAP", best_SWAP))
            decay_array[best_SWAP[0]] = 1 + DECAY_VALUE
            decay_array[best_SWAP[1]] = 1 + DECAY_VALUE
            for i in range(len(qubit_graph)):
                if decay_array[i]:
                    decay_timer[i]+=1
                if decay_timer[i] > 5:
                    decay_array[i] = 1
                    decay_timer[i] = 0

        front_layer = circuit_dag.get_front_layer()

    return mapping, gate_execution_log


def sabre(qubit_graph, quantum_circuit):
    """
    return values:
        mapping: Bidict mapping where logical qubits are keys, Physical qubits are values
    """
    num_physical_qubits = 0
    num_logical_qubits = 0
    circuit_dag = None
    reverse_circuit_dag = None

    if type(quantum_circuit) == QuantumCircuit:
        quantum_circuit = quantum_circuit.decompose()
        print(quantum_circuit)
        num_logical_qubits = quantum_circuit.num_qubits
        num_physical_qubits = len(qubit_graph)

        reduced_quantum_circuit = reduce_2_qubit_gates(quantum_circuit)
        reverse_quantum_circuit = reduced_quantum_circuit.inverse()
        circuit_dag = from_qiskit(reduced_quantum_circuit)

        reverse_circuit_dag = from_qiskit(reverse_quantum_circuit)

        if num_logical_qubits > num_physical_qubits:
            raise ValueError("Too many circuit qubits to perform algorithm on hardware network")
        
    random_mapping = list(range(num_physical_qubits))
    random.shuffle(random_mapping)
    initial_mapping = bidict(enumerate(random_mapping))

    dist_matrix = qubit_graph.get_distance_matrix()

    final_mapping, _ = sabre_forward_pass(qubit_graph, dist_matrix, initial_mapping, circuit_dag)
    initial_mapping, _ = sabre_forward_pass(qubit_graph, dist_matrix, final_mapping, reverse_circuit_dag)
    _, gate_execution_log = sabre_forward_pass(qubit_graph, dist_matrix, initial_mapping, circuit_dag)

    print("Initial Mapping:")
    for i in range(num_logical_qubits):
        print(f"Logical Qubit {i}: Physical Qubit {initial_mapping[i]}")

    print("Physical Gate Log:")
    for gate_log in gate_execution_log:
        print(f"{gate_log[0]} -> {gate_log[1][0]} {gate_log[1][1]}")
