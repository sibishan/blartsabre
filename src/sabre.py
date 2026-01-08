from .convert import from_qiskit, from_qasm_file
from qiskit import QuantumCircuit

def reduce_2_qubit_gates(circuit):
    for i in range(len(circuit.data)-1,-1,-1):
        if circuit.data[i].operation.num_qubits != 2:
            del circuit.data[i]
    return circuit

def get_SWAP_candidates(graph, front_layer):
    # Current implementation: get all adjacent SWAPs of front layer qubits
    pass

def SWAP_heuristic(front_layer, circuit_dag, temp_mapping, dist_matrix, SWAP_candidate):
    pass

def sabre_forward_pass(qubit_graph, dist_matrix, initial_mapping, circuit_dag):

    mapping = initial_mapping
    inserted_SWAPs = []

    front_layer = circuit_dag.get_front_layer()
    while front_layer.size() != 0:
        executable_gates = []
        for gate in front_layer:
            if qubit_graph.check_gate_executable(gate):
                executable_gates.append(gate)
        
        if len(executable_gates) != 0:
            # There are gates ready to be executed as-is
            for gate in executable_gates:
                front_layer.remove(gate) # one of these
                circuit_dag.remove(gate) # one of these
                successor_gates = circuit_dag.get_successors(gate)
                if None: #successor_gates dependencies are resolved
                    front_layer.append(gate)
        else:
            # A SWAP is required for the next gate
            score = dict()
            SWAP_candidates = get_SWAP_candidates(qubit_graph, front_layer)
            for SWAP_candidate in SWAP_candidates:
                temp_mapping = mapping.swap(SWAP_candidate)
                score[SWAP_candidate] = SWAP_heuristic(front_layer, circuit_dag, temp_mapping, dist_matrix, SWAP_candidate)
            best_SWAP = score.min()
            mapping = mapping.swap(best_SWAP)
            inserted_SWAPs.append((best_SWAP, front_layer))

        front_layer = circuit_dag.get_front_layer()

    return mapping, inserted_SWAPs


def sabre(qubit_graph, quantum_circuit):
    if type(quantum_circuit) == QuantumCircuit:
        if quantum_circuit.num_qubits > len(qubit_graph):
            raise ValueError("Too many circuit qubits to perform algorithm on hardware network")
        

        reduced_quantum_circuit = reduce_2_qubit_gates(quantum_circuit)
        reverse_quantum_circuit = reduced_quantum_circuit.inverse()
        circuit_dag = from_qiskit(reduced_quantum_circuit)
        reverse_circuit_dag = from_qiskit(reverse_quantum_circuit)

    initial_mapping = None #random mapping

    dist_matrix = qubit_graph.get_distance_matrix()

    final_mapping, _ = sabre_forward_pass(qubit_graph, dist_matrix, initial_mapping, circuit_dag)
    _, inserted_SWAPs = sabre_forward_pass(qubit_graph, dist_matrix, final_mapping, reverse_circuit_dag)

    print(final_mapping)
    print(inserted_SWAPs)
