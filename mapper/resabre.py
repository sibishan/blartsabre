from convert import from_qiskit
from qiskit import QuantumCircuit
from mapping import Mapping
import random
from copy import deepcopy

EXTENDED_LAYER_SIZE = 10
EXTENDED_HEURISTIC_WEIGHT = 0.5
DECAY_VALUE = 0.001
COMM_SWAP_LAMBDA = 2.0
RESET_TIMER_START = 5000
# STAGNATION_WINDOW = 400  # how many swaps with no executed gate before we relax comm penalty

NUM_ITERATIONS = 5

class DeadlockError(RuntimeError): pass

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
    
    u, v = SWAP_candidate
    decay_factor = 1 + max(decay_array[u], decay_array[v])

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
    mapping.swap_p_qubits(p_q1, p_q2)
    return mapping


def sabre_forward_pass(arch, dist_matrix, initial_mapping, circuit_dag):
    circuit_dag = deepcopy(circuit_dag)
    mapping = initial_mapping.copy()

    gate_execution_log = []

    n_phys = len(arch)
    decay_array = [1.0 for _ in range(n_phys)]
    decay_timer = [0 for _ in range(n_phys)]

    front_layer = circuit_dag.get_front_layer()

    reset_timer = RESET_TIMER_START
    no_progress_swaps = 0  # counts swaps since last executed gate

    while len(front_layer) != 0:
        executable_gate_nodes = []
        for gate_node in front_layer:
            gate = circuit_dag.get_gate_from_node(gate_node)
            if arch.check_gate_executable(gate, mapping):
                executable_gate_nodes.append(gate_node)

        if executable_gate_nodes:
            for gate_node in executable_gate_nodes:
                gate = circuit_dag.get_gate_from_node(gate_node)
                p1 = mapping[gate.qubits[0]]
                p2 = mapping[gate.qubits[1]]
                gate_execution_log.append((gate.gate_type + " " + str(gate.parameters), (p1, p2)))

                circuit_dag.remove_gate(gate_node)

                decay_array[p1] = 1.0
                decay_array[p2] = 1.0
                decay_timer[p1] = 0
                decay_timer[p2] = 0

            no_progress_swaps = 0
            reset_timer = RESET_TIMER_START

        else:
            score = {}
            front_layer_gates = circuit_dag.get_gates_from_nodes(front_layer)
            SWAP_candidates = list(get_SWAP_candidates(arch, mapping, front_layer_gates))

            if not SWAP_candidates:
                raise DeadlockError("No SWAP candidates available")

            # if we have been swapping too long with no progress, relax comm penalty for one step
            # relax_comm = (no_progress_swaps >= STAGNATION_WINDOW)

            for u, v in SWAP_candidates:
                temp_mapping = update_mapping(mapping.copy(), u, v)

                h = SWAP_heuristic(
                    circuit_dag=circuit_dag,
                    temp_mapping=temp_mapping,
                    dist_matrix=dist_matrix,
                    SWAP_candidate=(u, v),
                    decay_array=decay_array
                )

                decay_factor = 1.0 + max(decay_array[u], decay_array[v])
                h *= decay_factor

                # # comm penalty
                # if relax_comm:
                #     comm_penalty = 0.0
                # else:
                #     comm_penalty = COMM_SWAP_LAMBDA if arch.is_comm_edge(u, v) else 0.0
                #     comm_penalty = COMM_SWAP_LAMBDA if arch.is_comm_edge(u, v) else 0.0
                comm_penalty = COMM_SWAP_LAMBDA if arch.is_comm_edge(u, v) else 0.0

                score[(u, v)] = h + comm_penalty

            best_u, best_v = min(score, key=score.get)

            update_mapping(mapping, best_u, best_v)
            gate_execution_log.append(("SWAP", (best_u, best_v)))

            decay_array[best_u] = 1.0 + DECAY_VALUE
            decay_array[best_v] = 1.0 + DECAY_VALUE
            decay_timer[best_u] = 0
            decay_timer[best_v] = 0

            for i in range(n_phys):
                if decay_array[i] != 1.0:
                    decay_timer[i] += 1
                    if decay_timer[i] > 5:
                        decay_array[i] = 1.0
                        decay_timer[i] = 0

            no_progress_swaps += 1
            reset_timer -= 1
            if reset_timer <= 0:
                raise DeadlockError("reset_timer expired, stuck swapping")

        front_layer = circuit_dag.get_front_layer()

    return mapping, gate_execution_log



def resabre(arch, quantum_circuit, verbose = False, return_log = False, seed=None):
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

    deadlocks = 0
    for iteration in range(NUM_ITERATIONS):
        try:
            random_mapping = list(range(num_physical_qubits))
            random.shuffle(random_mapping)
            initial_mapping = Mapping(enumerate(random_mapping))


            final_mapping, _ = sabre_forward_pass(arch, dist_matrix, initial_mapping, circuit_dag)
            initial_mapping, _ = sabre_forward_pass(arch, dist_matrix, final_mapping, reverse_circuit_dag)
            _, gate_execution_log = sabre_forward_pass(arch, dist_matrix, initial_mapping, circuit_dag)

            gate_execution_log_iterations[iteration] = (initial_mapping,gate_execution_log)
        except DeadlockError:
            print(deadlocks)
            deadlocks += 1
            continue
    
    if not gate_execution_log_iterations:
        raise RuntimeError(f"No successful iterations, deadlocks={deadlocks}/{NUM_ITERATIONS}")

    best_iteration = min(gate_execution_log_iterations, key=lambda k: len(gate_execution_log_iterations[k][1]))
    best_initial_mapping, best_gate_execution_log = gate_execution_log_iterations[best_iteration]

    comm_swaps = sum(
        1 for op, e in best_gate_execution_log
        if op == "SWAP" and arch.is_comm_edge(*e)
    )
    total_swaps = sum(1 for op, _ in best_gate_execution_log if op == "SWAP")
    print("comm_swaps", comm_swaps, "total_swaps", total_swaps)

    if verbose:
        best_swap_log = [k for k in best_gate_execution_log if (k[0] == "SWAP")]
        print("Best Iteration:")
        print(f"#{len(best_gate_execution_log)} total gates")
        print(f"#{len(best_swap_log)} inserted SWAPS")
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