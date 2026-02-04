import math
from copy import deepcopy
from mapping import Mapping
from qiskit import QuantumCircuit
from convert import from_qiskit

EXTENDED_LAYER_SIZE = 10
EXTENDED_HEURISTIC_WEIGHT = 0.5
DECAY_VALUE = 0.001

def is_2q(gate):
    return len(gate.qubits) == 2

def emit_gate_qiskit(out_qc, gate, mapping):
    """Convert GateNode into Qiskit operation on physical qubits using current mapping"""
    
    phys = [mapping[q] for q in gate.qubits] # here gate.qubits are logical qubits
    gt = gate.gate_type.upper()
    p = gate.parameters or {}

    # 1q gates
    if gt == "H":
        out_qc.h(phys[0])
    elif gt == "X":
        out_qc.x(phys[0])
    elif gt == "Y":
        out_qc.y(phys[0])
    elif gt == "Z":
        out_qc.z(phys[0])
    elif gt == "S":
        out_qc.s(phys[0])
    elif gt == "SDG":
        out_qc.sdg(phys[0])
    elif gt == "T":
        out_qc.t(phys[0])
    elif gt == "TDG":
        out_qc.tdg(phys[0])
    elif gt == "SX":
        out_qc.sx(phys[0])
    elif gt == "RX":
        out_qc.rx(p.get("param_0"), phys[0])
    elif gt == "RY":
        out_qc.ry(p.get("param_0"), phys[0])
    elif gt == "RZ":
        out_qc.rz(p.get("param_0"), phys[0])
    elif gt == "BARRIER":
        out_qc.barrier(*phys)
    elif gt == "MEASURE":
        if len(gate.qubits) != 1 or len(gate.clbits) != 1:
            raise ValueError(f"MEASURE expects 1 qubit and 1 clbit, got {gate.qubits}, {gate.clbits}")
        out_qc.measure(phys[0], gate.clbits[0])
    elif gt in ("U1", "P", "PHASE"):
        lam = p.get("param_0")
        if lam is None:
            raise ValueError(f"U1 missing param_0, gate.parameters={gate.parameters}")
        out_qc.p(lam, phys[0])
    elif gt == "U2":
        phi = p.get("param_0")
        lam = p.get("param_1")
        if phi is None or lam is None:
            raise ValueError(f"U2 missing params, gate.parameters={gate.parameters}")
        out_qc.u(math.pi / 2, phi, lam, phys[0])
    elif gt in ("U3", "U"):
        theta = p.get("param_0")
        phi   = p.get("param_1")
        lam   = p.get("param_2")
        if theta is None or phi is None or lam is None:
            raise ValueError(f"{gt} missing params, gate.parameters={gate.parameters}")
        out_qc.u(theta, phi, lam, phys[0])


    # 2q gates
    elif gt == "CX":
        out_qc.cx(phys[0], phys[1])
    elif gt == "CZ":
        out_qc.cz(phys[0], phys[1])
    elif gt == "SWAP":
        out_qc.swap(phys[0], phys[1])

    else:
        raise ValueError(f"Unsupported gate_type for emission: {gate.gate_type}") # add the gate when it is raised

def safe_swap_mapping(mapping, p1, p2):
    """
    mapping: bidict logical -> physical
    p1, p2: physical qubits to swap
    """
    l1 = mapping.inv.get(p1, None)
    l2 = mapping.inv.get(p2, None)

    # both physical qubits unused
    if l1 is None and l2 is None:
        return mapping

    # one unused, just move the logical qubit
    if l1 is None:
        del mapping[l2]
        mapping[l2] = p1
        return mapping

    if l2 is None:
        del mapping[l1]
        mapping[l1] = p2
        return mapping

    # both used, do a true swap without transient duplication
    del mapping[l1]
    del mapping[l2]
    mapping[l1] = p2
    mapping[l2] = p1
    return mapping


def get_SWAP_candidates(arch, mapping, front_layer_gates):
    front_nodes = set()
    for gate in front_layer_gates:
        for q in gate.qubits:
            front_nodes.add(mapping[q])
    return list(arch.edges(front_nodes))

def SWAP_heuristic(dag, temp_mapping, dist_matrix, swap_candidate, decay_array):
    front_nodes = dag.get_front_layer()
    front_gates = [g for g in dag.get_gates_from_nodes(front_nodes) if is_2q(g)]

    if not front_gates:
        return 0.0
    
    h_basic = 0.0
    for g in front_gates:
        q1, q2 = g.qubits
        h_basic += dist_matrix[temp_mapping[q1]][temp_mapping[q2]]

    p1, p2 = swap_candidate
    decay_factor = 1 + max(decay_array[p1], decay_array[p2])

    ext_nodes = dag.get_extended_layer(EXTENDED_LAYER_SIZE)
    ext_gates = [g for g in dag.get_gates_from_nodes(ext_nodes) if is_2q(g)]

    if not ext_gates:
        return decay_factor * (h_basic / len(front_gates))
    
    h_ext = 0.0
    for g in ext_gates:
        q1, q2 = g.qubits
        h_ext += dist_matrix[temp_mapping[q1]][temp_mapping[q2]]

    return (
        decay_factor* (h_basic / len(front_gates)) + EXTENDED_HEURISTIC_WEIGHT * (h_ext / len(ext_gates))
    )

def sabre_swap(arch, quantum_circuit, initial_mapping):
    """
    Routes QuantumDAG by inserting SWAPs, starting from initial mapping
    
    Returns:
        routed_qc: Routed QuantumCricuit
        final_mapping: bidict logical -> physical
        log: list of ("GATE", ...) abd (SWAP, (p1,p2))
    """

    if isinstance(quantum_circuit, QuantumCircuit):
        dag = from_qiskit(quantum_circuit)
    else:
        raise ValueError("SABRE Router only accepts Qiskit QuantumCircuit")
    
    mapping = Mapping(initial_mapping.copy())
    num_physical_qubits = len(arch)
    dist_matrix = arch.get_distance_matrix()

    routed_qc = QuantumCircuit(num_physical_qubits, dag.num_clbits)
    decay_array = [1.0 for _ in range(num_physical_qubits)]
    decay_timer = [0 for _ in range(num_physical_qubits)]
    log = []

    while True:
        front = dag.get_front_layer()
        if not front:
            break

        executable = []
        for node in front:
            gate = dag.get_gate_from_node(node)

            if not is_2q(gate):
                executable.append(node)
                continue

            if arch.check_gate_executable(gate, mapping):
                executable.append(node)
        
        if executable:
            for node in executable:
                gate = dag.get_gate_from_node(node)

                emit_gate_qiskit(routed_qc, gate, mapping)
                log.append(("GATE", (gate.gate_type, tuple(mapping[q] for q in gate.qubits))))

                for q in gate.qubits:
                    decay_array[mapping[q]] = 1.0

                dag.remove_gate(node)

        else:
            front_gates = [g for g in dag.get_gates_from_nodes(front) if is_2q(g)]
            swap_cands = get_SWAP_candidates(arch, mapping, front_gates)
            if not swap_cands:
                raise RuntimeError("No SWAP candidates found, check graph connectivity and mapping.")
            
            scores = {}
            for (p1, p2) in swap_cands:
                tmp = Mapping(mapping.copy())
                safe_swap_mapping(tmp, p1, p2)
                scores[(p1, p2)] = SWAP_heuristic(dag, tmp, dist_matrix, (p1, p2), decay_array)
            
            best = min(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
            p1, p2 = best

            routed_qc.swap(p1, p2)
            safe_swap_mapping(mapping, p1, p2)
            log.append(("SWAP", (p1, p2)))

            decay_array[p1] = 1.0 + DECAY_VALUE
            decay_array[p2] = 1.0 + DECAY_VALUE

            # only tick timers for penalised qubits
            for i in range(num_physical_qubits):
                if decay_array[i] > 1.0:
                    decay_timer[i] += 1
                    if decay_timer[i] > 5:
                        decay_array[i] = 1.0
                        decay_timer[i] = 0
            
    return routed_qc, mapping, log