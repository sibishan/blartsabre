import random
import math
from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister

from mapper.telesabre import get_gate_paths, get_SWAP_candidates, get_teleport_candidates, mapping_energy
from convert import from_qiskit
from mapping import Mapping

class DeadlockError(RuntimeError):
    pass

def is_2q(gate):
    return len(gate.qubits) == 2

_TELE_REG_COUNTER = {"teledata": 0, "telegate": 0, "tele": 0}

def _alloc_cbits(qc, n, name="tele"):
    if not hasattr(qc, "_tele_reg_counter"):
        qc._tele_reg_counter = {}

    idx = qc._tele_reg_counter.get(name, 0)
    qc._tele_reg_counter[name] = idx + 1

    reg_name = f"{name}_{idx}"
    creg = ClassicalRegister(n, reg_name)
    qc.add_register(creg)
    return list(creg)


def _apply_if_one(qc, clbit, apply_fn):
    try:
        with qc.if_test((clbit, 1)):
            apply_fn()
    except Exception:
        inst = apply_fn()
        if inst is None:
            raise RuntimeError("Fallback c_if requires apply_fn to return the instruction handle.")
        inst.c_if(clbit, 1)

def emit_teledata_teleport(qc, p_data_src, p_comm_src, p_comm_dst, *,
                          create_epr=True, log=None):
    
    c_data, c_comm = _alloc_cbits(qc, 2, name="teledata")

    if log is not None:
        log.append(("TELEDATA", (p_data_src, p_comm_src, p_comm_dst)))

    if create_epr:
        qc.h(p_comm_src)
        qc.cx(p_comm_src, p_comm_dst)

    qc.cx(p_data_src, p_comm_src)
    qc.h(p_data_src)

    qc.measure(p_comm_src, c_comm)
    qc.measure(p_data_src, c_data)

    _apply_if_one(qc, c_comm, lambda: qc.x(p_comm_dst))
    _apply_if_one(qc, c_data, lambda: qc.z(p_comm_dst))


def emit_telegate_cx(qc, p_data_ctrl, p_comm_ctrl, p_comm_tgt, p_data_tgt, *,
                     create_epr=True, log=None):
    
    c_ctrl_comm, c_tgt_comm = _alloc_cbits(qc, 2, name="telegate")

    if log is not None:
        log.append(("TELEGATE_CX", (p_data_ctrl, p_comm_ctrl, p_comm_tgt, p_data_tgt)))

    if create_epr:
        qc.h(p_comm_ctrl)
        qc.cx(p_comm_ctrl, p_comm_tgt)

    qc.cx(p_data_ctrl, p_comm_ctrl)

    qc.cx(p_comm_tgt, p_data_tgt)

    qc.h(p_comm_tgt)
    qc.measure(p_comm_ctrl, c_ctrl_comm)
    qc.measure(p_comm_tgt, c_tgt_comm)

    _apply_if_one(qc, c_tgt_comm, lambda: qc.z(p_data_ctrl))
    _apply_if_one(qc, c_ctrl_comm, lambda: qc.x(p_data_tgt))


def emit_gate_qiskit(out_qc, gate, mapping):
    phys = [mapping[q] for q in gate.qubits]  # logical -> physical
    gt = gate.gate_type.upper()
    p = gate.parameters or {}

    # 1q
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
            raise ValueError(f"U1/P missing param_0, gate.parameters={gate.parameters}")
        out_qc.p(lam, phys[0])
    elif gt == "U2":
        phi = p.get("param_0")
        lam = p.get("param_1")
        if phi is None or lam is None:
            raise ValueError(f"U2 missing params, gate.parameters={gate.parameters}")
        out_qc.u(math.pi / 2, phi, lam, phys[0])
    elif gt in ("U3", "U"):
        theta = p.get("param_0")
        phi = p.get("param_1")
        lam = p.get("param_2")
        if theta is None or phi is None or lam is None:
            raise ValueError(f"{gt} missing params, gate.parameters={gate.parameters}")
        out_qc.u(theta, phi, lam, phys[0])

    # 2q
    elif gt == "CX":
        out_qc.cx(phys[0], phys[1])
    elif gt == "CZ":
        out_qc.cz(phys[0], phys[1])
    elif gt == "SWAP":
        out_qc.swap(phys[0], phys[1])
    else:
        raise ValueError(f"Unsupported gate_type for emission: {gate.gate_type}")


def _emit_swap_physical(routed_qc, mapping: Mapping, p1: int, p2: int, log):
    routed_qc.swap(p1, p2)
    mapping.swap_p_qubits(p1, p2)
    log.append(("SWAP", (p1, p2)))


def apply_teleport_op(arch, routed_qc, mapping: Mapping, op, log, *, create_epr=True):

    if len(op) != 3:
        raise ValueError(f"Teleport op must be len 3, got {op}")

    p_data_src, p_comm_src, p_comm_dst = op

    free = set(mapping.get_free_p_nodes())
    if p_comm_src not in free or p_comm_dst not in free:
        raise RuntimeError(f"Teleport requires comm qubits free, op={op}, free={sorted(free)}")

    if hasattr(mapping, "p_to_l"):
        lq = mapping.p_to_l(p_data_src)
    else:
        inv = getattr(mapping, "inv", None)
        lq = inv.get(p_data_src, None) if inv is not None else None

    if lq is None:
        raise RuntimeError(f"Teleport called on empty physical qubit p_data_src={p_data_src}, op={op}")

    emit_teledata_teleport(
        routed_qc,
        p_data_src=p_data_src,
        p_comm_src=p_comm_src,
        p_comm_dst=p_comm_dst,
        create_epr=create_epr,
        log=log,
    )

    if hasattr(mapping, "move_l_qubit"):
        mapping.move_l_qubit(lq, p_comm_dst)   
    else:
        if hasattr(mapping, "remove_l_qubit"):
            mapping.remove_l_qubit(lq)
            mapping.set_l_qubit(lq, p_comm_dst)
        else:
            mapping.swap_p_qubits(p_data_src, p_comm_dst)

    log.append(("TELEPORT", op))


def apply_telegate_op(arch, routed_qc, mapping: Mapping, op, log, *, create_epr=True):

    if len(op) != 4:
        raise ValueError(f"Telegate op must be len 4, got {op}")

    p_data_ctrl, p_comm_ctrl, p_comm_tgt, p_data_tgt = op

    free = set(mapping.get_free_p_nodes())
    if p_comm_ctrl not in free or p_comm_tgt not in free:
        raise RuntimeError(f"Telegate requires comm qubits free, op={op}, free={sorted(free)}")

    emit_telegate_cx(
        routed_qc,
        p_data_ctrl=p_data_ctrl,
        p_comm_ctrl=p_comm_ctrl,
        p_comm_tgt=p_comm_tgt,
        p_data_tgt=p_data_tgt,
        create_epr=create_epr,
        log=log,
    )

    log.append(("TELEGATE", op))


def telesabre_swap(
    arch,
    qc,
    initial_mapping,
    *,
    EXTENDED_LAYER_SIZE=10,
    EXTENDED_HEURISTIC_WEIGHT=0.25,
    DECAY_VALUE=0.001,
    RESET_TIMER_START=50,
    create_epr=True,
):

    if not isinstance(qc, QuantumCircuit):
        raise ValueError("telesabre_router expects a Qiskit QuantumCircuit")

    dag = from_qiskit(qc)

    mapping = Mapping(initial_mapping.copy()) if not isinstance(initial_mapping, Mapping) else initial_mapping.copy()
    num_physical = len(arch)

    routed_qc = QuantumCircuit(num_physical, dag.num_clbits)
    decay_array = [1.0 for _ in range(num_physical)]
    decay_timer = [0 for _ in range(num_physical)]
    log = []

    reset_timer = RESET_TIMER_START

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

                # log physical operands
                if gate.gate_type.upper() == "MEASURE":
                    log.append(("GATE", ("MEASURE", (mapping[gate.qubits[0]], gate.clbits[0]))))
                else:
                    log.append(("GATE", (gate.gate_type, tuple(mapping[q] for q in gate.qubits))))

                for q in gate.qubits:
                    decay_array[mapping[q]] = 1.0

                dag.remove_gate(node)

            reset_timer = RESET_TIMER_START

        else:
            front_gates = [g for g in dag.get_gates_from_nodes(front) if is_2q(g)]

            gate_paths = get_gate_paths(arch, mapping, front_gates, True)

            swap_cands = get_SWAP_candidates(arch, mapping, front_gates, gate_paths, True)
            tele_cands = get_teleport_candidates(arch, mapping, front_gates, gate_paths, True)

            op_cands = list(swap_cands) + list(tele_cands)
            if not op_cands:
                raise RuntimeError("No candidates found, check connectivity, free qubits, and candidate generation")

            scores = {}
            for i, op in enumerate(op_cands):
                scores[i] = mapping_energy(arch, dag, mapping, op, decay_array, True)

            best_i = min(scores, key=scores.get)
            best_op = op_cands[best_i]

            if hasattr(arch, "clear_active_telegate_qubits"):
                arch.clear_active_telegate_qubits()

            if len(best_op) == 2:
                p1, p2 = best_op
                _emit_swap_physical(routed_qc, mapping, p1, p2, log)

            elif len(best_op) == 3:
                apply_teleport_op(arch, routed_qc, mapping, best_op, log, create_epr=create_epr)

            elif len(best_op) == 4:
                apply_telegate_op(arch, routed_qc, mapping, best_op, log, create_epr=create_epr)

                if hasattr(arch, "register_active_telegate_qubits"):
                    p_data_ctrl, _, _, p_data_tgt = best_op
                    arch.register_active_telegate_qubits(p_data_ctrl, p_data_tgt)

            else:
                raise RuntimeError(f"Unknown op length {len(best_op)} for op={best_op}")

            decay_array[best_op[0]] = 1.0 + DECAY_VALUE
            decay_array[best_op[-1]] = 1.0 + DECAY_VALUE

            for q in range(num_physical):
                if decay_array[q] > 1.0:
                    decay_timer[q] += 1
                    if decay_timer[q] > 5:
                        decay_array[q] = 1.0
                        decay_timer[q] = 0

            reset_timer -= 1
            if reset_timer < 0:
                raise DeadlockError("reset_timer expired in telesabre_router")

    return routed_qc, mapping, log
