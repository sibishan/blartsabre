import time
import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure,
    Unroll3qOrMore,
    RemoveResetInZeroState,
    Decompose
)


def load_qasm(dir, pattern="*.qasm", recursive=False, sort=True, return_dict=True, show_progress=True):
    dir = Path(dir).expanduser().resolve()
    if not dir.exists() or not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir}")
    
    globber = dir.rglob if recursive else dir.glob
    files = list(globber(pattern))

    if sort:
        files.sort(key=lambda p: p.name)
    
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {dir}")

    loader = lambda path: QuantumCircuit.from_qasm_file(str(path))

    loaded = []
    errors = []

    iterable = files
    if show_progress:
        from tqdm import tqdm
        iterable = tqdm(files, desc="Loading QASM", unit="file")

    total = len(files)
    for idx, path in enumerate(iterable, start=1):
        try:
            loaded.append((path.stem, loader(path)))
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}: {e}")

    if errors:
        preview = "\n".join(errors[:10])
        more = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more"
        raise RuntimeError(f"Some QASM files failed to load:\n{preview}{more}")
    
    if return_dict:
        return {name: qc for name, qc in loaded}
    return [qc for _, qc in loaded]


def get_non_single_qg_names(non_single_qg):
    """From the provided non-single-qubit gates, get all of the gate names"""
    non_single_qg_names = set()
    for gate in non_single_qg:
        if gate.name == "cx":
            continue
        non_single_qg_names.add(gate.name)
    return list(non_single_qg_names)

def init_circuit(qc, verbose=False):
    dag = circuit_to_dag(qc)
    two_qg_list = dag.two_qubit_ops()
    mul_qg_list = dag.multi_qubit_ops()
    non_single_qg_names = get_non_single_qg_names(two_qg_list + mul_qg_list)
    if verbose:
        if non_single_qg_names:
            print(f"GATES TO DECOMPOSE: {non_single_qg_names}")
        else:
            print("NO GATES TO DECOMPOSE")

    init_pm = PassManager()
    init_pm.append([
        Unroll3qOrMore(),
        RemoveResetInZeroState(),
        OptimizeSwapBeforeMeasure(),
        RemoveDiagonalGatesBeforeMeasure()
    ])

    # if there are non-single-qubit gates which are not CNOTs, decompose those gates before routing
    if non_single_qg_names:
        init_pm.append(Decompose(non_single_qg_names))

    init_start = time.perf_counter()
    init_cir = init_pm.run(qc)
    init_end = time.perf_counter()
    init_time = init_end - init_start
    
    og_ops = init_cir.count_ops()
    og_cx = og_ops.get("cx", 0)
    og_swaps = og_ops.get("swap", 0)
    og_depth = init_cir.depth()
    num_qubits = init_cir.num_qubits
    og_size = init_cir.size()

    return init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size

def save_stats_json(all_rows, out_path, indent=2):
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=indent)

    return out_path

def ops_from_our_log(gate_execution_log):
    """
      ('gate', p1, p2)                 for CX
      ('move', p1, p2)                 for SWAP
      ('move', p1, p2, p3)             for Teleport
      ('move', p1, p2, p3, p4)         for Telegate
    """
    ops = []
    for op, payload in gate_execution_log:
        if op == "SWAP":
            p1, p2 = map(int, payload)
            ops.append(("move", p1, p2))

        elif op == "Teleport":
            p1, p2, p3 = map(int, payload)
            ops.append(("move", p1, p2, p3))

        elif op == "Telegate":
            p1, p2, p3, p4 = map(int, payload)
            ops.append(("move", p1, p2, p3, p4))

        else:
            # executed gate, you said only CX and SWAP matter
            name = str(op).upper()
            if "CX" in name:
                p1, p2 = map(int, payload)
                ops.append(("gate", p1, p2))

    return ops


def mapped_ops(operations, num_phys_qubits):
    swap_count = 0
    teleportation_count = 0
    telegate_count = 0

    used_qubits = [False] * num_phys_qubits
    for op in operations:
        tag = op[0]
        qubits = op[1:]

        if tag == "move":
            if len(qubits) == 2 and any(used_qubits[q] for q in qubits):
                swap_count += 1
                for q in qubits:
                    used_qubits[q] = True

            elif len(qubits) == 3 and any(used_qubits[q] for q in qubits):
                teleportation_count += 1
                for q in qubits:
                    used_qubits[q] = True

            elif len(qubits) == 4:
                telegate_count += 1
                for q in qubits:
                    used_qubits[q] = True

        elif tag == "gate":
            for q in qubits:
                used_qubits[q] = True

    return swap_count, teleportation_count, telegate_count


def mapped_depth(operations, num_phys_qubits, tp_duration=5):
    qubit_depths = [0] * num_phys_qubits
    qubit_depths_tp = [0] * num_phys_qubits

    for op in operations:
        qubits = op[1:]
        max_depth = max(qubit_depths[p] for p in qubits)

        duration = 1
        if op[0] == "move":
            if len(qubits) == 2:
                duration = 1
            elif len(qubits) in (3, 4):
                duration = tp_duration

        for p in qubits:
            qubit_depths[p] = max_depth + duration
            if op[0] == "move" and len(qubits) > 2:
                qubit_depths_tp[p] = max_depth + duration

    circuit_depth = max(qubit_depths) if qubit_depths else 0
    teleport_depth = max(qubit_depths_tp) if qubit_depths_tp else 0
    return circuit_depth, teleport_depth


def stats_from_our_telesabre_log(gate_execution_log, num_phys_qubits):
    ops = ops_from_our_log(gate_execution_log)

    (swap_count, teleport_count, telegate_count) = mapped_ops(ops, num_phys_qubits)

    depth, tp_depth = mapped_depth(ops, num_phys_qubits, tp_duration=5)

    cx_count = sum(1 for op in ops if op[0] == "gate" and len(op[1:]) == 2)

    return {
        "mapped_swaps": swap_count,
        "mapped_teleports": teleport_count,
        "mapped_telegates": telegate_count,
        "mapped_depth": depth,
        "mapped_teleport_depth": tp_depth,
        "mapped_cx": cx_count,
    }

def blart_counts_from_log(gate_execution_log, num_phys_qubits):
    local_swaps = 0
    remote_swaps = 0
    telegates = 0
    cx_local = 0

    used = [False] * num_phys_qubits

    for op, payload in gate_execution_log:
        name = str(op)
        up = name.upper()

        # swaps (moves)
        if name == "SWAP" or name == "Remote SWAP":
            p1, p2 = map(int, payload)
            if used[p1] or used[p2]:
                if name == "SWAP":
                    local_swaps += 1
                else:
                    remote_swaps += 1
                used[p1] = True
                used[p2] = True
            continue

        # remote gate cx (telegate)
        if up.startswith("REMOTE GATE") and "CX" in up and len(payload) == 2:
            p1, p2 = map(int, payload)
            telegates += 1
            used[p1] = True
            used[p2] = True
            continue

        # local cx
        if "CX" in up and len(payload) == 2:
            p1, p2 = map(int, payload)
            cx_local += 1
            used[p1] = True
            used[p2] = True
            continue

    return {
        "mapped_swaps": local_swaps + remote_swaps,
        "mapped_swaps_local": local_swaps,
        "mapped_swaps_remote": remote_swaps,
        "mapped_telegates": telegates,
        "mapped_cx_local": cx_local,
    }


def blart_depths_from_log(
    gate_execution_log,
    num_phys_qubits,
    duration_cx=1,
    duration_swap=1,
    duration_remote_swap=5,
    duration_telegate=5,
):
    depth = [0] * num_phys_qubits
    remote_depth = [0] * num_phys_qubits

    for op, payload in gate_execution_log:
        name = str(op)
        up = name.upper()

        if len(payload) != 2:
            continue

        p1, p2 = map(int, payload)

        # classify op + duration
        is_remote = False
        dur = None

        if name == "SWAP":
            dur = duration_swap

        elif name == "Remote SWAP":
            dur = duration_remote_swap
            is_remote = True

        elif up.startswith("REMOTE GATE") and "CX" in up:
            dur = duration_telegate
            is_remote = True

        elif "CX" in up:
            dur = duration_cx

        else:
            continue

        start = max(depth[p1], depth[p2])
        end = start + dur
        depth[p1] = end
        depth[p2] = end

        if is_remote:
            rstart = max(remote_depth[p1], remote_depth[p2])
            rend = rstart + dur
            remote_depth[p1] = rend
            remote_depth[p2] = rend

    return (max(depth) if depth else 0), (max(remote_depth) if remote_depth else 0)


def stats_from_blartsabre_log(
    gate_execution_log,
    num_phys_qubits,
    duration_remote_swap=5,
    duration_telegate=5,
):
    counts = blart_counts_from_log(gate_execution_log, num_phys_qubits)
    mapped_depth, mapped_remote_depth = blart_depths_from_log(
        gate_execution_log,
        num_phys_qubits,
        duration_cx=1,
        duration_swap=1,
        duration_remote_swap=duration_remote_swap,
        duration_telegate=duration_telegate,
    )

    return {
        **counts,
        "mapped_depth": mapped_depth,
        "mapped_remote_depth": mapped_remote_depth,
    }

