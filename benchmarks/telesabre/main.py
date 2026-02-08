import time
from itertools import product
from copy import deepcopy
from tqdm import tqdm
from qiskit import qasm2

from .architecture import TWO_TOKYO, FIVE_TOKYO
from .circuit import Circuit

# og telesabre
from .config import Config
from .telesabre import run_telesabre as og_telesabre_layout
config = Config()

# our telesabre
from mapper.telesabre import telesabre_layout
from router.telesabre import telesabre_swap

from architecture import two_tokyo, five_tokyo
from benchmarks.utils import load_qasm, init_circuit, save_stats_json

BASE_SEED = 42

CIRCUITS = load_qasm("./data/telesabre", recursive=True)

def build_our_arch(num_qubits):
    if num_qubits <= 25:
        arch = two_tokyo()
    else:
        arch = five_tokyo()

    return arch

def build_og_arch(num_qubits):
    if num_qubits <= 25:
        arch = TWO_TOKYO()
    else:
        arch = FIVE_TOKYO()

    return arch


def validate_log(arch, log, *, require_pair_tags=True):

    def _has_node(u):
        try:
            return (u in arch)
        except TypeError:
            # if arch isn't directly iterable, assume 0..len(arch)-1
            return 0 <= int(u) < len(arch)

    def _check_nodes(step, *nodes):
        for u in nodes:
            if not _has_node(u):
                raise RuntimeError(f"Step {step}: invalid physical qubit {u}")

    def _check_edge(step, u, v, label):
        if not arch.has_edge(u, v):
            raise RuntimeError(f"Step {step}: illegal {label}: ({u},{v}) not an edge")

    def _norm_phys(phys):
        if isinstance(phys, int):
            return (phys,)
        if isinstance(phys, tuple):
            return phys
        return tuple(phys)

    i = 0
    n = len(log)

    while i < n:
        op, payload = log[i]

        # ---------------- SWAP ----------------
        if op == "SWAP":
            if not (isinstance(payload, (tuple, list)) and len(payload) == 2):
                raise RuntimeError(f"Step {i}: SWAP payload must be (u,v), got {payload}")
            u, v = payload
            _check_nodes(i, u, v)
            _check_edge(i, u, v, "SWAP")
            i += 1
            continue

        # ---------------- GATE ----------------
        if op == "GATE":
            if not (isinstance(payload, (tuple, list)) and len(payload) == 2):
                raise RuntimeError(f"Step {i}: GATE payload must be (gt, phys), got {payload}")

            gt, phys = payload
            gt = str(gt).upper()
            phys = _norm_phys(phys)
            _check_nodes(i, *phys)

            if gt in ("CX", "CZ", "SWAP"):
                if len(phys) != 2:
                    raise RuntimeError(f"Step {i}: {gt} expected 2 qubits, got {phys}")
                u, v = phys
                _check_edge(i, u, v, gt)

            # 1q gates are already node-checked, multi-q non-routing ops also node-checked
            i += 1
            continue

        # ------------- TELEDATA / TELEPORT -------------
        if op in ("TELEDATA", "TELEPORT"):
            if not (isinstance(payload, (tuple, list)) and len(payload) == 3):
                raise RuntimeError(f"Step {i}: {op} payload must be (p_data_src,p_comm_src,p_comm_dst), got {payload}")

            p_data_src, p_comm_src, p_comm_dst = payload
            _check_nodes(i, p_data_src, p_comm_src, p_comm_dst)

            # local Bell measurement CX(data_src, comm_src)
            _check_edge(i, p_data_src, p_comm_src, f"{op} local CX(data_src,comm_src)")
            # EPR creation CX(comm_src, comm_dst)
            _check_edge(i, p_comm_src, p_comm_dst, f"{op} EPR CX(comm_src,comm_dst)")

            if require_pair_tags:
                if op == "TELEDATA":
                    if i + 1 >= n:
                        raise RuntimeError(f"Step {i}: TELEDATA must be followed by TELEPORT, but log ended")
                    op2, payload2 = log[i + 1]
                    if op2 != "TELEPORT" or tuple(payload2) != tuple(payload):
                        raise RuntimeError(
                            f"Step {i}: TELEDATA must be immediately followed by matching TELEPORT. "
                            f"Next is {(op2, payload2)}"
                        )
                    i += 2
                    continue

                # If TELEPORT appears without TELEDATA immediately before, still accept unless pairing is strict
                if op == "TELEPORT" and i > 0:
                    op_prev, payload_prev = log[i - 1]
                    if op_prev != "TELEDATA" or tuple(payload_prev) != tuple(payload):
                        raise RuntimeError(
                            f"Step {i}: TELEPORT must be immediately preceded by matching TELEDATA. "
                            f"Prev is {(op_prev, payload_prev)}"
                        )

            i += 1
            continue

        # ------------- TELEGATE_CX / TELEGATE -------------
        if op in ("TELEGATE_CX", "TELEGATE"):
            if not (isinstance(payload, (tuple, list)) and len(payload) == 4):
                raise RuntimeError(f"Step {i}: {op} payload must be (p_data_ctrl,p_comm_ctrl,p_comm_tgt,p_data_tgt), got {payload}")

            p_data_ctrl, p_comm_ctrl, p_comm_tgt, p_data_tgt = payload
            _check_nodes(i, p_data_ctrl, p_comm_ctrl, p_comm_tgt, p_data_tgt)

            # local CX(data_ctrl -> comm_ctrl)
            _check_edge(i, p_data_ctrl, p_comm_ctrl, f"{op} local CX(data_ctrl,comm_ctrl)")
            # local CX(comm_tgt -> data_tgt)
            _check_edge(i, p_comm_tgt, p_data_tgt, f"{op} local CX(comm_tgt,data_tgt)")
            # EPR creation CX(comm_ctrl -> comm_tgt)
            _check_edge(i, p_comm_ctrl, p_comm_tgt, f"{op} EPR CX(comm_ctrl,comm_tgt)")

            if require_pair_tags:
                if op == "TELEGATE_CX":
                    if i + 1 >= n:
                        raise RuntimeError(f"Step {i}: TELEGATE_CX must be followed by TELEGATE, but log ended")
                    op2, payload2 = log[i + 1]
                    if op2 != "TELEGATE" or tuple(payload2) != tuple(payload):
                        raise RuntimeError(
                            f"Step {i}: TELEGATE_CX must be immediately followed by matching TELEGATE. "
                            f"Next is {(op2, payload2)}"
                        )
                    i += 2
                    continue

                if op == "TELEGATE" and i > 0:
                    op_prev, payload_prev = log[i - 1]
                    if op_prev != "TELEGATE_CX" or tuple(payload_prev) != tuple(payload):
                        raise RuntimeError(
                            f"Step {i}: TELEGATE must be immediately preceded by matching TELEGATE_CX. "
                            f"Prev is {(op_prev, payload_prev)}"
                        )

            i += 1
            continue

        # ------------- Unknown ops -------------
        raise RuntimeError(f"Step {i}: unknown log op {op} with payload {payload}")

    return True

def run_og_telesabre_pass(qc, arch, seed=BASE_SEED):
    cir = Circuit.from_qiskit(qc)

    map_start = time.perf_counter()
    swap_count, teleportation_count, telegate_count, circuit_depth, teleport_depth, solved_deadlocks, _, solving_deadlock = og_telesabre_layout(
        config, cir, arch, seed=seed
        )
    map_end = time.perf_counter()
    mapping_time = map_end - map_start

    return {
        "seed": seed,
        "mapped_swaps": swap_count,
        "mapped_teleportations": teleportation_count,
        "mapped_telegates": telegate_count,
        "mapped_depth": circuit_depth,
        "mapped_teleport_depth": teleport_depth,
        "solved_deadlocks": solved_deadlocks,
        "solving_deadlock": solving_deadlock,
        "mapping_time": mapping_time,
        "routing_time": None,
    }

def run_our_telesabre_pass(qc, arch, seed=None):
    map_start = time.perf_counter()
    init_mapping = telesabre_layout(arch, qc, seed=seed)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start
    
    route_start = time.perf_counter()
    routed_qc, _, log = telesabre_swap(arch, qc, init_mapping)
    route_end = time.perf_counter()
    routing_time = route_end - route_start
    
    # Count operations from log
    swap_count = 0
    teleportation_count = 0
    telegate_count = 0
    
    for entry in log:
        op_type = entry[0]
        if op_type == "SWAP":
            swap_count += 1
        elif op_type == "TELEDATA":
            teleportation_count += 1
        elif op_type == "TELEGATE_CX":
            telegate_count += 1
    
    # Calculate depths (including teleportation costs)
    num_qubits = len(arch)
    qubit_depths = [0] * num_qubits
    qubit_depths_tp = [0] * num_qubits
    
    for entry in log:
        op_type = entry[0]
        
        if op_type == "SWAP":
            phys_qubits = entry[1]
            max_depth = max(qubit_depths[p] for p in phys_qubits)
            duration = 1
            for p in phys_qubits:
                qubit_depths[p] = max_depth + duration
                
        elif op_type == "TELEDATA":
            phys_qubits = entry[1]  # (p_data_src, p_comm_src, p_comm_dst)
            max_depth = max(qubit_depths[p] for p in phys_qubits)
            duration = 5  # teleportation takes longer
            for p in phys_qubits:
                qubit_depths[p] = max_depth + duration
                qubit_depths_tp[p] = max_depth + duration
                
        elif op_type == "TELEPORT":
            # Skip TELEPORT as it's paired with TELEDATA
            pass
            
        elif op_type == "TELEGATE_CX":
            phys_qubits = entry[1]  # (p_data_ctrl, p_comm_ctrl, p_comm_tgt, p_data_tgt)
            max_depth = max(qubit_depths[p] for p in phys_qubits)
            duration = 5  # telegate takes longer
            for p in phys_qubits:
                qubit_depths[p] = max_depth + duration
                qubit_depths_tp[p] = max_depth + duration
                
        elif op_type == "TELEGATE":
            # Skip TELEGATE as it's paired with TELEGATE_CX
            pass
            
        elif op_type == "GATE":
            gate_info = entry[1]
            if isinstance(gate_info, tuple) and len(gate_info) >= 2:
                gate_type, phys_qubits = gate_info[0], gate_info[1]
                if isinstance(phys_qubits, tuple):
                    max_depth = max(qubit_depths[p] for p in phys_qubits if isinstance(p, int))
                    duration = 1
                    for p in phys_qubits:
                        if isinstance(p, int):
                            qubit_depths[p] = max_depth + duration
    
    circuit_depth = max(qubit_depths) if qubit_depths else 0
    teleport_depth = max(qubit_depths_tp) if qubit_depths_tp else 0
    
    # Validate without breaking benchmarking
    try:
        is_valid = validate_log(arch, log, require_pair_tags=True)
        valid_error = None
    except RuntimeError as e:
        is_valid = False
        valid_error = str(e)
    
    # Get routed circuit stats
    routed_ops = routed_qc.count_ops()
    routed_cx = routed_ops.get("cx", 0)
    routed_swaps_in_circuit = routed_ops.get("swap", 0)
    
    return {
    "seed": seed,
    "routed_swaps": swap_count,
    "routed_teleportations": teleportation_count,
    "routed_telegates": telegate_count,
    "routed_depth": circuit_depth,
    "routed_teleport_depth": teleport_depth,
    "routed_cx": routed_cx,
    "routed_size": routed_qc.size(),
    "mapping_time": mapping_time,
    "routing_time": routing_time,
    "is_valid": is_valid,
    "valid_error": valid_error,
    }


all_rows_og_telesabre = []
all_rows_our_telesabre = []
pairs = list(CIRCUITS.items())
it = tqdm(pairs, desc="Benchmarking (OG TELESABRE)", unit="run")

for cir_name, cir in it:
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    
    # OG TeleSABRE
    arch_og = build_og_arch(num_qubits)
    try:
        stats = run_og_telesabre_pass(init_cir, arch_og, seed=BASE_SEED)
        stats.update({
            "impl": "og_telesabre",
            "name": cir_name,
            "config_name": f"seed{BASE_SEED}",
            "arch_name": arch_og.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
        })
        all_rows_og_telesabre.append(stats)
    except Exception as e:
        all_rows_og_telesabre.append({
            "impl": "og_telesabre",
            "name": cir_name,
            "config_name": f"seed{BASE_SEED}",
            "arch_name": arch_og.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
            "error": repr(e),
        })

out_og_telesabre = save_stats_json(all_rows_og_telesabre, "./benchmarks/telesabre/results/og_telesabre.json", indent=4)
print(f"Saved {len(all_rows_og_telesabre)} rows to {out_og_telesabre}")


it = tqdm(pairs, desc="Benchmarking (OG TELESABRE)", unit="run")
for cir_name, cir in it:
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    
    # Our TeleSABRE
    arch_our = build_our_arch(num_qubits)
    try:
        stats = run_our_telesabre_pass(init_cir, arch_our, seed=BASE_SEED)
        stats.update({
            "impl": "our_telesabre",
            "name": cir_name,
            "config_name": f"seed{BASE_SEED}",
            "arch_name": arch_our.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
        })
        all_rows_our_telesabre.append(stats)
    except Exception as e:
        all_rows_our_telesabre.append({
            "impl": "our_telesabre",
            "name": cir_name,
            "config_name": f"seed{BASE_SEED}",
            "arch_name": arch_our.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
            "error": repr(e),
        })

out_our_telesabre = save_stats_json(all_rows_our_telesabre, "./benchmarks/telesabre/results/our_telesabre.json", indent=4)
print(f"Saved {len(all_rows_our_telesabre)} rows to {out_our_telesabre}")