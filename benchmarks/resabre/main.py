import time
from itertools import product
from copy import deepcopy
from tqdm import tqdm

from mapper.resabre import resabre_layout
from router.resabre import resabre_swap

from architecture import two_tokyo, three_tokyo, twenty_qubit_star_line_ring
from benchmarks.utils import load_qasm, init_circuit, save_stats_json

BASE_SEED = 1

CIRCUITS = load_qasm("./data/queko", recursive=True)

def build_arch(num_qubits):
    if num_qubits > 60:
        raise ValueError("no arch available for more than 60 qubits")
    elif num_qubits > 40:
        arch = three_tokyo()
    elif num_qubits > 20:
        arch = two_tokyo()
    else:
        arch = twenty_qubit_star_line_ring()

    return arch


def validate_log(arch, log):
    for i, (op, payload) in enumerate(log):

        if op == "SWAP":
            u, v = payload
            if not arch.has_edge(u, v):
                raise RuntimeError(f"Illegal SWAP at step {i}: ({u},{v}) not an edge")
            continue

        if op != "GATE":
            continue

        gt, phys = payload
        gt = gt.upper()

        # normalise phys
        if isinstance(phys, int):
            phys = (phys,)
        elif not isinstance(phys, tuple):
            phys = tuple(phys)

        # Validate 2q connectivity for 2q ops only
        if gt in ("CX", "SWAP"):
            if len(phys) != 2:
                raise RuntimeError(f"{gt} at step {i} expected 2 qubits, got {phys}")
            u, v = phys
            if not arch.has_edge(u, v):
                raise RuntimeError(f"Illegal {gt} at step {i}: ({u},{v}) not an edge")

        # Validate 1q gates point to a real node
        elif len(phys) == 1:
            (u,) = phys
            if u not in arch:
                raise RuntimeError(f"Gate {gt} at step {i} uses invalid physical qubit {u}")

        # Barrier or other multi-qubit ops, just check nodes exist
        else:
            for u in phys:
                if u not in arch:
                    raise RuntimeError(f"Gate {gt} at step {i} uses invalid physical qubit {u}")

    return True

def count_swaps(log):
    return sum(1 for op, _ in log if op == "SWAP")

def count_comm_swaps(arch, log):
    if not hasattr(arch, "is_comm_edge"):
        return 0
    return sum(
        1 for op, (u, v) in log
        if op == "SWAP" and arch.is_comm_edge(u, v)
    )


def run_resabre_pass(qc, arch, config):
    layout_seed = config["layout_seed"]

    map_start = time.perf_counter()
    init_mapping = resabre_layout(arch, qc, seed=layout_seed)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start

    route_start = time.perf_counter()
    routed_qc, _, log = resabre_swap(arch, qc, init_mapping)
    route_end = time.perf_counter()
    routing_time = route_end - route_start

    routed_ops = routed_qc.count_ops()
    routed_cx = routed_ops.get("cx", 0)
    routed_swaps = routed_ops.get("swap", 0)

    # comm swaps from log (more reliable than counting in circuit because comm is an edge class)
    comm_swaps = count_comm_swaps(arch, log)

    # validate without breaking benchmarking
    try:
        is_valid = validate_log(arch, log)
        valid_error = None
    except RuntimeError as e:
        is_valid = False
        valid_error = str(e)

    return {
        "config": {
            "layout_seed": layout_seed,
            "max_iterations": 50,
            "swap_heuristic": {
                "EXTENDED_LAYER_SIZE": 10,
                "EXTENDED_HEURISTIC_WEIGHT": 0.5,
                "DECAY_VALUE": 0.001
            },
            "swap_seed": None,
        },
        "routed_swaps": routed_swaps,
        "comm_swaps": comm_swaps,
        "routed_cx": routed_cx,
        "routed_depth": routed_qc.depth(),
        "routed_size": routed_qc.size(),
        "mapping_time": mapping_time,
        "routing_time": routing_time,
        "is_valid": is_valid,
        "valid_error": valid_error,
    }


# RESABRE
all_rows_our = []
pairs_our = list(CIRCUITS.items())
it2 = tqdm(pairs_our, desc="Benchmarking (RESABRE)", unit="run")

for cir_name, cir in it2:
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    arch = build_arch(num_qubits)

    qc_our = deepcopy(init_cir)

    try:
        stats = run_resabre_pass(qc_our, arch, {"layout_seed": BASE_SEED})
        stats.update({
            "impl": "resabre",
            "name": cir_name,
            "config_name": f"seed{BASE_SEED}",
            "arch_name": arch.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
        })
        stats["total_time"] = stats["init_time"] + stats["mapping_time"] + stats["routing_time"]
        all_rows_our.append(stats)
    except Exception as e:
        all_rows_our.append({
            "impl": "resabre",
            "name": cir_name,
            "config_name": f"seed{BASE_SEED}",
            "arch_name": arch.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
            "error": repr(e),
        })

out_re = save_stats_json(all_rows_our, "./benchmarks/resabre/results/resabre.json", indent=4)

print(f"Saved {len(all_rows_our)} rows to {out_re}")