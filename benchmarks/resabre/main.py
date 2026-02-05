import time
from itertools import product
from copy import deepcopy
from tqdm import tqdm

from mapper.resabre import sabre_layout
from router.resabre import sabre_swap

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

        # normalise phys to tuple
        if not isinstance(phys, tuple):
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


def run_our_sabre_pass(qc, arch, config):
    layout_seed = config["layout_seed"]

    map_start = time.perf_counter()
    init_mapping = sabre_layout(arch, qc, seed=layout_seed)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start

    route_start = time.perf_counter()
    routed_qc, _, log = sabre_swap(arch, qc, init_mapping)
    route_end = time.perf_counter()
    routing_time = route_end - route_start

    routed_ops = routed_qc.count_ops()
    routed_cx = routed_ops.get("cx", 0)
    routed_swaps = routed_ops.get("swap", 0)

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
            "swap_heuristic": {"EXTENDED_LAYER_SIZE": 10,
                               "EXTENDED_HEURISTIC_WEIGHT": 0.5,
                               "DECAY_VALUE": 0.001},
            "swap_seed": None,
        },
        "routed_swaps": routed_swaps,
        "routed_cx": routed_cx,
        "routed_depth": routed_qc.depth(),
        "routed_size": routed_qc.size(),
        "mapping_time": mapping_time,
        "routing_time":  routing_time,
        "is_valid": is_valid,
        "valid_error": valid_error,
    }


# OG SABRE and Light SABRE
all_rows_og = []
all_rows_light = []

pairs_qiskit = list(product(CIRCUITS.items(), CONFIGS.items()))
it = tqdm(pairs_qiskit, desc="Benchmarking (OG + Light)", unit="run")

for (cir_name, cir), (config_name, config) in it:
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    arch, cm = build_arch(num_qubits)

    qc_og = deepcopy(init_cir)
    qc_light = deepcopy(init_cir)

    # OG
    try:
        stats = sabre_pass(qc_og, cm, config)
        stats.update({
            "impl": "og_sabre",
            "name": cir_name,
            "config_name": config_name,
            "arch_name": arch.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
        })
        stats["total_time"] = stats["init_time"] + stats["mapping_time"] + stats["routing_time"]
        all_rows_og.append(stats)
    except Exception as e:
        all_rows_og.append({
            "impl": "og_sabre",
            "name": cir_name,
            "config_name": config_name,
            "arch_name": arch.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
            "error": repr(e),
        })

    # Light
    try:
        stats = lightsabre_pass(qc_light, cm, config)
        stats.update({
            "impl": "light_sabre",
            "name": cir_name,
            "config_name": config_name,
            "arch_name": arch.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
        })
        stats["total_time"] = stats["init_time"] + stats["mapping_time"] + stats["routing_time"]
        all_rows_light.append(stats)
    except Exception as e:
        all_rows_light.append({
            "impl": "light_sabre",
            "name": cir_name,
            "config_name": config_name,
            "arch_name": arch.name,
            "num_qubits": num_qubits,
            "init_time": init_time,
            "og_cx": og_cx,
            "og_swaps": og_swaps,
            "og_depth": og_depth,
            "og_size": og_size,
            "error": repr(e),
        })


# OUR SABRE
all_rows_our = []
pairs_our = list(CIRCUITS.items())
it2 = tqdm(pairs_our, desc="Benchmarking (Our SABRE)", unit="run")

for cir_name, cir in it2:
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    arch, _cm = build_arch(num_qubits)

    qc_our = deepcopy(init_cir)

    try:
        stats = run_our_sabre_pass(qc_our, arch, {"layout_seed": BASE_SEED})
        stats.update({
            "impl": "our_sabre",
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
            "impl": "our_sabre",
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

out_og = save_stats_json(all_rows_og, "./benchmarks/sabre/results/og_sabre.json", indent=4)
out_light = save_stats_json(all_rows_light, "./benchmarks/sabre/results/light_sabre.json", indent=4)
out_our = save_stats_json(all_rows_our, "./benchmarks/sabre/results/our_sabre.json", indent=4)

print(f"Saved {len(all_rows_og)} rows to {out_og}")
print(f"Saved {len(all_rows_light)} rows to {out_light}")
print(f"Saved {len(all_rows_our)} rows to {out_our}")