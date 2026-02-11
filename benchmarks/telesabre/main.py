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

from architecture import two_tokyo, five_tokyo
from benchmarks.utils import load_qasm, init_circuit, save_stats_json

BASE_SEED = 1

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