import time
from tqdm import tqdm

# our blartsabre
from mapper.blartsabre import blartsabre_layout

from blart_architecture import blart_grid
from benchmarks.utils import load_qasm, init_circuit, save_stats_json


BASE_SEED = 1

CIRCUITS = load_qasm("./data/telesabre", recursive=True)

def build_blart_arch(num_qubits):
    if num_qubits <= 25:
        arch = blart_grid(4,3,2,2)
    else:
        arch = blart_grid(4,4,3,2)

    return arch

def run_blartsabre_pass(qc, arch, seed=None):
    map_start = time.perf_counter()
    _, _, total_gates, local_swaps, remote_swaps, tele_gates = blartsabre_layout(arch, qc, seed=seed, num_iterations=5, return_log=True, verbose=True)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start
    
    return {
    "iterations": 5,
    "seed": seed,
    "total_gates": total_gates,
    "local_swaps": local_swaps,
    "remote_swaps": remote_swaps,
    "telegates": tele_gates,
    "mapping_time": mapping_time
    }


all_rows_blartsabre = []
pairs = list(CIRCUITS.items())
it = tqdm(pairs, desc="Benchmarking (BLARTSABRE)", unit="run")
for cir_name, cir in it:
    print(cir_name)
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    
    # BLARTSABRE
    arch_our = build_blart_arch(num_qubits)
    try:
        stats = run_blartsabre_pass(init_cir, arch_our, seed=BASE_SEED)
        stats.update({
            "impl": "blartsabre",
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
        all_rows_blartsabre.append(stats)
    except Exception as e:
        all_rows_blartsabre.append({
            "impl": "blartsabre",
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

out_blartsabre = save_stats_json(all_rows_blartsabre, "./benchmarks/telesabre/results/blartsabre_improved.json", indent=4)
print(f"Saved {len(all_rows_blartsabre)} rows to {out_blartsabre}")
