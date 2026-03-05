import time
from tqdm import tqdm

from mapper.telesabre import telesabre_layout
from architecture import multi_core_grid

from benchmarks.utils import load_qasm, init_circuit, save_stats_json, stats_from_our_telesabre_log, stats_from_blartsabre_log


BASE_SEED = 1

CIRCUITS = load_qasm("./data/telesabre", recursive=True)

def build_our_arch(num_qubits):
    if num_qubits <= 25:
        arch = multi_core_grid(3,3,2,2) # 36 qubits, 4 cores of 9 qubits each, with 2 inter-core connections
    else:
        arch = multi_core_grid(4,4,3,2) # 96 qubits, 4 cores of 16 qubits each, with 3 inter-core connections

    return arch

def run_our_telesabre_pass(qc, arch, seed=None):
    map_start = time.perf_counter()
    _, _, total_gates, swaps, teleports, telegates = telesabre_layout(arch, qc, seed=seed, return_log=True, verbose=True, num_iterations=5)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start
    
    return {
    "iterations": 5,
    "seed": seed,
    "total_gates": total_gates,
    "swaps": swaps,
    "teleports": teleports,
    "telegates": telegates,
    "mapping_time": mapping_time,
    }

all_rows_our_telesabre = []
pairs = list(CIRCUITS.items())
it = tqdm(pairs, desc="Benchmarking (OUR TELESABRE)", unit="run")
for cir_name, cir in it:
    print(cir_name)
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    
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

out_our_telesabre = save_stats_json(all_rows_our_telesabre, "./benchmarks/telesabre/results/our_telesabre_improved.json", indent=4)
print(f"Saved {len(all_rows_our_telesabre)} rows to {out_our_telesabre}")