import time
from tqdm import tqdm

from .architecture import TWO_TOKYO, FOUR_TOKYO
from .circuit import Circuit

# og telesabre
from .config import Config
from .telesabre import run_telesabre as og_telesabre_layout
config = Config()

# our telesabre
from mapper.telesabre import telesabre_layout

# our blartsabre
from mapper.blartsabre import blartsabre_layout

from architecture import two_tokyo, four_tokyo
from blart_architecture import blart_two_tokyo, blart_four_tokyo
from benchmarks.utils import load_qasm, init_circuit, save_stats_json, stats_from_our_telesabre_log, stats_from_blartsabre_log

from qiskit import QuantumCircuit

BASE_SEED = 1

CIRCUITS = load_qasm("./data/telesabre", recursive=True)
qc = QuantumCircuit.from_qasm_file("./data/telesabre/qasm_64/ghz_nativegates_ibm_qiskit_opt3_64.qasm")

def build_blart_arch(num_qubits):
    if num_qubits <= 20:
        arch = blart_two_tokyo()
    else:
        arch = blart_four_tokyo()

    return arch

def build_our_arch(num_qubits):
    if num_qubits <= 20:
        arch = two_tokyo()
    else:
        arch = four_tokyo()

    return arch

def build_og_arch(num_qubits):
    if num_qubits <= 20:
        arch = TWO_TOKYO()
    else:
        arch = FOUR_TOKYO()

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
        "iterations": 1,
        "seed": seed,
        "mapped_swaps": swap_count,
        "mapped_teleports": teleportation_count,
        "mapped_telegates": telegate_count,
        "mapped_depth": circuit_depth,
        "mapped_teleport_depth": teleport_depth,
        "solved_deadlocks": solved_deadlocks,
        "solving_deadlock": solving_deadlock,
        "mapping_time": mapping_time
    }

def run_our_telesabre_pass(qc, arch, seed=None):
    map_start = time.perf_counter()
    _, gate_log = telesabre_layout(arch, qc, seed=seed, return_log=True, verbose=True, num_iterations=5)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start

    metrics = stats_from_our_telesabre_log(gate_log, len(arch))
    
    return {
    "iterations": 5,
    "seed": seed,
    **metrics,
    "mapping_time": mapping_time,
    "gate_log": gate_log
    }

def run_blartsabre_pass(qc, arch, seed=None):
    map_start = time.perf_counter()
    _, gate_log = blartsabre_layout(arch, qc, seed=seed, num_iterations=5, return_log=True, verbose=True)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start

    metrics = stats_from_blartsabre_log(gate_log, len(arch))
    
    return {
    "iterations": 5,
    "seed": seed,
    **metrics,
    "mapping_time": mapping_time,
    "gate_log": gate_log
    }


all_rows_og_telesabre = []
all_rows_our_telesabre = []
all_rows_blartsabre = []
pairs = list(CIRCUITS.items())
# it = tqdm(pairs, desc="Benchmarking (OG TELESABRE)", unit="run")

# for cir_name, cir in it:
#     init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    
#     # OG TeleSABRE
#     arch_og = build_og_arch(num_qubits)
#     try:
#         stats = run_og_telesabre_pass(init_cir, arch_og, seed=BASE_SEED)
#         stats.update({
#             "impl": "og_telesabre",
#             "name": cir_name,
#             "config_name": f"seed{BASE_SEED}",
#             "arch_name": arch_og.name,
#             "num_qubits": num_qubits,
#             "init_time": init_time,
#             "og_cx": og_cx,
#             "og_swaps": og_swaps,
#             "og_depth": og_depth,
#             "og_size": og_size,
#         })
#         all_rows_og_telesabre.append(stats)
#     except Exception as e:
#         all_rows_og_telesabre.append({
#             "impl": "og_telesabre",
#             "name": cir_name,
#             "config_name": f"seed{BASE_SEED}",
#             "arch_name": arch_og.name,
#             "num_qubits": num_qubits,
#             "init_time": init_time,
#             "og_cx": og_cx,
#             "og_swaps": og_swaps,
#             "og_depth": og_depth,
#             "og_size": og_size,
#             "error": repr(e),
#         })

# out_og_telesabre = save_stats_json(all_rows_og_telesabre, "./benchmarks/telesabre/results/og_telesabre.json", indent=4)
# print(f"Saved {len(all_rows_og_telesabre)} rows to {out_og_telesabre}")


# it = tqdm(pairs, desc="Benchmarking (OUR TELESABRE)", unit="run")
# for cir_name, cir in it:
#     init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)
    
#     # Our TeleSABRE
#     arch_our = build_our_arch(num_qubits)
#     try:
#         stats = run_our_telesabre_pass(init_cir, arch_our, seed=BASE_SEED)
#         stats.update({
#             "impl": "our_telesabre",
#             "name": cir_name,
#             "config_name": f"seed{BASE_SEED}",
#             "arch_name": arch_our.name,
#             "num_qubits": num_qubits,
#             "init_time": init_time,
#             "og_cx": og_cx,
#             "og_swaps": og_swaps,
#             "og_depth": og_depth,
#             "og_size": og_size,
#         })
#         all_rows_our_telesabre.append(stats)
#     except Exception as e:
#         all_rows_our_telesabre.append({
#             "impl": "our_telesabre",
#             "name": cir_name,
#             "config_name": f"seed{BASE_SEED}",
#             "arch_name": arch_our.name,
#             "num_qubits": num_qubits,
#             "init_time": init_time,
#             "og_cx": og_cx,
#             "og_swaps": og_swaps,
#             "og_depth": og_depth,
#             "og_size": og_size,
#             "error": repr(e),
#         })

# out_our_telesabre = save_stats_json(all_rows_our_telesabre, "./benchmarks/telesabre/results/our_telesabre.json", indent=4)
# print(f"Saved {len(all_rows_our_telesabre)} rows to {out_our_telesabre}")


it = tqdm(pairs, desc="Benchmarking (BLARTSABRE)", unit="run")
for cir_name, cir in it:
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

out_blartsabre = save_stats_json(all_rows_blartsabre, "./benchmarks/telesabre/results/blartsabre.json", indent=4)
print(f"Saved {len(all_rows_blartsabre)} rows to {out_blartsabre}")
