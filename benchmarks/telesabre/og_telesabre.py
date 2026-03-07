import time
from tqdm import tqdm

from .architecture import Architecture, Edge
from .circuit import Circuit

# og telesabre
from .config import Config
from .telesabre import run_telesabre as og_telesabre_layout
config = Config()


from benchmarks.utils import load_qasm, init_circuit, save_stats_json

from qiskit import QuantumCircuit

BASE_SEED = 1

CIRCUITS = load_qasm("./data/telesabre", recursive=True)

def build_og_arch(num_qubits):
    if num_qubits <= 25:
        arch = Architecture(4,3,2,2)
        arch.inter_core_edges = [
            Edge(5,15),
            Edge(22,37),
            Edge(32,42),
            Edge(10,25),
        ]
        
        arch._update_qubit_to_edges()
        arch._build_teleport_edges()
        
        arch.communication_qubits = list(set(arch.communication_qubits))
        
        arch.core_comm_qubits = [[] for _ in range(arch.num_cores)]
        for p in arch.communication_qubits:
            arch.core_comm_qubits[arch.qubit_to_core[p]].append(p)
            
        arch.core_qubits = [[] for _ in range(arch.num_cores)]
        for p in range(arch.num_qubits):
            arch.core_qubits[arch.qubit_to_core[p]].append(p)
        
        arch.name = "2x2C 4x3Q"
    else:
        arch = Architecture.H()

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
        "swaps": swap_count,
        "teleports": teleportation_count,
        "telegates": telegate_count,
        "depth": circuit_depth,
        "teleport_depth": teleport_depth,
        "solved_deadlocks": solved_deadlocks,
        "solving_deadlock": solving_deadlock,
        "mapping_time": mapping_time
    }


all_rows_og_telesabre = []
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

out_og_telesabre = save_stats_json(all_rows_og_telesabre, "./benchmarks/telesabre/results/og_telesabre_opt_true.json", indent=4)
print(f"Saved {len(all_rows_og_telesabre)} rows to {out_og_telesabre}")
