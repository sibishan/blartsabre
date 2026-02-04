import time
from itertools import product
from tqdm import tqdm

from qiskit import qasm2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes.layout import (
    SetLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout
)

from mapper.sabre import sabre_layout  # our sabre_layout implementation
from router.sabre import sabre_swap # our sabre_swap implementation
from .sabre_layout import SabreLayout  # qiskit sabre_layout implementation
from .sabre_swap import SabreSwap   # qiskit sabre_swap implementation
from qiskit.transpiler.passes import SabreLayout as LightSabreLayout # qiskit lightsabre_layout implementation
from qiskit.transpiler.passes.routing import SabreSwap as LightSabreSwap # qiskit lightsabre_swap implementation

from architecture import tokyo, rochester

from benchmarks.utils import load_qasm, init_circuit, save_stats_json


# def make_configs(heuristics=("basic", "lookahead", "decay"), iterations=(1, 3, 5), base_seed=1):
#     cfgs = {}
#     for h in heuristics:
#         for it in iterations:
#             cfgs[f"{h}_it{it}"] = {
#                 "layout_seed": base_seed,
#                 "max_iterations": it,
#                 "swap_heuristic": h,
#                 "swap_seed": base_seed,
#             }
#     return cfgs

# CONFIGS = make_configs(heuristics=("basic", "lookahead", "decay"), iterations=(3,))

# CIRCUITS = load_qasm("data/qasmbench", recursive=True)

qc = qasm2.load("./data/qasmbench/large/qft_n29/qft_n29.qasm")

config = {"layout_seed": 1,
            "max_iterations": 3,
            "swap_heuristic": "basic",
            "swap_seed": 1}

def our_sabre_pass(qc, arch, config):
    layout_seed = config["layout_seed"]

    map_start = time.perf_counter()
    init_mapping = sabre_layout(arch, qc, seed=layout_seed)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start

    route_start = time.perf_counter()
    routed_qc, _, _ = sabre_swap(arch, qc, init_mapping)
    route_end = time.perf_counter()
    routing_time = route_end - route_start

    routed_ops = routed_qc.count_ops()
    routed_cx = routed_ops.get("cx", 0)
    routed_swaps = routed_ops.get("swap", 0)

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
    }

arch = rochester()

init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(qc)
stats = our_sabre_pass(init_cir, arch, config)

print(stats)
