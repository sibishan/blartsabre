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

qc = qasm2.load("./data/qasmbench/medium/bv_n19/bv_n19.qasm")
config = {"layout_seed": 1,
            "max_iterations": 3,
            "swap_heuristic": "basic",
            "swap_seed": 1}

def sabre_pass(qc, coupling_map, config):
    layout_seed = config['layout_seed']
    max_iterations = config['max_iterations']
    swap_heuristic = config['swap_heuristic']
    swap_seed = config['swap_seed']

    routing_pass_for_layout = SabreSwap(coupling_map, heuristic=swap_heuristic, seed=swap_seed)

    pm_init_layout = PassManager([
        SabreLayout(
            coupling_map, 
            routing_pass=routing_pass_for_layout, 
            seed=layout_seed, 
            max_iterations=max_iterations)
        ])
    
    map_start = time.perf_counter()
    _ = pm_init_layout.run(qc)
    map_end = time.perf_counter()
    mapping_time = map_end - map_start
    chosen_layout = pm_init_layout.property_set["layout"]

    pm_route = PassManager([
        SetLayout(chosen_layout),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(coupling_map, heuristic=swap_heuristic, seed=swap_seed)
    ])
    route_start = time.perf_counter()
    routed = pm_route.run(qc)
    route_end = time.perf_counter()
    routing_time = route_end - route_start

    routed_ops = routed.count_ops()
    routed_cx = routed_ops.get("cx", 0)
    routed_swaps = routed_ops.get("swap", 0)

    return {
        "config": {
            "layout_seed": layout_seed,
            "max_iterations": max_iterations,
            "swap_heuristic": swap_heuristic,
            "swap_seed": swap_seed,
        },
        "routed_swaps": routed_swaps,
        "routed_cx": routed_cx,
        "routed_depth": routed.depth(),
        "routed_size": routed.size(),
        "mapping_time": mapping_time,
        "routing_time":  routing_time,
    }

arch = rochester()
cm = CouplingMap(couplinglist=arch.edges())
cm.make_symmetric()

init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(qc)
stats = sabre_pass(init_cir, cm, config)

print(stats)
