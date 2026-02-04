import time
from itertools import product
from tqdm import tqdm

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


def make_configs(heuristics=("basic", "lookahead", "decay"), iterations=(1, 3, 5), base_seed=1):
    cfgs = {}
    for h in heuristics:
        for it in iterations:
            cfgs[f"{h}_it{it}"] = {
                "layout_seed": base_seed,
                "max_iterations": it,
                "swap_heuristic": h,
                "swap_seed": base_seed,
            }
    return cfgs

CONFIGS = make_configs(heuristics=("basic", "lookahead", "decay"), iterations=(3,))

CIRCUITS = load_qasm("data/qasmbench", recursive=True)

def run_sabre_pass(
    qc,
    coupling_map,
    config,
    *,
    LayoutPassCls,
    SwapPassCls,
):
    """
    Run: (1) SabreLayout/LightSabreLayout to choose a layout, then
         (2) ApplyLayout + SabreSwap/LightSabreSwap to route,
    returning timing + routed circuit stats.

    LayoutPassCls: SabreLayout or qiskit.transpiler.passes.SabreLayout (Light)
    SwapPassCls:   your SabreSwap or qiskit.transpiler.passes.SabreSwap (Light)
    """
    layout_seed = config["layout_seed"]
    max_iterations = config["max_iterations"]
    swap_heuristic = config["swap_heuristic"]
    swap_seed = config["swap_seed"]

    # Pass instance used by the layout stage for scoring
    routing_pass_for_layout = SwapPassCls(
        coupling_map,
        heuristic=swap_heuristic,
        seed=swap_seed,
    )

    pm_init_layout = PassManager([
        LayoutPassCls(
            coupling_map,
            routing_pass=routing_pass_for_layout,
            seed=layout_seed,
            max_iterations=max_iterations,
        )
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
        SwapPassCls(coupling_map, heuristic=swap_heuristic, seed=swap_seed),
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
        "routing_time": routing_time,
    }

def sabre_pass(qc, coupling_map, config):
    return run_sabre_pass(
        qc,
        coupling_map,
        config,
        LayoutPassCls=SabreLayout,
        SwapPassCls=SabreSwap,
    )

def lightsabre_pass(qc, coupling_map, config):
    return run_sabre_pass(
        qc,
        coupling_map,
        config,
        LayoutPassCls=LightSabreLayout,
        SwapPassCls=LightSabreSwap,
    )


all_rows_sabre = []
all_rows_lightsabre = []

pairs = list(product(CIRCUITS.items(), CONFIGS.items()))
it = tqdm(pairs, desc="Benchmarking", unit="run") if tqdm else pairs

total = len(pairs)
for idx, ((cir_name, cir), (config_name, config)) in enumerate(it, start=1):
    init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size = init_circuit(cir)

    if num_qubits > 53:
        raise ValueError("no arch available for more than 53 qubits")
    elif num_qubits > 20:
        arch = rochester()
        cm = CouplingMap(couplinglist=arch.edges())
    else:
        arch = tokyo()
        cm = CouplingMap(couplinglist=arch.edges())
    cm.make_symmetric()

    # OG SABRE
    try:
        stats_sabre = sabre_pass(init_cir, cm, config)
        stats_sabre["impl"] = "sabre"
        stats_sabre["init_time"] = init_time
        stats_sabre["og_cx"] = og_cx
        stats_sabre["og_swaps"] = og_swaps
        stats_sabre["og_depth"] = og_depth
        stats_sabre["num_qubits"] = num_qubits
        stats_sabre["og_size"] = og_size
        stats_sabre["total_time"] = (
            stats_sabre["init_time"] + stats_sabre["mapping_time"] + stats_sabre["routing_time"]
        )
        stats_sabre["name"] = cir_name
        stats_sabre["config_name"] = config_name
        stats_sabre["arch_name"] = arch.name

        all_rows_sabre.append(stats_sabre)
    except Exception as e:
        # keep going, but record the failure
        all_rows_sabre.append({
            "impl": "sabre",
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

    # LIGHTSABRE
    try:
        stats_light = lightsabre_pass(init_cir, cm, config)
        stats_light["impl"] = "light_sabre"
        stats_light["init_time"] = init_time
        stats_light["og_cx"] = og_cx
        stats_light["og_swaps"] = og_swaps
        stats_light["og_depth"] = og_depth
        stats_light["num_qubits"] = num_qubits
        stats_light["og_size"] = og_size
        stats_light["total_time"] = (
            stats_light["init_time"] + stats_light["mapping_time"] + stats_light["routing_time"]
        )
        stats_light["name"] = cir_name
        stats_light["config_name"] = config_name
        stats_light["arch_name"] = arch.name

        all_rows_lightsabre.append(stats_light)
    except Exception as e:
        all_rows_lightsabre.append({
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

# Save outputs
out_sabre = save_stats_json(all_rows_sabre, "./benchmarks/og_sabre/sabre.json", indent=4)
out_light = save_stats_json(all_rows_lightsabre, "./benchmarks/og_sabre/light_sabre.json", indent=4)

print(f"Saved {len(all_rows_sabre)} rows to {out_sabre}")
print(f"Saved {len(all_rows_lightsabre)} rows to {out_light}")

