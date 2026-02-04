import time
from itertools import product
from copy import deepcopy
from tqdm import tqdm

from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes.layout import (
    SetLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout
)

from mapper.sabre import sabre_layout
from router.sabre import sabre_swap
from .sabre_layout import SabreLayout
from .sabre_swap import SabreSwap
from qiskit.transpiler.passes import SabreLayout as LightSabreLayout
from qiskit.transpiler.passes.routing import SabreSwap as LightSabreSwap

from architecture import tokyo, sycamore
from benchmarks.utils import load_qasm, init_circuit, save_stats_json


def make_configs(heuristics=("basic", "lookahead", "decay"), iterations=(50,), base_seed=1):
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

BASE_SEED = 1
CONFIGS = make_configs(heuristics=("basic", "lookahead", "decay"), iterations=(50,), base_seed=BASE_SEED)

CIRCUITS = load_qasm("./data/queko", recursive=True)

def build_arch(num_qubits):
    if num_qubits > 54:
        raise ValueError("no arch available for more than 54 qubits")
    if num_qubits > 20:
        arch = sycamore()
    else:
        arch = tokyo()

    cm = CouplingMap(couplinglist=arch.edges())
    cm.make_symmetric()
    return arch, cm


def run_qiskit_sabre_pass(qc, coupling_map, config, *, LayoutPassCls, SwapPassCls):
    layout_seed = config["layout_seed"]
    max_iterations = config["max_iterations"]
    swap_heuristic = config["swap_heuristic"]
    swap_seed = config["swap_seed"]

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

    t0 = time.perf_counter()
    _ = pm_init_layout.run(qc)
    mapping_time = time.perf_counter() - t0

    chosen_layout = pm_init_layout.property_set["layout"]

    pm_route = PassManager([
        SetLayout(chosen_layout),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SwapPassCls(coupling_map, heuristic=swap_heuristic, seed=swap_seed),
    ])

    t1 = time.perf_counter()
    routed = pm_route.run(qc)
    routing_time = time.perf_counter() - t1

    ops = routed.count_ops()
    return {
        "config": {
            "layout_seed": layout_seed,
            "max_iterations": max_iterations,
            "swap_heuristic": swap_heuristic,
            "swap_seed": swap_seed,
        },
        "routed_swaps": ops.get("swap", 0),
        "routed_cx": ops.get("cx", 0),
        "routed_depth": routed.depth(),
        "routed_size": routed.size(),
        "mapping_time": mapping_time,
        "routing_time": routing_time,
    }


def sabre_pass(qc, coupling_map, config):
    return run_qiskit_sabre_pass(qc, coupling_map, config, LayoutPassCls=SabreLayout, SwapPassCls=SabreSwap)


def lightsabre_pass(qc, coupling_map, config):
    return run_qiskit_sabre_pass(qc, coupling_map, config, LayoutPassCls=LightSabreLayout, SwapPassCls=LightSabreSwap)


def run_our_sabre_pass(qc, arch, seed):
    t0 = time.perf_counter()
    init_mapping = sabre_layout(arch, qc, seed=seed)
    mapping_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    routed_qc, _, _ = sabre_swap(arch, qc, init_mapping)
    routing_time = time.perf_counter() - t1

    ops = routed_qc.count_ops()
    return {
        "config": {
            "layout_seed": seed,
            "max_iterations": None,
            "swap_heuristic": {
                "EXTENDED_LAYER_SIZE": 10,
                "EXTENDED_HEURISTIC_WEIGHT": 0.5,
                "DECAY_VALUE": 0.001,
            },
            "swap_seed": None,
        },
        "routed_swaps": ops.get("swap", 0),
        "routed_cx": ops.get("cx", 0),
        "routed_depth": routed_qc.depth(),
        "routed_size": routed_qc.size(),
        "mapping_time": mapping_time,
        "routing_time": routing_time,
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
        stats = run_our_sabre_pass(qc_our, arch, BASE_SEED)
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