import time
import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure,
    Unroll3qOrMore,
    RemoveResetInZeroState,
    Decompose
)


def load_qasm(dir, pattern="*.qasm", recursive=False, sort=True, return_dict=True, show_progress=True):
    dir = Path(dir).expanduser().resolve()
    if not dir.exists() or not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir}")
    
    globber = dir.rglob if recursive else dir.glob
    files = list(globber(pattern))

    if sort:
        files.sort(key=lambda p: p.name)
    
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {dir}")

    loader = lambda path: QuantumCircuit.from_qasm_file(str(path))

    loaded = []
    errors = []

    iterable = files
    if show_progress:
        from tqdm import tqdm
        iterable = tqdm(files, desc="Loading QASM", unit="file")

    total = len(files)
    for idx, path in enumerate(iterable, start=1):
        try:
            loaded.append((path.stem, loader(path)))
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}: {e}")

    if errors:
        preview = "\n".join(errors[:10])
        more = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more"
        raise RuntimeError(f"Some QASM files failed to load:\n{preview}{more}")
    
    if return_dict:
        return {name: qc for name, qc in loaded}
    return [qc for _, qc in loaded]


def get_non_single_qg_names(non_single_qg):
    """From the provided non-single-qubit gates, get all of the gate names"""
    non_single_qg_names = set()
    for gate in non_single_qg:
        if gate.name == "cx":
            continue
        non_single_qg_names.add(gate.name)
    return list(non_single_qg_names)

def init_circuit(qc, verbose=False):
    dag = circuit_to_dag(qc)
    two_qg_list = dag.two_qubit_ops()
    mul_qg_list = dag.multi_qubit_ops()
    non_single_qg_names = get_non_single_qg_names(two_qg_list + mul_qg_list)
    if verbose:
        if non_single_qg_names:
            print(f"GATES TO DECOMPOSE: {non_single_qg_names}")
        else:
            print("NO GATES TO DECOMPOSE")

    init_pm = PassManager()
    init_pm.append([
        Unroll3qOrMore(),
        RemoveResetInZeroState(),
        OptimizeSwapBeforeMeasure(),
        RemoveDiagonalGatesBeforeMeasure()
    ])

    # if there are non-single-qubit gates which are not CNOTs, decompose those gates before routing
    if non_single_qg_names:
        init_pm.append(Decompose(non_single_qg_names))

    init_start = time.perf_counter()
    init_cir = init_pm.run(qc)
    init_end = time.perf_counter()
    init_time = init_end - init_start
    
    og_ops = init_cir.count_ops()
    og_cx = og_ops.get("cx", 0)
    og_swaps = og_ops.get("swap", 0)
    og_depth = init_cir.depth()
    num_qubits = init_cir.num_qubits
    og_size = init_cir.size()

    return init_cir, init_time, og_cx, og_swaps, og_depth, num_qubits, og_size

def save_stats_json(all_rows, out_path, indent=2):
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=indent)

    return out_path