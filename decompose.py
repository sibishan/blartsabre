from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure,
    Unroll3qOrMore,
    RemoveResetInZeroState,
    Decompose
)

def get_non_single_qg_names(non_single_qg):
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

    init_cir = init_pm.run(qc)
    
    return init_cir