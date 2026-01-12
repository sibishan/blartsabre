from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeOsaka

from qiskit.transpiler import PassManager

# init stage
from qiskit.transpiler.passes import (
    UnitarySynthesis,
    HighLevelSynthesis,
    BasisTranslator,
    ElidePermutations,
    RemoveDiagonalGatesBeforeMeasure,
    RemoveIdentityEquivalent,
    InverseCancellation,
    ContractIdleWiresInControlFlow,
    CommutativeCancellation,
    ConsolidateBlocks,
    Split2QUnitaries
)
# for BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
basis_gates = ["rz", "sx", "x", "id", "cz"]

# stage 1
def compile_init(qc, opt_lvl):
    init_pm = PassManager()
    match opt_lvl:
        case 3:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                BasisTranslator(sel, basis_gates),
                ElidePermutations(),
                RemoveDiagonalGatesBeforeMeasure(),
                RemoveIdentityEquivalent(),
                InverseCancellation(),
                ContractIdleWiresInControlFlow(),
                CommutativeCancellation(),
                ConsolidateBlocks(),
                Split2QUnitaries()
            ])
        case 2:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                BasisTranslator(sel, basis_gates),
                ElidePermutations(),
                RemoveDiagonalGatesBeforeMeasure(),
                RemoveIdentityEquivalent(),
                InverseCancellation(),
                ContractIdleWiresInControlFlow(),
                CommutativeCancellation(),
                ConsolidateBlocks(),
                Split2QUnitaries()
            ])
        case 1:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                BasisTranslator(sel, basis_gates),
                InverseCancellation(),
                ContractIdleWiresInControlFlow(),

            ])
        case 0:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                BasisTranslator(sel, basis_gates)
            ])
        
    init_cir = init_pm.run(qc)
    return init_cir


# def compile():
#     pm = StagedPassManager()

#     # init stage
#     pm.init = UnitarySynthesis()
#     pm.init
    

# print("Plugins run by default init stage")
# print("=================================")

# backend = FakeTorino()

# for i in range(4):
#     print(f"\nOptimization level {i}:")
#     pm = generate_preset_pass_manager(backend=backend, optimization_level=i, init_method="default", seed_transpiler=1000)
#     for task in pm.init.to_flow_controller().tasks:
#         print(" -", type(task).__name__)

# # test
# from qiskit import QuantumCircuit
# num_rows = 3
# qc_entangled = QuantumCircuit(2 * num_rows, 2 * num_rows)
# for i in range(num_rows):
#     qc_entangled.h(i)
#     qc_entangled.cx(i, i + num_rows)
#     qc_entangled.measure(i, i)
#     qc_entangled.measure(i + num_rows, i + num_rows)
# from qiskit.converters import circuit_to_dag
# dag = circuit_to_dag(qc_entangled)

# print(compile_init(qc_entangled, opt_lvl=3))
