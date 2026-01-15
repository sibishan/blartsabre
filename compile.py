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
    Split2QUnitaries,
    Unroll3qOrMore,
    RemoveResetInZeroState,
    OptimizeSwapBeforeMeasure
)
# for BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
BASIS_GATES = ["rz", "sx", "x", "id", "cz"]

# layout stage
from mapper.sabre import sabre
from qiskit.transpiler import Layout

# routing stage
from router.sabre import sabre_swap

# stage 1
def compile_init(qc, opt_lvl=0):
    init_pm = PassManager()
    # match opt_lvl:
    #     case 3 | 2:
    #         init_pm.append([
    #             UnitarySynthesis(),
    #             HighLevelSynthesis(),
    #             ElidePermutations(),
    #             RemoveDiagonalGatesBeforeMeasure(),
    #             RemoveIdentityEquivalent(),
    #             InverseCancellation(),
    #             ContractIdleWiresInControlFlow(),
    #             CommutativeCancellation(),
    #             ConsolidateBlocks(),
    #             Split2QUnitaries(),
    #             # BasisTranslator(sel, BASIS_GATES)
    #         ])
    #     case 1:
    #         init_pm.append([
    #             UnitarySynthesis(),
    #             HighLevelSynthesis(),
    #             InverseCancellation(),
    #             ContractIdleWiresInControlFlow(),
    #             # BasisTranslator(sel, BASIS_GATES)
    #         ])
    #     case 0:
    #         init_pm.append([
    #             UnitarySynthesis(),
    #             HighLevelSynthesis(),
    #             # BasisTranslator(sel, BASIS_GATES)
    #         ])
    init_pm.append([
        Unroll3qOrMore(),
        RemoveResetInZeroState(),
        OptimizeSwapBeforeMeasure(),
        RemoveDiagonalGatesBeforeMeasure()
    ])
    init_cir = init_pm.run(qc)
    return init_cir

# stage 2
def compile_layout(arch, init_cir, seed = None, verbose = False):
    if seed is not None:
        import random
        random.seed(seed)
    
    initial_mapping = sabre(arch, init_cir, return_log=False, verbose=verbose)

    # #initial_mapping maps logical to physical
    # v2p = {init_cir.qubits[v]:  int(p) for v, p in initial_mapping.items()}
    # layout = Layout(v2p)

    return initial_mapping

# stage 3
def compile_route(arch, qc, initial_mapping):
    routed_qc, final_mapping, log = sabre_swap(arch, qc, initial_mapping)
    
    return routed_qc, final_mapping, log
    

# print("Plugins run by default stages")
# print("=================================")

# backend = FakeTorino()

# for i in range(4):
#     print(f"\nOptimization level {i}:")
#     pm = generate_preset_pass_manager(backend=backend, optimization_level=i, init_method="default", seed_transpiler=1000)
#     for task in pm.layout.to_flow_controller().tasks:
#         print(" -", type(task).__name__)

