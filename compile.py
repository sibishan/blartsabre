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

# layout & routing stage
from mapper.sabre import sabre
from qiskit.transpiler import Layout

# stage 1
def compile_init(qc, opt_lvl):
    init_pm = PassManager()
    match opt_lvl:
        case 3 | 2:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                ElidePermutations(),
                RemoveDiagonalGatesBeforeMeasure(),
                RemoveIdentityEquivalent(),
                InverseCancellation(),
                ContractIdleWiresInControlFlow(),
                CommutativeCancellation(),
                ConsolidateBlocks(),
                Split2QUnitaries(),
                BasisTranslator(sel, basis_gates)
            ])
        case 1:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                InverseCancellation(),
                ContractIdleWiresInControlFlow(),
                BasisTranslator(sel, basis_gates)
            ])
        case 0:
            init_pm.append([
                UnitarySynthesis(),
                HighLevelSynthesis(),
                BasisTranslator(sel, basis_gates)
            ])
        
    init_cir = init_pm.run(qc)
    return init_cir

# stage 2
def compile_layout(arch, init_cir, seed = None, verbose = False):
    if seed is not None:
        import random
        random.seed(seed)
    
    initial_mapping = sabre(arch, init_cir, return_log=False, verbose=verbose)

    #initial_mapping maps logical to physical
    v2p = {init_cir.qubits[v]:  int(p) for v, p in initial_mapping.items()}
    layout = Layout(v2p)

    return layout, initial_mapping
    

# print("Plugins run by default init stage")
# print("=================================")

# backend = FakeTorino()

# for i in range(4):
#     print(f"\nOptimization level {i}:")
#     pm = generate_preset_pass_manager(backend=backend, optimization_level=i, init_method="default", seed_transpiler=1000)
#     for task in pm.init.to_flow_controller().tasks:
#         print(" -", type(task).__name__)

