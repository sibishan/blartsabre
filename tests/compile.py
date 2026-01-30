from qiskit import QuantumCircuit
from compile import compile_init, compile_layout, compile_route
from legacy.architecture import QubitNetworkGraph

# num_rows = 2
# qc_entangled = QuantumCircuit(2 * num_rows, 2 * num_rows)
# for i in range(num_rows):
#     qc_entangled.h(i)
#     qc_entangled.cx(i, i + num_rows)
#     qc_entangled.measure(i, i)
#     qc_entangled.measure(i + num_rows, i + num_rows)

import qiskit.qasm2

qc = qiskit.qasm2.load("./data/example_4q/example_4q.qasm")

init_cir = compile_init(qc)

arch = QubitNetworkGraph([(0,1),(1,2),(2,3)], name="4-qubit line")
initial_mapping = compile_layout(arch, init_cir, verbose=True)

routed_qc, final_mapping, log = compile_route(arch, init_cir, initial_mapping)
print(routed_qc)
print(final_mapping)
