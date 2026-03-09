from mapper.sabre import sabre_layout
from router.sabre import sabre_swap
from architecture import QubitNetworkGraph, tokyo
from qiskit import QuantumCircuit
from convert import from_qiskit

qc = QuantumCircuit.from_qasm_file("./data/example_4q/example_4q.qasm")
# qc = QuantumCircuit.from_qasm_file("./data/quekno/20Q_depth_Tokyo/20QBT_depth_Tokyo_large_opt_1_1.5_no.0.qasm")

# num_rows = 2
# qc_entangled = QuantumCircuit(2 * num_rows, 2 * num_rows)
# for i in range(num_rows):
#     qc_entangled.h(i)
#     qc_entangled.cx(i, i + num_rows)
#     qc_entangled.measure(i, i)
#     qc_entangled.measure(i + num_rows, i + num_rows)

arch = QubitNetworkGraph([(0,1),(1,2),(2,3),(3,4)], name="5-qubit line")
# arch = tokyo()

initial_mapping = sabre_layout(arch, qc, verbose=True)

routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)
print(arch.distance_matrix)
print(routed_qc)
print(mapping)
print(log)