from mapper.sabre import sabre
from router.sabre import sabre_swap
from architecture import QubitNetworkGraph
from qiskit import QuantumCircuit
from convert import from_qiskit
import qiskit.qasm2

qc = qiskit.qasm2.load("./data/example_4q/example_4q.qasm")

# num_rows = 2
# qc_entangled = QuantumCircuit(2 * num_rows, 2 * num_rows)
# for i in range(num_rows):
#     qc_entangled.h(i)
#     qc_entangled.cx(i, i + num_rows)
#     qc_entangled.measure(i, i)
#     qc_entangled.measure(i + num_rows, i + num_rows)

arch = QubitNetworkGraph([(0,1),(1,2),(2,3)], name="4-qubit line")

initial_mapping = sabre(arch, qc, verbose=True)

dag = from_qiskit(qc)

routed_qc, mapping, log = sabre_swap(arch, dag, initial_mapping)

print(routed_qc)
print(mapping)
print(log)