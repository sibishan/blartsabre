from mapper.sabre import sabre
from architecture import QubitNetworkGraph
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

num_rows = 2
qc_entangled = QuantumCircuit(2 * num_rows, 2 * num_rows)
for i in range(num_rows):
    qc_entangled.h(i)
    qc_entangled.cx(i, i + num_rows)
    qc_entangled.measure(i, i)
    qc_entangled.measure(i + num_rows, i + num_rows)

graph = QubitNetworkGraph([(0,1),(1,2),(2,3)], name="4-qubit line")

sabre(graph, qc_entangled, verbose=True)