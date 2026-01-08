from src.sabre import sabre
from src.architecture_graph import QubitNetworkGraph
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

qc_entangled = QuantumCircuit(2)
qc_entangled.h(0)
qc_entangled.cnot(0,1)
qc_entangled.h(1)

QFT_circuit = QuantumCircuit(4)
QFT_circuit.append(QFT(4).decompose(),range(4))

graph = QubitNetworkGraph([(0,1),(1,2),(2,3)], name="4-qubit line")

sabre(graph, QFT_circuit)