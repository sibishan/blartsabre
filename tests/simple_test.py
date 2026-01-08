from src.sabre import sabre
from src.graph import QubitNetworkGraph
from qiskit import QuantumCircuit

qc_entangled = QuantumCircuit(2)
qc_entangled.h(0)
qc_entangled.cnot(0,1)
qc_entangled.h(1)
qc_entangled.draw(fold=-1)

graph = QubitNetworkGraph([(0,1),(1,2),(2,3)], name="4-qubit line")

QFT_circuit = qc_entangled # TODO: Add qiskit circuit

sabre(graph, QFT_circuit)