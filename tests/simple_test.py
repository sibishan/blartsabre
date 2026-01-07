from sabre import sabre
from graph import QubitNetworkGraph

graph = QubitNetworkGraph([(0,1),(1,2),(2,3)], name="4-qubit line")

QFT_circuit = None # TODO: Add qiskit circuit

sabre(graph, QFT_circuit)