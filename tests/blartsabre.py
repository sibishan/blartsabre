from mapper.blartsabre import blartsabre
from blart_architecture import BLARTNetworkGraph, blart_grid
from qiskit import QuantumCircuit
from convert import from_qiskit
import qiskit.qasm2

qc = qiskit.qasm2.load("./data/20QBT_100CYC_QSE_8.qasm")

# arch = BLARTNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)], blart_edge_groups=[([1,3],[4,6])])
# arch = BLARTNetworkGraph([(0,1),(0,2),(1,3),(2,3),(0,8),(2,8),(4,5),(4,6),(5,7),(6,7),(5,9),(7,9)], blart_edge_groups=[([1,3],[4,6])])
arch = blart_grid(3,3,2,2)
arch.draw()

initial_mapping = blartsabre(arch, qc, verbose=True, num_iterations=10)

# routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)
# print(arch.distance_matrix)
# print(routed_qc)
# print(mapping)
# print(log)