from mapper.blartsabre import blartsabre_layout
from blart_architecture import BLARTNetworkGraph, blart_grid, blart_two_tokyo, blart_four_tokyo
from qiskit import QuantumCircuit
from convert import from_qiskit
import qiskit.qasm2

qc = qiskit.qasm2.load("./data/quekno/53Q_depth_Rochester/53QBT_depth_Rochester_large_opt_1_1.5_no.0.qasm")

# arch = BLARTNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)], blart_edge_groups=[([1,3],[4,6])])
# arch = BLARTNetworkGraph([(0,1),(0,2),(1,3),(2,3),(0,8),(2,8),(4,5),(4,6),(5,7),(6,7),(5,9),(7,9)], blart_edge_groups=[([1,3],[4,6])])
arch = blart_grid(4,4,3,2)
arch.draw()

# initial_mapping = blartsabre_layout(arch, qc, verbose=True, num_iterations=10)

# routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)
# print(arch.distance_matrix)
# print(routed_qc)
# print(mapping)
# print(log)