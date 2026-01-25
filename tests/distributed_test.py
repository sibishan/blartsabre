from mapper.telesabre import telesabre
from router.sabre import sabre_swap
from architecture import DistributedQubitNetworkGraph, QubitNetworkGraph, tokyo_arch
from qiskit import QuantumCircuit
from convert import from_qiskit
import qiskit.qasm2

# qc = qiskit.qasm2.load("./data/quekno/20Q_depth_Tokyo/20QBT_depth_Tokyo_large_opt_1_1.5_no.0.qasm")

num_rows = 4
qc = QuantumCircuit(2 * num_rows, 2 * num_rows)
for i in range(num_rows):
    qc.h(i)
    qc.cx(i, i + num_rows)
    qc.measure(i, i)
    qc.measure(i + num_rows, i + num_rows)

arch = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),
                                                         (8,9),(8,10),(9,11),(10,11),(12,13),(12,14),(13,15),(14,15),
                                                         (1,4),(2,8),(7,13),(11,14)],
                                                     core_node_groups=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
# arch = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(1,6),(3,4),(1,4),(3,6)],
#                                                     core_node_groups=[[0,1,2,3],[4,5,6,7]])
arch.draw()

initial_mapping = telesabre(arch, qc, verbose=True)

# routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)

# print(routed_qc)
# print(mapping)
# print(log)