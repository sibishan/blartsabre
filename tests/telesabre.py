from mapper.telesabre import telesabre_layout
from architecture import DistributedQubitNetworkGraph, multi_core_grid, two_tokyo, four_tokyo
from qiskit import qasm2

qc = qasm2.load("./data/quekno/53Q_depth_Rochester/53QBT_depth_Rochester_large_opt_1_1.5_no.1.qasm")

# arch = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),
#                                                          (8,9),(8,10),(9,11),(10,11),(12,13),(12,14),(13,15),(14,15),
#                                                          (1,4),(2,8),(7,13),(11,14)],
#                                                      core_node_groups=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

# 15 qubit device
# arch = DistributedQubitNetworkGraph([(0,1),(0,2),(0,3),(0,4),
#                                      (5,6),(6,7),(7,8),(8,9),
#                                      (10,11),(11,12),(12,13),(13,14),(14,10),
#                                      (0,5), (9,10)],
#                                     core_node_groups=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]],
#                                     name="15-qubit-star-line-ring"
#                                     )

# 18 qubit device
# arch = DistributedQubitNetworkGraph([(0,1),(0,2),(0,3),(0,4),(0,5),
#                                      (6,7),(7,8),(8,9),(9,10),(10,11),
#                                      (11,13),(13,14),(14,15),(15,16),(16,17),(17,12),
#                                      (0,6), (11,12)],
#                                     core_node_groups=[[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17]],
#                                     name="18-qubit-star-line-ring"
#                                     )

arch = four_tokyo() # 40 qubits
arch.draw()

initial_mapping = telesabre_layout(arch, qc, verbose=True, seed=1, num_iterations=3)