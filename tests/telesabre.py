from mapper.telesabre import telesabre_layout
from router.telesabre import telesabre_swap
from architecture import DistributedQubitNetworkGraph, multi_core_grid, two_tokyo, five_tokyo
from qiskit import qasm2

qc = qasm2.load("./data/telesabre/qasm_25/ae_nativegates_ibm_qiskit_opt3_25.qasm")

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

arch = two_tokyo()

initial_mapping = telesabre_layout(arch, qc, verbose=True, seed=1)
routed_qc, mapping, log = telesabre_swap(arch, qc, initial_mapping)





# qc = qasm2.load("./data/example_4q/example_4q.qasm")

# # num_rows = 2
# # qc_entangled = QuantumCircuit(2 * num_rows, 2 * num_rows)
# # for i in range(num_rows):
# #     qc_entangled.h(i)
# #     qc_entangled.cx(i, i + num_rows)
# #     qc_entangled.measure(i, i)
# #     qc_entangled.measure(i + num_rows, i + num_rows)

# arch = DistributedQubitNetworkGraph([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(3,4)], core_node_groups=[[0,1,2,3],[4,5,6,7]])
# arch.draw()
# initial_mapping = telesabre(arch, qc, verbose=True)

# print(initial_mapping)





# Prof's idea test

# arch = SingleCoreDQGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),
#                                                          (8,9),(8,10),(9,11),(10,11),(12,13),(12,14),(13,15),(14,15),
#                                                          (1,4),(2,8),(7,13),(11,14)],
#                                                      name="Distributed Qubit Graph with Single Core",
#                                                      comm_edges=[(1,4),(2,8),(7,13),(11,14)])
# # arch = DistributedQubitNetworkGraph([(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(1,6),(3,4),(1,4),(3,6)],
# #                                                     core_node_groups=[[0,1,2,3],[4,5,6,7]])
# # arch.draw()

# initial_mapping = sabre(arch, qc, verbose=True)

# routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)
# # print(arch.distance_matrix)
# print(routed_qc)
# print(mapping)
# print(log)