from mapper.resabre import resabre
from architecture import SingleCoreDQGraph
import qiskit.qasm2

qc = qiskit.qasm2.load("./data/example_9q/example_9q.qasm")

# 15 qubit device
arch = SingleCoreDQGraph([(0,1),(0,2),(0,3),(0,4),
                            (5,6),(6,7),(7,8),(8,9),
                            (10,11),(11,12),(12,13),(13,14),(14,10),
                            (0,5), (9,10)],
                        comm_edges=[(0,5),(9,10)],
                        core_node_groups=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]],
                        name="15-qubit-star-line-ring"
                        )

initial_mapping = resabre(arch, qc, verbose=True)

# routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)
# print(arch.distance_matrix)
# print(routed_qc)
# print(mapping)
# print(log)