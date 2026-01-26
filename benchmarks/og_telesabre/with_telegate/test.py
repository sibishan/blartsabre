from qiskit import qasm2

from benchmarks.og_telesabre.architecture import Architecture, Edge, TeleportEdge
from benchmarks.og_telesabre.circuit import Circuit

from .config import Config
from .telesabre import run_telesabre


config = Config()

qc = qasm2.load("./data/example_9q/example_9q.qasm")
cir = Circuit.from_qiskit(qc)
print(cir)

num_qubits = 15
qubit_to_core = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]

# core 0: line 0-1-2-3-4
edges0 = [Edge(0,1), Edge(1,2), Edge(2,3), Edge(3,4)]
# core 1: star centred at 5
edges1 = [Edge(5,6), Edge(5,7), Edge(5,8), Edge(5,9)]
# core 2: ring
edges2 = [Edge(10,11), Edge(11,12), Edge(12,13), Edge(13,14), Edge(14,10)]

intra_core_edges = edges0 + edges1 + edges2

# inter core links between communication qubits
inter_core_edges = [
    Edge(2,5), # core0 comm qubit 2 <-> core1 comm qubit 5
    Edge(3,10) # core0 comm qubit 3 <-> core1 comm qubit 10
]

arch = Architecture(
    num_qubits=num_qubits,
    qubit_to_core=qubit_to_core,
    intra_core_edges=intra_core_edges,
    inter_core_edges=inter_core_edges,
    name="star-line-ring"
)

# install graphviz for FileNotFoundError: [Errno 2] "neato" not found in path using sudo apt-get update && sudo apt-get install -y graphviz
swaps, tps, telegate, depth, tp_depth, deadlocks, initial_layout, solving_deadlock = run_telesabre(config, cir, arch)

print(initial_layout)

