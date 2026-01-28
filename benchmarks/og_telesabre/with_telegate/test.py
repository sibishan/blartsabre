from qiskit import qasm2

from benchmarks.og_telesabre.architecture import STAR_LINE_RING, TWO_TOKYO
from benchmarks.og_telesabre.circuit import Circuit

from .config import Config
from .telesabre import run_telesabre


config = Config()

# qc = qasm2.load("./data/example_4q/example_4q.qasm")
qc = qasm2.load("./data/telesabre/qasm_25/ghz_nativegates_ibm_qiskit_opt3_25.qasm")
cir = Circuit.from_qiskit(qc)
print(cir)

# arch = STAR_LINE_RING()
arch = TWO_TOKYO()

# install graphviz for FileNotFoundError: [Errno 2] "neato" not found in path using sudo apt-get update && sudo apt-get install -y graphviz
swaps, tps, telegate, depth, tp_depth, deadlocks, initial_layout, solving_deadlock = run_telesabre(config, cir, arch)

print(tps)

