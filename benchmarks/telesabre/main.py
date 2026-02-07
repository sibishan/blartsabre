from qiskit import qasm2

from .architecture import STAR_LINE_RING, TWO_TOKYO
from .circuit import Circuit

from .config import Config
from .telesabre_layout import run_telesabre
from .telesabre_swap import telesabre_swap


config = Config()

# qc = qasm2.load("./data/example_4q/example_4q.qasm")
qc = qasm2.load("./data/telesabre/qasm_25/ghz_nativegates_ibm_qiskit_opt3_25.qasm")
cir = Circuit.from_qiskit(qc)
print(cir)

# arch = STAR_LINE_RING()
arch = TWO_TOKYO()

# install graphviz for FileNotFoundError: [Errno 2] "neato" not found in path using sudo apt-get update && sudo apt-get install -y graphviz
swaps, tps, telegate, depth, tp_depth, deadlocks, initial_layout, solving_deadlock = run_telesabre(config, cir, arch)

routed_qc, final_layout, log = telesabre_swap(config, cir, arch, initial_layout)

print(log)