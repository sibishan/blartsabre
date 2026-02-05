from mapper.resabre import resabre
from router.resabre import sabre_swap
from architecture import SingleCoreDQGraph
import qiskit.qasm2

qc = qiskit.qasm2.load("./data/queko/BSS/16QBT_900CYC_QSE_9.qasm")

# 20 qubit device
arch = SingleCoreDQGraph([(0,1),(0,2),(0,3),(0,4),
                            (5,6),(6,7),(7,8),(8,9),
                            (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,10),
                            (0,5), (9,10)],
                        comm_edges=[(0,5),(9,10)],
                        core_node_groups=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]],
                        name="20-qubit-star-line-ring"
                        )

initial_mapping = resabre(arch, qc, verbose=True)

routed_qc, mapping, log = sabre_swap(arch, qc, initial_mapping)
print(mapping)

def validate_physical_log(arch, log):
    for i, (op, payload) in enumerate(log):
        if op == "SWAP":
            u, v = payload
            if not arch.has_edge(u, v):
                raise RuntimeError(f"Illegal SWAP at step {i}: ({u},{v}) not an edge")
        elif op == "GATE":
            gt, qubits = payload
            if gt.upper() in ("CX", "CZ", "SWAP"):
                u, v = qubits
                if not arch.has_edge(u, v):
                    raise RuntimeError(f"Illegal {gt} at step {i}: ({u},{v}) not an edge")
    return True

# usage:
validate_physical_log(arch, log)
print("Log is physically valid on arch")

def count_comm_swaps(arch, log):
    return sum(1 for op, e in log if op == "SWAP" and arch.is_comm_edge(*e))

print("comm swaps =", count_comm_swaps(arch, log))
print("total swaps =", sum(1 for op, _ in log if op == "SWAP"))

from collections import Counter

def count_swaps_in_log(log):
    return sum(1 for op, _ in log if op == "SWAP")

def count_2q_gates_in_log(log):
    c = 0
    for op, payload in log:
        if op == "GATE":
            gt, _ = payload
            if gt.upper() in ("CX","CZ","SWAP"):
                c += 1
    return c

print("log swaps:", count_swaps_in_log(log))
print("qc swaps :", routed_qc.count_ops().get("swap", 0))
print("log CX/CZ:", count_2q_gates_in_log(log))
print("qc CX/CZ :", routed_qc.count_ops().get("cx", 0) + routed_qc.count_ops().get("cz", 0))


