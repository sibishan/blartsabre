from dag import QuantumDAG
from qiskit import qasm2


def from_qiskit(circuit):
    dag = QuantumDAG(num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits)

    for instruction in circuit.data:
        if not hasattr(instruction, "operation"):
            raise TypeError("Qiskit version < 1.0.2")

        gate = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        clbits = [circuit.find_bit(c).index for c in instruction.clbits] if getattr(instruction, "clbits", None) else []

        gate_type = gate.name.upper()

        parameters = {}
        if hasattr(gate, "params") and gate.params:
            for i, param in enumerate(gate.params):
                try:
                    parameters[f"param_{i}"] = float(param)
                except (TypeError, ValueError):
                    raise TypeError("Issues at Decomposed Qiskit Circuit Conversion")

        dag.add_gate(
            gate_type=gate_type,
            qubits=qubits,
            clbits=clbits if clbits else None,
            parameters=parameters if parameters else None,
        )

    return dag


def from_qasm_file(filepath: str):
    circuit = qasm2.load(filepath)
    return from_qiskit(circuit)