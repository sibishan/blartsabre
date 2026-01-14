from dag import QuantumDAG


def from_qiskit(circuit):
    """
    Convert a Decomposed Qiskit QuantumCircuit to QuantumDAG.
    """
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
    """
    Load OpenQASM file and convert to QuantumDAG.
    
    Args:
        filepath: Path to .qasm file
        
    Returns:
        QuantumDAG representation
    """
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit.from_qasm_file(filepath)
    return from_qiskit(circuit)