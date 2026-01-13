from dag import QuantumDAG


def from_qiskit(circuit):
    """
    Convert a Decomposed Qiskit QuantumCircuit to QuantumDAG.
    
    Args:
        circuit: A Decomposed Qiskit QuantumCircuit object
        
    Returns:
        QuantumDAG representation of the decomposed circuit
    """
    dag = QuantumDAG(num_qubits=circuit.num_qubits)
    
    for instruction in circuit.data:
        if hasattr(instruction, 'operation'):
            gate = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        else:
            raise TypeError("Qiskit version < 1.0.2")
        
        gate_type = gate.name.upper()
        
        # Extract parameters if any
        parameters = {}
        if hasattr(gate, 'params') and gate.params:
            for i, param in enumerate(gate.params):
                try:
                    parameters[f'param_{i}'] = float(param)
                except (TypeError, ValueError):
                    raise TypeError("Issues at Decomposed Qiskit Circuit Conversion")
        
        dag.add_gate(
            gate_type=gate_type,
            qubits=qubits,
            parameters=parameters if parameters else None
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