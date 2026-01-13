"""
Qiskit Circuit Converter for DQC DAG

This module provides utilities to convert between Qiskit QuantumCircuit
and our QuantumDAG representation for distributed quantum computing compilation.
"""

from typing import Optional, List, Dict
from .dag import QuantumDAG


def from_qiskit(circuit) -> QuantumDAG:
    """
    Convert a Decomposed Qiskit QuantumCircuit to QuantumDAG.
    
    Args:
        circuit: A Decomposed Qiskit QuantumCircuit object
        
    Returns:
        QuantumDAG representation of the decomposed circuit
        
    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(4)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> dag = from_qiskit(qc)
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


def from_qasm_file(filepath: str) -> QuantumDAG:
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