from qiskit import QuantumCircuit, Aer, execute

# create a quantum circuit with 4 qubits and 1 classical bit
circuit = QuantumCircuit(4, 1)

# apply Hadamard gate to the first qubit
circuit.h(0)

# rotate the second qubit by pi/3 around X
circuit.rx(pi/3, 1)

# apply Hadamard gate to the third and fourth qubit
circuit.h(2)
circuit.h(3)

# perform a swap test between qubits 1 and 2, and qubits 3 and 4
circuit.h(1)
circuit.cx(1, 2)
circuit.cx(3, 2)
circuit.h(1)
circuit.measure(2, 0)

# draw the circuit
print(circuit.draw())

# run the circuit on a simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(circuit, backend, shots=1024)
result = job.result()

# print the measurement results
print(result.get_counts())
