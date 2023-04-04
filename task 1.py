import cirq
import matplotlib.pyplot as plt

# Define qubits
q0, q1, q2, q3, q4 = cirq.LineQubit.range(5)

# Define circuit
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.H(q1),
    cirq.H(q2),
    cirq.H(q3),
    cirq.H(q4),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    cirq.CNOT(q2, q3),
    cirq.CNOT(q3, q4),
    cirq.SWAP(q0, q4),
    cirq.X(q3)**0.5,
    cirq.measure(q0, q1, q2, q3, q4, key='result')
)

# Plot circuit
print(circuit)
cirq.plot(circuit)
plt.show()
