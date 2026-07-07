"""
VQE Solver for H2 Molecule Ground-State Energy
================================================

Uses cqlib SDK to construct parameterized circuits and StatevectorSimulator
for expectation-value evaluation. Optimizes with scipy.minimize.

Hamiltonian: H2 molecule at bond distance R=0.735 Å (minimal basis, 2 qubits)
Reference: O'Malley et al. (2016), Phys. Rev. X 6, 031007
"""

import json
import numpy as np
from cqlib import Circuit, Parameter
from cqlib.simulator import StatevectorSimulator
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Hamiltonian definition (H2 minimal, Bravyi-Kitaev mapping, R=0.735 Å)
# ---------------------------------------------------------------------------
H2_HAMILTONIAN = [
    (-1.052373245772859, []),                            # identity
    (0.39793742484318045, [(0, "Z")]),                   # Z0
    (-0.39793742484318045, [(1, "Z")]),                  # Z1
    (-0.01128010425623538, [(0, "Z"), (1, "Z")]),        # Z0Z1
    (0.18093119978423156, [(0, "X"), (1, "X")]),         # X0X1
]

EXACT_GROUND_ENERGY = -1.85727503  # from exact diagonalization of the same Hamiltonian


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------
def hardware_efficient_ansatz(n_qubits: int = 2, layers: int = 2):
    """Build a hardware-efficient ansatz with Ry-Rz-CX layers.

    Returns:
        circuit: cqlib Circuit with unbound parameters
        param_names: list of parameter name strings in binding order
    """
    param_names = []
    for layer in range(layers):
        for q in range(n_qubits):
            param_names.extend([f"ry_{layer}_{q}", f"rz_{layer}_{q}"])

    circuit = Circuit(n_qubits, parameters=param_names)
    for layer in range(layers):
        for q in range(n_qubits):
            circuit.ry(q, Parameter(f"ry_{layer}_{q}"))
            circuit.rz(q, Parameter(f"rz_{layer}_{q}"))
        for q in range(n_qubits - 1):
            circuit.cx(q, q + 1)
    return circuit, param_names


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def copy_with_basis_rotation(circuit, pauli_ops):
    """Append basis-rotation gates for measuring in X/Y basis."""
    measured = circuit.copy()
    for qubit, op in pauli_ops:
        if op == "X":
            measured.h(qubit)
        elif op == "Y":
            measured.rx(qubit, np.pi / 2)
        elif op != "Z":
            raise ValueError(f"Unsupported Pauli operator: {op}")
    measured.measure_all()
    return measured


def pauli_expectation(circuit, pauli_ops):
    """Compute ⟨ψ|P|ψ⟩ using StatevectorSimulator measurement probabilities."""
    measured = copy_with_basis_rotation(circuit, pauli_ops)
    probs = StatevectorSimulator(circuit=measured).measure()
    value = 0.0
    for bits, prob in probs.items():
        parity = sum(int(bits[-1 - qubit]) for qubit, _ in pauli_ops)
        value += ((-1) ** parity) * prob
    return value


def energy(params, circuit, param_names, hamiltonian):
    """Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ for given parameter vector."""
    bound = circuit.assign_parameters(dict(zip(param_names, params)))
    total = 0.0
    for coeff, ops in hamiltonian:
        if len(ops) == 0:
            total += coeff
        else:
            total += coeff * pauli_expectation(bound, ops)
    return total


# ---------------------------------------------------------------------------
# Classical baseline: exact diagonalization
# ---------------------------------------------------------------------------
def exact_diagonalization(hamiltonian, n_qubits=2):
    """Build the full Hamiltonian matrix and find the smallest eigenvalue."""
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    for coeff, ops in hamiltonian:
        # Build tensor product for this term
        matrices = [pauli["I"]] * n_qubits
        for qubit, op in ops:
            matrices[qubit] = pauli[op]
        term = matrices[0]
        for m in matrices[1:]:
            term = np.kron(term, m)
        H += coeff * term

    eigenvalues = np.linalg.eigvalsh(H)
    return float(eigenvalues[0])


# ---------------------------------------------------------------------------
# VQE run
# ---------------------------------------------------------------------------
def run_vqe(
    hamiltonian=H2_HAMILTONIAN,
    n_qubits=2,
    layers=2,
    max_iter=200,
    seed=42,
):
    """Run VQE optimization and return results dict."""
    rng = np.random.default_rng(seed)
    circuit, param_names = hardware_efficient_ansatz(n_qubits, layers)
    x0 = rng.uniform(0, 2 * np.pi, size=len(param_names))

    convergence_trace = []

    def callback(xk):
        e = energy(xk, circuit, param_names, hamiltonian)
        convergence_trace.append(float(e))

    result = minimize(
        energy,
        x0,
        args=(circuit, param_names, hamiltonian),
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 0.5},
        callback=callback,
    )

    return {
        "optimal_params": result.x.tolist(),
        "optimal_energy": float(result.fun),
        "iterations": len(convergence_trace),
        "convergence_trace": convergence_trace,
        "success": result.success,
        "message": result.message,
        "n_qubits": n_qubits,
        "layers": layers,
        "seed": seed,
        "param_count": len(param_names),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("VQE: H2 Molecule Ground-State Energy")
    print("=" * 60)

    # Classical baseline
    exact_energy = exact_diagonalization(H2_HAMILTONIAN)
    print(f"\n[Classical] Exact diagonalization energy: {exact_energy:.8f}")

    # VQE
    print("\n[Quantum] Running VQE optimization ...")
    vqe_result = run_vqe()
    print(f"  VQE optimal energy:   {vqe_result['optimal_energy']:.8f}")
    print(f"  Exact reference:      {EXACT_GROUND_ENERGY:.8f}")
    print(f"  Absolute error:       {abs(vqe_result['optimal_energy'] - EXACT_GROUND_ENERGY):.6e}")
    print(f"  Iterations:           {vqe_result['iterations']}")
    print(f"  Parameters:           {vqe_result['param_count']}")

    # Save reports
    baseline_report = {
        "task": "H2 ground-state energy estimation",
        "data": "H2 Hamiltonian (R=0.735 Å, minimal basis, 2 qubits)",
        "primary_metric": "energy (Hartree)",
        "higher_is_better": False,
        "value": exact_energy,
        "method": "exact_diagonalization",
        "command": "python vqe_solver.py",
        "artifact_paths": [],
        "backend": "numpy.linalg.eigvalsh",
        "qubits": 2,
        "limitations": ["Analytical result for fixed Hamiltonian; no variational approximation"],
    }

    quantum_report = {
        "task": "H2 ground-state energy estimation",
        "data": "H2 Hamiltonian (R=0.735 Å, minimal basis, 2 qubits)",
        "primary_metric": "energy (Hartree)",
        "higher_is_better": False,
        "value": vqe_result["optimal_energy"],
        "method": "VQE (hardware-efficient ansatz, COBYLA optimizer)",
        "command": "python vqe_solver.py",
        "artifact_paths": [],
        "seed": vqe_result["seed"],
        "backend": "cqlib StatevectorSimulator",
        "qubits": 2,
        "circuit_depth": vqe_result["layers"],
        "param_count": vqe_result["param_count"],
        "iterations": vqe_result["iterations"],
        "convergence_trace": vqe_result["convergence_trace"],
        "limitations": [
            "Statevector simulator (no shot noise)",
            "Small Hamiltonian (2 qubits)",
            f"Energy error vs exact: {abs(vqe_result['optimal_energy'] - exact_energy):.6e} Hartree",
        ],
    }

    with open("baseline_report.json", "w") as f:
        json.dump(baseline_report, f, indent=2, ensure_ascii=False)
    with open("quantum_report.json", "w") as f:
        json.dump(quantum_report, f, indent=2, ensure_ascii=False)

    print("\nReports saved: baseline_report.json, quantum_report.json")
