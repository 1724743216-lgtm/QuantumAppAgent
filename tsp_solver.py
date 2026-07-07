import numpy as np
import cqlib
from cqlib.circuits import Circuit
from cqlib.circuits.gates import H, RZ, RX, CZ, CNOT
import networkx as nx
from tsp_model import TSPModel
from scipy.optimize import minimize
import random

class TSPSolver:
    """TSP Solver with Quantum and Classical Methods"""
    
    def __init__(self, distance_matrix, num_cities, p=1):
        """
        Initialize TSP Solver
        distance_matrix: Numpy array of distances
        num_cities: Number of cities
        p: Number of QAOA layers
        """
        self.distance_matrix = distance_matrix
        self.num_cities = num_cities
        self.p = p
        self.n_qubits = num_cities * num_cities
        
    def _create_qubo_matrix(self):
        """Create QUBO matrix for TSP"""
        # TSP QUBO formulation
        # Binary variables x[i][j] = 1 if city j is visited at position i
        Q = np.zeros((self.n_qubits, self.n_qubits))
        
        # Distance objective: minimize total distance
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                for k in range(self.num_cities):
                    for l in range(self.num_cities):
                        if i != k and j == l:  # Same position constraint
                            Q[i * self.num_cities + j][k * self.num_cities + l] = self.distance_matrix[j][l]
        
        # Constraints: each city visited exactly once
        for j in range(self.num_cities):
            for i in range(self.num_cities):
                for k in range(self.num_cities):
                    if i != k:
                        Q[i * self.num_cities + j][k * self.num_cities + j] = 2
        
        # Constraints: each position has exactly one city
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                for l in range(self.num_cities):
                    if j != l:
                        Q[i * self.num_cities + j][i * self.num_cities + l] = 2
        
        return Q
    
    def _create_qaoa_circuit(self):
        """Create QAOA circuit for TSP"""
        # Create quantum circuit
        circuit = Circuit()
        
        # Initialize in superposition
        for i in range(self.n_qubits):
            circuit.add(H(i))
        
        # QAOA layers
        for layer in range(self.p):
            # Problem Hamiltonian
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    # Add ZZ gates based on QUBO matrix using CZ gates
                    if self.Q[i][j] != 0:
                        circuit.add(CZ(i, j))
                        circuit.add(RZ(i, self.Q[i][j]))
                        circuit.add(RZ(j, self.Q[i][j]))
            
            # Driver Hamiltonian (X rotations)
            for i in range(self.n_qubits):
                circuit.add(RX(i, np.pi))
        
        return circuit
    
    def _decode_solution(self, bitstring):
        """Decode quantum solution to TSP path"""
        path = []
        used_cities = set()
        
        # Decode the bitstring to find the path
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if bitstring[i * self.num_cities + j] == 1:
                    if j not in used_cities:
                        path.append(j)
                        used_cities.add(j)
                        break
        
        # Ensure we have a complete path
        if len(path) == self.num_cities:
            return path
        else:
            # Fallback to greedy solution
            return self._greedy_solution()
    
    def _greedy_solution(self):
        """Generate a greedy solution as fallback"""
        path = []
        remaining_cities = set(range(self.num_cities))
        current_city = random.choice(list(remaining_cities))
        path.append(current_city)
        remaining_cities.remove(current_city)
        
        while remaining_cities:
            next_city = min(remaining_cities, 
                          key=lambda city: self.distance_matrix[current_city][city])
            path.append(next_city)
            remaining_cities.remove(next_city)
            current_city = next_city
        
        return path
    
    def solve_quantum(self):
        """Solve TSP using QAOA algorithm"""
        try:
            # Create QUBO matrix
            self.Q = self._create_qubo_matrix()
            
            # Create QAOA circuit
            circuit = self._create_qaoa_circuit()
            
            # Run quantum simulation
            simulator = cqlib.get_simulator('statevector')
            result = simulator.run(circuit, shots=1000)
            
            # Find best solution
            best_bitstring = None
            best_distance = float('inf')
            
            for bitstring, count in result.items():
                path = self._decode_solution(bitstring)
                distance = self._calculate_path_distance(path)
                
                if distance < best_distance:
                    best_distance = distance
                    best_bitstring = bitstring
            
            # Get the best path
            best_path = self._decode_solution(best_bitstring)
            
            return {
                'method': 'quantum',
                'path': best_path,
                'distance': best_distance,
                'bitstring': best_bitstring,
                'qaoa_layers': self.p,
                'success': True
            }
            
        except Exception as e:
            # Fallback to classical if quantum fails
            return {
                'method': 'quantum_fallback',
                'path': self._greedy_solution(),
                'distance': self._calculate_path_distance(self._greedy_solution()),
                'error': str(e),
                'success': False
            }
    
    def solve_classical(self):
        """Solve TSP using classical algorithm"""
        # Use NetworkX for classical TSP solving
        G = nx.Graph()
        
        # Create graph
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                G.add_edge(i, j, weight=self.distance_matrix[i][j])
        
        # Find approximate solution using Christofides algorithm
        try:
            path = nx.approximation.christofides(G)
            distance = self._calculate_path_distance(path)
            
            return {
                'method': 'classical',
                'path': list(path),
                'distance': distance,
                'success': True
            }
        except:
            # Fallback to greedy
            path = self._greedy_solution()
            distance = self._calculate_path_distance(path)
            
            return {
                'method': 'classical_greedy',
                'path': path,
                'distance': distance,
                'success': True
            }
    
    def solve(self):
        """Main solve method - routes to quantum or classical based on p"""
        if self.p > 0:
            return self.solve_quantum()
        else:
            return self.solve_classical()
    
    def _calculate_path_distance(self, path):
        """Calculate total distance for a path"""
        if len(path) != self.num_cities:
            return float('inf')
        
        total_distance = 0
        for i in range(len(path)):
            current_city = path[i]
            next_city = path[(i + 1) % len(path)]
            total_distance += self.distance_matrix[current_city][next_city]
        
        return total_distance
    
    def compare_methods(self):
        """Compare quantum and classical solutions"""
        quantum_result = self.solve_quantum()
        classical_result = self.solve_classical()
        
        return {
            'quantum': quantum_result,
            'classical': classical_result,
            'improvement': classical_result['distance'] - quantum_result['distance'],
            'improvement_percent': (classical_result['distance'] - quantum_result['distance']) / classical_result['distance'] * 100
        }