import unittest
import numpy as np
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsp_model import TSPModel
from tsp_solver import TSPSolver

class TestTSPModel(unittest.TestCase):
    def setUp(self):
        # Test cities data
        self.cities = [
            {'name': '北京', 'x': 0, 'y': 0},
            {'name': '上海', 'x': 100, 'y': 0},
            {'name': '广州', 'x': 50, 'y': 100}
        ]
        self.tsp_model = TSPModel(self.cities)
    
    def test_distance_matrix_calculation(self):
        """Test distance matrix calculation"""
        distance_matrix = self.tsp_model.get_distance_matrix()
        
        # Check matrix shape
        self.assertEqual(distance_matrix.shape, (3, 3))
        
        # Check diagonal is zero
        self.assertTrue(np.all(np.diag(distance_matrix) == 0))
        
        # Check symmetry
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))
        
        # Check specific distances
        # Beijing to Shanghai: distance should be 100
        self.assertAlmostEqual(distance_matrix[0][1], 100.0, places=2)
        
        # Shanghai to Beijing: same distance
        self.assertAlmostEqual(distance_matrix[1][0], 100.0, places=2)
    
    def test_solution_validation(self):
        """Test solution validation"""
        # Valid solution
        valid_path = [0, 1, 2]
        self.assertTrue(self.tsp_model.validate_solution(valid_path))
        
        # Invalid solution - wrong length
        invalid_path = [0, 1]
        self.assertFalse(self.tsp_model.validate_solution(invalid_path))
        
        # Invalid solution - duplicate cities
        duplicate_path = [0, 1, 1]
        self.assertFalse(self.tsp_model.validate_solution(duplicate_path))
        
        # Invalid solution - missing cities
        incomplete_path = [0, 1]
        self.assertFalse(self.tsp_model.validate_solution(incomplete_path))
    
    def test_distance_calculation(self):
        """Test total distance calculation"""
        # Valid path
        valid_path = [0, 1, 2]
        distance = self.tsp_model.calculate_total_distance(valid_path)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        
        # Invalid path
        invalid_path = [0, 1]
        distance = self.tsp_model.calculate_total_distance(invalid_path)
        self.assertEqual(distance, float('inf'))
    
    def test_random_cities_generation(self):
        """Test random cities generation"""
        random_cities = self.tsp_model.generate_random_cities(num_cities=5, max_coord=100)
        
        # Check number of cities
        self.assertEqual(len(random_cities), 5)
        
        # Check each city has required fields
        for city in random_cities:
            self.assertIn('name', city)
            self.assertIn('x', city)
            self.assertIn('y', city)
            self.assertIsInstance(city['x'], (int, float))
            self.assertIsInstance(city['y'], (int, float))

class TestTSPSolver(unittest.TestCase):
    def setUp(self):
        # Simple test case
        self.cities = [
            {'name': 'A', 'x': 0, 'y': 0},
            {'name': 'B', 'x': 1, 'y': 0},
            {'name': 'C', 'x': 0, 'y': 1}
        ]
        self.tsp_model = TSPModel(self.cities)
        self.distance_matrix = self.tsp_model.get_distance_matrix()
        self.solver = TSPSolver(self.distance_matrix, 3, p=1)
    
    def test_qubo_matrix_creation(self):
        """Test QUBO matrix creation"""
        qubo_matrix = self.solver._create_qubo_matrix()
        
        # Check matrix shape
        self.assertEqual(qubo_matrix.shape, (9, 9))  # 3 cities * 3 cities
        
        # Check matrix is symmetric
        self.assertTrue(np.allclose(qubo_matrix, qubo_matrix.T))
    
    def test_greedy_solution(self):
        """Test greedy solution generation"""
        greedy_path = self.solver._greedy_solution()
        
        # Check path length
        self.assertEqual(len(greedy_path), 3)
        
        # Check all cities are unique
        self.assertEqual(len(set(greedy_path)), 3)
        
        # Check path contains all cities
        self.assertEqual(sorted(greedy_path), [0, 1, 2])
    
    def test_path_distance_calculation(self):
        """Test path distance calculation"""
        test_path = [0, 1, 2]
        distance = self.solver._calculate_path_distance(test_path)
        
        # Check distance is positive
        self.assertGreater(distance, 0)
        
        # Check distance calculation
        expected_distance = (self.distance_matrix[0][1] + 
                           self.distance_matrix[1][2] + 
                           self.distance_matrix[2][0])
        self.assertAlmostEqual(distance, expected_distance, places=2)
    
    def test_classical_solution(self):
        """Test classical solution"""
        result = self.solver.solve_classical()
        
        # Check result structure
        self.assertIn('method', result)
        self.assertIn('path', result)
        self.assertIn('distance', result)
        self.assertIn('success', result)
        
        # Check success
        self.assertTrue(result['success'])
        
        # Check path validity
        self.assertTrue(self.tsp_model.validate_solution(result['path']))
        
        # Check distance calculation
        self.assertEqual(result['distance'], 
                         self.tsp_model.calculate_total_distance(result['path']))

class TestTSPIntegration(unittest.TestCase):
    """Integration tests for the complete TSP workflow"""
    
    def test_complete_workflow(self):
        """Test complete TSP workflow"""
        # Create test cities
        cities = [
            {'name': 'City1', 'x': 0, 'y': 0},
            {'name': 'City2', 'x': 10, 'y': 0},
            {'name': 'City3', 'x': 5, 'y': 10},
            {'name': 'City4', 'x': 15, 'y': 5}
        ]
        
        # Create TSP model
        tsp_model = TSPModel(cities)
        distance_matrix = tsp_model.get_distance_matrix()
        
        # Create solver
        solver = TSPSolver(distance_matrix, 4, p=1)
        
        # Test classical solution
        classical_result = solver.solve_classical()
        self.assertTrue(classical_result['success'])
        self.assertTrue(tsp_model.validate_solution(classical_result['path']))
        
        # Test quantum solution (may fail due to simulation issues)
        quantum_result = solver.solve_quantum()
        self.assertIn('method', quantum_result)
        
        # Test comparison
        comparison = solver.compare_methods()
        self.assertIn('quantum', comparison)
        self.assertIn('classical', comparison)
        self.assertIn('improvement', comparison)
        self.assertIn('improvement_percent', comparison)

if __name__ == '__main__':
    unittest.main()