import numpy as np
import networkx as nx

class TSPModel:
    """Traveling Salesman Problem Model with distance matrix generation"""
    
    def __init__(self, cities):
        """
        Initialize TSP model with cities data
        cities: List of dictionaries with 'name' and 'x', 'y' coordinates
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self):
        """Calculate Euclidean distance matrix between cities"""
        distance_matrix = np.zeros((self.num_cities, self.num_cities))
        
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    city1 = self.cities[i]
                    city2 = self.cities[j]
                    # Euclidean distance
                    distance = np.sqrt(
                        (city1['x'] - city2['x'])**2 + 
                        (city1['y'] - city2['y'])**2
                    )
                    distance_matrix[i][j] = distance
        
        return distance_matrix
    
    def get_distance_matrix(self):
        """Get the distance matrix"""
        return self.distance_matrix
    
    def get_graph(self):
        """Get the graph representation of the TSP"""
        G = nx.Graph()
        
        # Add nodes
        for i, city in enumerate(self.cities):
            G.add_node(i, name=city['name'], x=city['x'], y=city['y'])
        
        # Add edges with weights
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                G.add_edge(i, j, weight=self.distance_matrix[i][j])
        
        return G
    
    def generate_random_cities(self, num_cities=5, max_coord=100):
        """Generate random cities for testing"""
        import random
        
        cities = []
        for i in range(num_cities):
            cities.append({
                'name': f'City_{i+1}',
                'x': random.uniform(0, max_coord),
                'y': random.uniform(0, max_coord)
            })
        
        return cities
    
    def validate_solution(self, path):
        """Validate if a solution is valid (visits all cities exactly once)"""
        if len(path) != self.num_cities:
            return False
        
        # Check if all cities are visited exactly once
        if sorted(path) != list(range(self.num_cities)):
            return False
        
        return True
    
    def calculate_total_distance(self, path):
        """Calculate total distance for a given path"""
        if not self.validate_solution(path):
            return float('inf')
        
        total_distance = 0
        for i in range(len(path)):
            current_city = path[i]
            next_city = path[(i + 1) % len(path)]
            total_distance += self.distance_matrix[current_city][next_city]
        
        return total_distance