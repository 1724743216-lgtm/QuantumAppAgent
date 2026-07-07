from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import numpy as np
from tsp_solver import TSPSolver
from tsp_model import TSPModel

app = Flask(__name__)
CORS(app)

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/cities', methods=['POST'])
def define_cities():
    """Define cities and their coordinates"""
    data = request.json
    cities = data.get('cities', [])
    
    if not cities:
        return jsonify({'error': 'No cities provided'}), 400
    
    # Create TSP model
    tsp_model = TSPModel(cities)
    distance_matrix = tsp_model.get_distance_matrix()
    
    return jsonify({
        'distance_matrix': distance_matrix.tolist(),
        'city_names': [city['name'] for city in cities],
        'num_cities': len(cities)
    })

@app.route('/api/solve', methods=['POST'])
def solve_tsp():
    """Solve TSP using quantum algorithm"""
    data = request.json
    distance_matrix = np.array(data.get('distance_matrix', []))
    num_cities = data.get('num_cities', 0)
    p = data.get('p', 1)  # QAOA layers
    
    if distance_matrix.size == 0 or num_cities == 0:
        return jsonify({'error': 'Invalid distance matrix or city count'}), 400
    
    try:
        # Create and solve TSP
        tsp_solver = TSPSolver(distance_matrix, num_cities, p)
        result = tsp_solver.solve()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classical', methods=['POST'])
def solve_classical():
    """Solve TSP using classical algorithm for comparison"""
    data = request.json
    distance_matrix = np.array(data.get('distance_matrix', []))
    num_cities = data.get('num_cities', 0)
    
    if distance_matrix.size == 0 or num_cities == 0:
        return jsonify({'error': 'Invalid distance matrix or city count'}), 400
    
    try:
        # Create and solve TSP classically
        tsp_solver = TSPSolver(distance_matrix, num_cities, 0)  # p=0 for classical
        result = tsp_solver.solve_classical()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'TSP Quantum API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)