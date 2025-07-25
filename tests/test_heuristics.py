
"""
Unit tests for heuristic optimization algorithms.
"""

import pytest
import numpy as np
from src.heuristics import RouteOptimizer, SAConfig, TwoOptConfig
from src.utils import generate_random_cities

class TestRouteOptimizer:
    """Test cases for RouteOptimizer class."""
    
    @pytest.fixture
    def sample_cities(self):
        """Create sample cities for testing."""
        return [
            {'id': 0, 'name': 'City A', 'latitude': 40.7128, 'longitude': -74.0060},
            {'id': 1, 'name': 'City B', 'latitude': 34.0522, 'longitude': -118.2437},
            {'id': 2, 'name': 'City C', 'latitude': 41.8781, 'longitude': -87.6298},
            {'id': 3, 'name': 'City D', 'latitude': 29.7604, 'longitude': -95.3698},
            {'id': 4, 'name': 'City E', 'latitude': 33.4484, 'longitude': -112.0740}
        ]
    
    def test_initialization(self, sample_cities):
        """Test RouteOptimizer initialization."""
        optimizer = RouteOptimizer(sample_cities)
        
        assert optimizer.num_cities == len(sample_cities)
        assert optimizer.distance_matrix.shape == (5, 5)
        assert np.all(optimizer.distance_matrix >= 0)
        assert np.all(np.diag(optimizer.distance_matrix) == 0)
    
    def test_nearest_neighbor(self, sample_cities):
        """Test Nearest Neighbor algorithm."""
        optimizer = RouteOptimizer(sample_cities)
        
        route, distance = optimizer.nearest_neighbor(verbose=False)
        
        # Check route validity
        assert len(route) == len(sample_cities)
        assert set(route) == set(range(len(sample_cities)))
        assert distance > 0
        assert isinstance(distance, float)
        
        # Test different starting cities
        route1, distance1 = optimizer.nearest_neighbor(start_city=0, verbose=False)
        route2, distance2 = optimizer.nearest_neighbor(start_city=1, verbose=False)
        
        assert route1[0] == 0
        assert route2[0] == 1
    
    def test_two_opt(self, sample_cities):
        """Test 2-opt algorithm."""
        optimizer = RouteOptimizer(sample_cities)
        
        # Get initial route from nearest neighbor
        initial_route, initial_distance = optimizer.nearest_neighbor(verbose=False)
        
        # Apply 2-opt improvement
        config = TwoOptConfig(max_iterations=100, improvement_threshold=0.01)
        improved_route, improved_distance = optimizer.two_opt(
            initial_route=initial_route, config=config, verbose=False
        )
        
        # Check route validity
        assert len(improved_route) == len(sample_cities)
        assert set(improved_route) == set(range(len(sample_cities)))
        
        # Distance should be same or better
        assert improved_distance <= initial_distance
        
        # Test without initial route (should use nearest neighbor)
        route_auto, distance_auto = optimizer.two_opt(verbose=False)
        assert len(route_auto) == len(sample_cities)
        assert set(route_auto) == set(range(len(sample_cities)))
    
    def test_simulated_annealing(self, sample_cities):
        """Test Simulated Annealing algorithm."""
        optimizer = RouteOptimizer(sample_cities)
        
        config = SAConfig(
            initial_temperature=100,
            cooling_rate=0.95,
            min_temperature=1,
            max_iterations=1000
        )
        
        route, distance = optimizer.simulated_annealing(config=config, verbose=False)
        
        # Check route validity
        assert len(route) == len(sample_cities)
        assert set(route) == set(range(len(sample_cities)))
        assert distance > 0
        assert isinstance(distance, float)
        
        # Test without initial route
        route_auto, distance_auto = optimizer.simulated_annealing(verbose=False)
        assert len(route_auto) == len(sample_cities)
        assert set(route_auto) == set(range(len(sample_cities)))
    
    def test_random_search(self, sample_cities):
        """Test Random Search algorithm."""
        optimizer = RouteOptimizer(sample_cities)
        
        route, distance = optimizer.random_search(num_iterations=100, verbose=False)
        
        # Check route validity
        assert len(route) == len(sample_cities)
        assert set(route) == set(range(len(sample_cities)))
        assert distance > 0
        assert isinstance(distance, float)
    
    def test_christofides_approximation(self, sample_cities):
        """Test Christofides approximation algorithm."""
        optimizer = RouteOptimizer(sample_cities)
        
        route, distance = optimizer.christofides_approximation(verbose=False)
        
        # Check route validity
        assert len(route) == len(sample_cities)
        assert set(route) == set(range(len(sample_cities)))
        assert distance > 0
        assert isinstance(distance, float)
    
    def test_minimum_spanning_tree(self, sample_cities):
        """Test MST construction."""
        optimizer = RouteOptimizer(sample_cities)
        
        mst_edges = optimizer._minimum_spanning_tree()
        
        # MST should have n-1 edges
        assert len(mst_edges) == len(sample_cities) - 1
        
        # All edges should be valid
        for u, v in mst_edges:
            assert 0 <= u < len(sample_cities)
            assert 0 <= v < len(sample_cities)
            assert u != v
    
    def test_odd_degree_vertices(self, sample_cities):
        """Test finding odd degree vertices."""
        optimizer = RouteOptimizer(sample_cities)
        
        # Create a simple set of edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        odd_vertices = optimizer._find_odd_degree_vertices(edges)
        
        # Vertices 0 and 4 should have degree 1 (odd)
        # Vertices 1, 2, 3 should have degree 2 (even)
        assert 0 in odd_vertices
        assert 4 in odd_vertices
        assert len(odd_vertices) == 2
    
    def test_minimum_weight_matching(self, sample_cities):
        """Test minimum weight perfect matching."""
        optimizer = RouteOptimizer(sample_cities)
        
        # Test with 4 vertices (even number for perfect matching)
        vertices = [0, 1, 2, 3]
        matching = optimizer._minimum_weight_matching(vertices)
        
        # Should have 2 edges for 4 vertices
        assert len(matching) == 2
        
        # All vertices should be matched exactly once
        matched_vertices = set()
        for u, v in matching:
            assert u not in matched_vertices
            assert v not in matched_vertices
            matched_vertices.add(u)
            matched_vertices.add(v)
        
        assert matched_vertices == set(vertices)
    
    def test_compare_algorithms(self, sample_cities):
        """Test algorithm comparison."""
        optimizer = RouteOptimizer(sample_cities)
        
        results = optimizer.compare_algorithms(verbose=False)
        
        # Should have results for multiple algorithms
        expected_algorithms = ['nearest_neighbor', 'two_opt', 'simulated_annealing']
        
        for alg in expected_algorithms:
            assert alg in results
            assert 'route' in results[alg]
            assert 'distance' in results[alg]
            assert 'route_names' in results[alg]
            
            # Check route validity
            route = results[alg]['route']
            assert len(route) == len(sample_cities)
            assert set(route) == set(range(len(sample_cities)))
    
    def test_algorithm_performance_order(self, sample_cities):
        """Test that algorithms generally perform in expected order."""
        optimizer = RouteOptimizer(sample_cities)
        
        # Run algorithms
        nn_route, nn_distance = optimizer.nearest_neighbor(verbose=False)
        two_opt_route, two_opt_distance = optimizer.two_opt(
            initial_route=nn_route, verbose=False
        )
        
        # 2-opt should improve or maintain nearest neighbor solution
        assert two_opt_distance <= nn_distance
    
    def test_large_problem(self):
        """Test algorithms on larger problem instance."""
        cities = generate_random_cities(15)
        optimizer = RouteOptimizer(cities)
        
        # Test nearest neighbor
        nn_route, nn_distance = optimizer.nearest_neighbor(verbose=False)
        assert len(nn_route) == 15
        assert set(nn_route) == set(range(15))
        
        # Test 2-opt improvement
        config = TwoOptConfig(max_iterations=500)
        improved_route, improved_distance = optimizer.two_opt(
            initial_route=nn_route, config=config, verbose=False
        )
        assert improved_distance <= nn_distance
        
        # Test simulated annealing
        sa_config = SAConfig(max_iterations=2000)
        sa_route, sa_distance = optimizer.simulated_annealing(
            config=sa_config, verbose=False
        )
        assert len(sa_route) == 15
        assert set(sa_route) == set(range(15))
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with 2 cities
        cities = generate_random_cities(2)
        optimizer = RouteOptimizer(cities)
        
        route, distance = optimizer.nearest_neighbor(verbose=False)
        assert len(route) == 2
        assert distance > 0
        
        # Test 2-opt with 2 cities (should not change anything)
        improved_route, improved_distance = optimizer.two_opt(
            initial_route=route, verbose=False
        )
        assert improved_route == route
        assert improved_distance == distance
    
    def test_configuration_objects(self):
        """Test configuration objects."""
        # Test SAConfig
        sa_config = SAConfig(
            initial_temperature=500,
            cooling_rate=0.9,
            min_temperature=0.5,
            max_iterations=5000
        )
        
        assert sa_config.initial_temperature == 500
        assert sa_config.cooling_rate == 0.9
        assert sa_config.min_temperature == 0.5
        assert sa_config.max_iterations == 5000
        
        # Test TwoOptConfig
        two_opt_config = TwoOptConfig(
            max_iterations=2000,
            improvement_threshold=0.001
        )
        
        assert two_opt_config.max_iterations == 2000
        assert two_opt_config.improvement_threshold == 0.001
    
    def test_distance_calculation_consistency(self, sample_cities):
        """Test that distance calculations are consistent across algorithms."""
        optimizer = RouteOptimizer(sample_cities)
        
        # Create a specific route
        route = [0, 1, 2, 3, 4]
        
        # Calculate distance using the optimizer method
        distance = optimizer._calculate_route_distance(route)
        
        # Calculate manually
        manual_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            manual_distance += optimizer.distance_matrix[from_city][to_city]
        
        assert abs(distance - manual_distance) < 1e-10
    
    def test_algorithm_determinism(self, sample_cities):
        """Test that deterministic algorithms produce consistent results."""
        optimizer = RouteOptimizer(sample_cities)
        
        # Nearest neighbor should be deterministic for same starting city
        route1, distance1 = optimizer.nearest_neighbor(start_city=0, verbose=False)
        route2, distance2 = optimizer.nearest_neighbor(start_city=0, verbose=False)
        
        assert route1 == route2
        assert abs(distance1 - distance2) < 1e-10
        
        # 2-opt should be deterministic for same initial route
        improved1, dist1 = optimizer.two_opt(initial_route=route1, verbose=False)
        improved2, dist2 = optimizer.two_opt(initial_route=route1, verbose=False)
        
        assert improved1 == improved2
        assert abs(dist1 - dist2) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__])
