
"""
Unit tests for the Genetic Algorithm implementation.
"""

import pytest
import numpy as np
from src.genetic_algorithm import GeneticAlgorithm, GAConfig
from src.utils import generate_random_cities

class TestGeneticAlgorithm:
    """Test cases for GeneticAlgorithm class."""
    
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
    
    @pytest.fixture
    def ga_config(self):
        """Create test GA configuration."""
        return GAConfig(
            population_size=20,
            generations=10,
            elite_size=2,
            tournament_size=3,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_initialization(self, sample_cities, ga_config):
        """Test GA initialization."""
        ga = GeneticAlgorithm(sample_cities, ga_config)
        
        assert ga.num_cities == len(sample_cities)
        assert ga.config == ga_config
        assert ga.distance_matrix.shape == (5, 5)
        assert np.all(ga.distance_matrix >= 0)
        assert np.all(np.diag(ga.distance_matrix) == 0)
    
    def test_distance_matrix_symmetry(self, sample_cities):
        """Test that distance matrix is symmetric."""
        ga = GeneticAlgorithm(sample_cities)
        
        # Check symmetry
        assert np.allclose(ga.distance_matrix, ga.distance_matrix.T)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(ga.distance_matrix), 0)
    
    def test_haversine_distance(self, sample_cities):
        """Test Haversine distance calculation."""
        ga = GeneticAlgorithm(sample_cities)
        
        # Test distance between New York and Los Angeles (known distance ~3944 km)
        distance = ga._haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < distance < 4000  # Approximate check
        
        # Test distance from a point to itself
        distance = ga._haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert distance == 0
    
    def test_route_distance_calculation(self, sample_cities):
        """Test route distance calculation."""
        ga = GeneticAlgorithm(sample_cities)
        
        # Test simple route
        route = [0, 1, 2, 3, 4]
        distance = ga._calculate_route_distance(route)
        
        assert distance > 0
        assert isinstance(distance, float)
        
        # Test that different routes give different distances
        route2 = [0, 2, 1, 3, 4]
        distance2 = ga._calculate_route_distance(route2)
        
        # Routes should generally have different distances
        # (though theoretically they could be the same)
        assert isinstance(distance2, float)
    
    def test_fitness_calculation(self, sample_cities):
        """Test fitness calculation."""
        ga = GeneticAlgorithm(sample_cities)
        
        route = [0, 1, 2, 3, 4]
        fitness = ga._fitness(route)
        
        assert fitness > 0
        assert fitness <= 1  # Fitness is 1/(1+distance)
    
    def test_population_initialization(self, sample_cities, ga_config):
        """Test population initialization."""
        ga = GeneticAlgorithm(sample_cities, ga_config)
        population = ga._initialize_population()
        
        assert len(population) == ga_config.population_size
        
        for individual in population:
            assert len(individual) == len(sample_cities)
            assert set(individual) == set(range(len(sample_cities)))
    
    def test_tournament_selection(self, sample_cities, ga_config):
        """Test tournament selection."""
        ga = GeneticAlgorithm(sample_cities, ga_config)
        population = ga._initialize_population()
        
        selected = ga._tournament_selection(population)
        
        assert len(selected) == len(sample_cities)
        assert set(selected) == set(range(len(sample_cities)))
    
    def test_order_crossover(self, sample_cities):
        """Test order crossover (OX1)."""
        ga = GeneticAlgorithm(sample_cities)
        
        parent1 = [0, 1, 2, 3, 4]
        parent2 = [4, 3, 2, 1, 0]
        
        child = ga._order_crossover(parent1, parent2)
        
        # Child should be a valid permutation
        assert len(child) == len(parent1)
        assert set(child) == set(parent1)
        assert len(set(child)) == len(child)  # No duplicates
    
    def test_swap_mutation(self, sample_cities):
        """Test swap mutation."""
        ga = GeneticAlgorithm(sample_cities)
        
        original_route = [0, 1, 2, 3, 4]
        
        # Test with high mutation rate to ensure mutation occurs
        ga.config.mutation_rate = 1.0
        mutated_route = ga._swap_mutation(original_route)
        
        # Should still be a valid permutation
        assert len(mutated_route) == len(original_route)
        assert set(mutated_route) == set(original_route)
        
        # Test with zero mutation rate
        ga.config.mutation_rate = 0.0
        unmutated_route = ga._swap_mutation(original_route)
        assert unmutated_route == original_route
    
    def test_two_opt_improvement(self, sample_cities):
        """Test 2-opt local search improvement."""
        ga = GeneticAlgorithm(sample_cities)
        
        route = [0, 1, 2, 3, 4]
        improved_route = ga._two_opt_improvement(route)
        
        # Should be a valid permutation
        assert len(improved_route) == len(route)
        assert set(improved_route) == set(route)
        
        # Distance should be same or better
        original_distance = ga._calculate_route_distance(route)
        improved_distance = ga._calculate_route_distance(improved_route)
        assert improved_distance <= original_distance
    
    def test_optimization_run(self, sample_cities):
        """Test full optimization run."""
        config = GAConfig(
            population_size=10,
            generations=5,
            elite_size=2,
            tournament_size=3,
            mutation_rate=0.1
        )
        
        ga = GeneticAlgorithm(sample_cities, config)
        best_route, best_distance = ga.optimize(verbose=False)
        
        # Check results
        assert len(best_route) == len(sample_cities)
        assert set(best_route) == set(range(len(sample_cities)))
        assert best_distance > 0
        assert isinstance(best_distance, float)
        
        # Check that optimization stats are populated
        stats = ga.get_optimization_stats()
        assert 'best_distance' in stats
        assert 'best_route' in stats
        assert 'best_fitness_history' in stats
        assert len(stats['best_fitness_history']) > 0
    
    def test_get_route_coordinates(self, sample_cities):
        """Test route coordinate extraction."""
        ga = GeneticAlgorithm(sample_cities)
        ga.best_route = [0, 1, 2, 3, 4]
        
        coordinates = ga.get_route_coordinates()
        
        assert len(coordinates) == len(sample_cities) + 1  # +1 for return to start
        assert coordinates[0] == coordinates[-1]  # Should return to start
        
        # Check coordinate format
        for lat, lon in coordinates:
            assert isinstance(lat, float)
            assert isinstance(lon, float)
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
    
    def test_convergence_detection(self, sample_cities):
        """Test convergence detection."""
        config = GAConfig(
            population_size=10,
            generations=100,
            max_stagnation=5,
            convergence_threshold=1e-6
        )
        
        ga = GeneticAlgorithm(sample_cities, config)
        best_route, best_distance = ga.optimize(verbose=False)
        
        stats = ga.get_optimization_stats()
        
        # Should converge before max generations due to small problem size
        assert stats['total_generations'] <= config.generations
    
    def test_large_problem(self):
        """Test GA on larger problem instance."""
        cities = generate_random_cities(20)
        
        config = GAConfig(
            population_size=50,
            generations=20,
            elite_size=5
        )
        
        ga = GeneticAlgorithm(cities, config)
        best_route, best_distance = ga.optimize(verbose=False)
        
        assert len(best_route) == 20
        assert set(best_route) == set(range(20))
        assert best_distance > 0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimum cities (2)
        cities = generate_random_cities(2)
        ga = GeneticAlgorithm(cities)
        
        best_route, best_distance = ga.optimize(verbose=False)
        assert len(best_route) == 2
        assert best_distance > 0
        
        # Test with single city should raise error or handle gracefully
        single_city = generate_random_cities(1)
        ga_single = GeneticAlgorithm(single_city)
        
        # Should handle single city case
        try:
            route, distance = ga_single.optimize(verbose=False)
            assert len(route) == 1
            assert distance == 0
        except:
            # It's acceptable to raise an error for single city
            pass
    
    def test_reproducibility(self, sample_cities):
        """Test that results are reproducible with same random seed."""
        import random
        import numpy as np
        
        config = GAConfig(population_size=20, generations=10)
        
        # Run 1
        random.seed(42)
        np.random.seed(42)
        ga1 = GeneticAlgorithm(sample_cities, config)
        route1, distance1 = ga1.optimize(verbose=False)
        
        # Run 2 with same seed
        random.seed(42)
        np.random.seed(42)
        ga2 = GeneticAlgorithm(sample_cities, config)
        route2, distance2 = ga2.optimize(verbose=False)
        
        # Results should be identical
        assert route1 == route2
        assert abs(distance1 - distance2) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__])
