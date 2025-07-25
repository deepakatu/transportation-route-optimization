
"""
Unit tests for utility functions.
"""

import pytest
import json
import csv
import tempfile
import os
import numpy as np
from src.utils import (
    load_cities, save_cities, load_config, get_default_config,
    haversine_distance, calculate_distance_matrix, validate_route,
    calculate_route_distance, generate_random_cities, export_route_to_gpx,
    save_optimization_results, load_optimization_results, PerformanceTimer
)

class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @pytest.fixture
    def sample_cities(self):
        """Create sample cities for testing."""
        return [
            {'id': 0, 'name': 'New York', 'latitude': 40.7128, 'longitude': -74.0060},
            {'id': 1, 'name': 'Los Angeles', 'latitude': 34.0522, 'longitude': -118.2437},
            {'id': 2, 'name': 'Chicago', 'latitude': 41.8781, 'longitude': -87.6298}
        ]
    
    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        # Test distance between New York and Los Angeles
        distance = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < distance < 4000  # Approximate known distance
        
        # Test distance from a point to itself
        distance = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert distance == 0
        
        # Test distance symmetry
        dist1 = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        dist2 = haversine_distance(34.0522, -118.2437, 40.7128, -74.0060)
        assert abs(dist1 - dist2) < 1e-10
    
    def test_load_cities_json(self, sample_cities):
        """Test loading cities from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'cities': sample_cities}, f)
            temp_path = f.name
        
        try:
            loaded_cities = load_cities(temp_path)
            assert len(loaded_cities) == len(sample_cities)
            
            for original, loaded in zip(sample_cities, loaded_cities):
                assert loaded['name'] == original['name']
                assert loaded['latitude'] == original['latitude']
                assert loaded['longitude'] == original['longitude']
        finally:
            os.unlink(temp_path)
    
    def test_load_cities_csv(self, sample_cities):
        """Test loading cities from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'name', 'latitude', 'longitude'])
            writer.writeheader()
            writer.writerows(sample_cities)
            temp_path = f.name
        
        try:
            loaded_cities = load_cities(temp_path)
            assert len(loaded_cities) == len(sample_cities)
            
            for original, loaded in zip(sample_cities, loaded_cities):
                assert loaded['name'] == original['name']
                assert float(loaded['latitude']) == original['latitude']
                assert float(loaded['longitude']) == original['longitude']
        finally:
            os.unlink(temp_path)
    
    def test_load_cities_file_not_found(self):
        """Test loading cities from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_cities('non_existent_file.json')
    
    def test_load_cities_invalid_format(self):
        """Test loading cities from invalid file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_cities(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_cities_json(self, sample_cities):
        """Test saving cities to JSON file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_cities(sample_cities, temp_path)
            
            # Verify file was created and contains correct data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'cities' in data
            assert 'metadata' in data
            assert len(data['cities']) == len(sample_cities)
            
            for original, saved in zip(sample_cities, data['cities']):
                assert saved['name'] == original['name']
                assert saved['latitude'] == original['latitude']
                assert saved['longitude'] == original['longitude']
        finally:
            os.unlink(temp_path)
    
    def test_save_cities_csv(self, sample_cities):
        """Test saving cities to CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_cities(sample_cities, temp_path)
            
            # Verify file was created and contains correct data
            loaded_cities = load_cities(temp_path)
            assert len(loaded_cities) == len(sample_cities)
        finally:
            os.unlink(temp_path)
    
    def test_get_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        
        assert 'algorithms' in config
        assert 'visualization' in config
        assert 'data' in config
        
        assert 'genetic_algorithm' in config['algorithms']
        assert 'simulated_annealing' in config['algorithms']
        assert 'two_opt' in config['algorithms']
        
        # Check some specific values
        ga_config = config['algorithms']['genetic_algorithm']
        assert ga_config['population_size'] == 100
        assert ga_config['generations'] == 500
    
    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        config = load_config('non_existent_config.yaml')
        
        # Should return default config
        default_config = get_default_config()
        assert config == default_config
    
    def test_calculate_distance_matrix(self, sample_cities):
        """Test distance matrix calculation."""
        matrix = calculate_distance_matrix(sample_cities)
        
        assert matrix.shape == (3, 3)
        assert np.all(matrix >= 0)
        assert np.all(np.diag(matrix) == 0)
        
        # Check symmetry
        assert np.allclose(matrix, matrix.T)
        
        # Check specific distance (NY to LA)
        ny_la_distance = matrix[0][1]
        assert 3900 < ny_la_distance < 4000
    
    def test_calculate_distance_matrix_with_cache(self, sample_cities):
        """Test distance matrix calculation with caching."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cache_path = f.name
        
        try:
            # First calculation - should create cache
            matrix1 = calculate_distance_matrix(sample_cities, cache_file=cache_path)
            assert os.path.exists(cache_path)
            
            # Second calculation - should load from cache
            matrix2 = calculate_distance_matrix(sample_cities, cache_file=cache_path)
            
            assert np.allclose(matrix1, matrix2)
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_validate_route(self):
        """Test route validation."""
        # Valid route
        assert validate_route([0, 1, 2, 3, 4], 5) == True
        assert validate_route([4, 2, 0, 3, 1], 5) == True
        
        # Invalid routes
        assert validate_route([0, 1, 2], 5) == False  # Wrong length
        assert validate_route([0, 1, 2, 3, 5], 5) == False  # Invalid city index
        assert validate_route([0, 1, 2, 2, 3], 5) == False  # Duplicate city
        assert validate_route([0, 1, 2, 3, 4, 5], 5) == False  # Too many cities
    
    def test_calculate_route_distance(self, sample_cities):
        """Test route distance calculation."""
        distance_matrix = calculate_distance_matrix(sample_cities)
        
        route = [0, 1, 2]
        distance = calculate_route_distance(route, distance_matrix)
        
        # Manual calculation
        expected_distance = (distance_matrix[0][1] + 
                           distance_matrix[1][2] + 
                           distance_matrix[2][0])
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_generate_random_cities(self):
        """Test random city generation."""
        cities = generate_random_cities(10)
        
        assert len(cities) == 10
        
        for i, city in enumerate(cities):
            assert city['id'] == i
            assert city['name'] == f'City_{i:03d}'
            assert 'latitude' in city
            assert 'longitude' in city
            assert -90 <= city['latitude'] <= 90
            assert -180 <= city['longitude'] <= 180
    
    def test_generate_random_cities_with_bounds(self):
        """Test random city generation with custom bounds."""
        bounds = (30.0, 40.0, -100.0, -80.0)
        cities = generate_random_cities(5, bounds)
        
        assert len(cities) == 5
        
        for city in cities:
            assert 30.0 <= city['latitude'] <= 40.0
            assert -100.0 <= city['longitude'] <= -80.0
    
    def test_export_route_to_gpx(self, sample_cities):
        """Test GPX export functionality."""
        route = [0, 1, 2]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gpx', delete=False) as f:
            temp_path = f.name
        
        try:
            export_route_to_gpx(route, sample_cities, temp_path)
            
            # Verify file was created and contains GPX content
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert '<?xml version="1.0" encoding="UTF-8"?>' in content
            assert '<gpx version="1.1"' in content
            assert '<trk>' in content
            assert '<trkpt' in content
            
            # Check that all cities are included
            for city in sample_cities:
                assert city['name'] in content
        finally:
            os.unlink(temp_path)
    
    def test_save_and_load_optimization_results(self):
        """Test saving and loading optimization results."""
        results = {
            'algorithm': 'genetic_algorithm',
            'best_route': [0, 1, 2, 3, 4],
            'best_distance': 1234.56,
            'fitness_history': [0.1, 0.2, 0.3],
            'numpy_array': np.array([1, 2, 3])
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_optimization_results(results, temp_path)
            loaded_results = load_optimization_results(temp_path)
            
            assert loaded_results['algorithm'] == results['algorithm']
            assert loaded_results['best_route'] == results['best_route']
            assert loaded_results['best_distance'] == results['best_distance']
            assert loaded_results['fitness_history'] == results['fitness_history']
            assert loaded_results['numpy_array'] == [1, 2, 3]  # Converted to list
            assert 'metadata' in loaded_results
        finally:
            os.unlink(temp_path)
    
    def test_performance_timer(self):
        """Test performance timer context manager."""
        import time
        
        with PerformanceTimer("Test operation") as timer:
            time.sleep(0.1)  # Sleep for 100ms
        
        assert timer.duration >= 0.1
        assert timer.duration < 0.2  # Should be close to 0.1 seconds
    
    def test_performance_timer_duration_property(self):
        """Test performance timer duration property."""
        timer = PerformanceTimer("Test")
        
        # Before entering context, duration should be 0
        assert timer.duration == 0.0
        
        with timer:
            pass
        
        # After exiting context, duration should be > 0
        assert timer.duration > 0.0
    
    def test_cities_data_validation(self):
        """Test city data validation during loading."""
        # Test with missing required fields
        invalid_cities = [
            {'name': 'City A', 'latitude': 40.0},  # Missing longitude
            {'latitude': 41.0, 'longitude': -87.0},  # Missing name
            {'name': 'City C', 'latitude': 'invalid', 'longitude': -95.0}  # Invalid latitude
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'cities': invalid_cities}, f)
            temp_path = f.name
        
        try:
            # Should skip invalid cities and only load valid ones
            loaded_cities = load_cities(temp_path)
            assert len(loaded_cities) == 0  # All cities are invalid
        except ValueError:
            # It's also acceptable to raise an error for no valid cities
            pass
        finally:
            os.unlink(temp_path)
    
    def test_cities_data_validation_partial(self):
        """Test city data validation with some valid cities."""
        mixed_cities = [
            {'name': 'Valid City', 'latitude': 40.0, 'longitude': -74.0},
            {'name': 'Invalid City', 'latitude': 'invalid', 'longitude': -87.0},
            {'name': 'Another Valid', 'latitude': 41.0, 'longitude': -95.0}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'cities': mixed_cities}, f)
            temp_path = f.name
        
        try:
            loaded_cities = load_cities(temp_path)
            assert len(loaded_cities) == 2  # Only valid cities loaded
            
            for city in loaded_cities:
                assert 'name' in city
                assert 'latitude' in city
                assert 'longitude' in city
                assert isinstance(city['latitude'], float)
                assert isinstance(city['longitude'], float)
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__])
