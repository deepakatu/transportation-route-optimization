
"""
Utility functions for the transportation route optimization system.
"""

import json
import csv
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import os
import logging
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cities(file_path: str) -> List[Dict]:
    """
    Load city data from JSON or CSV file.
    
    Args:
        file_path: Path to the city data file
        
    Returns:
        List of city dictionaries with required fields
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported or data is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"City data file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'cities' in data:
                    cities = data['cities']
                else:
                    cities = data
        
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
            cities = df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Validate city data
        validated_cities = []
        for i, city in enumerate(cities):
            if not all(key in city for key in ['name', 'latitude', 'longitude']):
                logger.warning(f"City {i} missing required fields, skipping")
                continue
            
            # Ensure numeric coordinates
            try:
                city['latitude'] = float(city['latitude'])
                city['longitude'] = float(city['longitude'])
                city['id'] = city.get('id', i)
                validated_cities.append(city)
            except (ValueError, TypeError):
                logger.warning(f"Invalid coordinates for city {i}, skipping")
                continue
        
        if not validated_cities:
            raise ValueError("No valid cities found in the data file")
        
        logger.info(f"Loaded {len(validated_cities)} cities from {file_path}")
        return validated_cities
        
    except Exception as e:
        raise ValueError(f"Error loading city data: {str(e)}")

def save_cities(cities: List[Dict], file_path: str) -> None:
    """
    Save city data to JSON or CSV file.
    
    Args:
        cities: List of city dictionaries
        file_path: Path to save the file
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_extension == '.json':
        data = {
            'cities': cities,
            'metadata': {
                'total_cities': len(cities),
                'created_at': datetime.now().isoformat(),
                'coordinate_system': 'WGS84'
            }
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif file_extension == '.csv':
        df = pd.DataFrame(cities)
        df.to_csv(file_path, index=False)
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    logger.info(f"Saved {len(cities)} cities to {file_path}")

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration parameters."""
    return {
        'algorithms': {
            'genetic_algorithm': {
                'population_size': 100,
                'generations': 500,
                'elite_size': 5,
                'tournament_size': 5,
                'mutation_rate': 0.02,
                'crossover_rate': 0.8
            },
            'simulated_annealing': {
                'initial_temperature': 1000,
                'cooling_rate': 0.95,
                'min_temperature': 1,
                'max_iterations': 10000
            },
            'two_opt': {
                'max_iterations': 1000,
                'improvement_threshold': 0.01
            }
        },
        'visualization': {
            'map_center': [40.7128, -74.0060],
            'zoom_level': 10,
            'route_color': "#FF0000",
            'city_marker_color': "#0000FF"
        },
        'data': {
            'default_cities_file': "data/sample_cities.json",
            'cache_distance_matrix': True,
            'distance_unit': "km"
        }
    }

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def calculate_distance_matrix(cities: List[Dict], cache_file: Optional[str] = None) -> np.ndarray:
    """
    Calculate distance matrix between all city pairs.
    
    Args:
        cities: List of city dictionaries
        cache_file: Optional file to cache the distance matrix
        
    Returns:
        Distance matrix as numpy array
    """
    # Try to load from cache
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == len(cities):
                    logger.info(f"Loaded distance matrix from cache: {cache_file}")
                    return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cached distance matrix: {str(e)}")
    
    # Calculate distance matrix
    n = len(cities)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(
                cities[i]['latitude'], cities[i]['longitude'],
                cities[j]['latitude'], cities[j]['longitude']
            )
            matrix[i][j] = matrix[j][i] = dist
    
    # Save to cache
    if cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(matrix, f)
            logger.info(f"Cached distance matrix to: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache distance matrix: {str(e)}")
    
    return matrix

def validate_route(route: List[int], num_cities: int) -> bool:
    """
    Validate that a route is a valid permutation of city indices.
    
    Args:
        route: List of city indices
        num_cities: Total number of cities
        
    Returns:
        True if route is valid, False otherwise
    """
    if len(route) != num_cities:
        return False
    
    if set(route) != set(range(num_cities)):
        return False
    
    return True

def calculate_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
    """
    Calculate total distance for a route using precomputed distance matrix.
    
    Args:
        route: List of city indices in visit order
        distance_matrix: Precomputed distance matrix
        
    Returns:
        Total route distance
    """
    total_distance = 0
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        total_distance += distance_matrix[from_city][to_city]
    return total_distance

def generate_random_cities(num_cities: int, bounds: Tuple[float, float, float, float] = None) -> List[Dict]:
    """
    Generate random cities for testing purposes.
    
    Args:
        num_cities: Number of cities to generate
        bounds: (min_lat, max_lat, min_lon, max_lon) boundaries
        
    Returns:
        List of randomly generated city dictionaries
    """
    if bounds is None:
        # Default to continental US bounds
        bounds = (25.0, 49.0, -125.0, -66.0)
    
    min_lat, max_lat, min_lon, max_lon = bounds
    
    cities = []
    for i in range(num_cities):
        city = {
            'id': i,
            'name': f'City_{i:03d}',
            'latitude': np.random.uniform(min_lat, max_lat),
            'longitude': np.random.uniform(min_lon, max_lon)
        }
        cities.append(city)
    
    logger.info(f"Generated {num_cities} random cities")
    return cities

def export_route_to_gpx(route: List[int], cities: List[Dict], file_path: str) -> None:
    """
    Export route to GPX format for GPS devices.
    
    Args:
        route: List of city indices in visit order
        cities: List of city dictionaries
        file_path: Path to save GPX file
    """
    gpx_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Route Optimizer">
<trk>
<name>Optimized Route</name>
<trkseg>
'''
    
    for city_idx in route:
        city = cities[city_idx]
        gpx_content += f'<trkpt lat="{city["latitude"]}" lon="{city["longitude"]}"><name>{city["name"]}</name></trkpt>\n'
    
    # Add return to start
    if route:
        start_city = cities[route[0]]
        gpx_content += f'<trkpt lat="{start_city["latitude"]}" lon="{start_city["longitude"]}"><name>{start_city["name"]}</name></trkpt>\n'
    
    gpx_content += '''</trkseg>
</trk>
</gpx>'''
    
    with open(file_path, 'w') as f:
        f.write(gpx_content)
    
    logger.info(f"Exported route to GPX: {file_path}")

def save_optimization_results(results: Dict, file_path: str) -> None:
    """
    Save optimization results to JSON file.
    
    Args:
        results: Dictionary containing optimization results
        file_path: Path to save results
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = value
    
    # Add metadata
    serializable_results['metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved optimization results to: {file_path}")

def load_optimization_results(file_path: str) -> Dict:
    """
    Load optimization results from JSON file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Dictionary containing optimization results
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded optimization results from: {file_path}")
    return results

class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.description} completed in {duration:.3f} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
