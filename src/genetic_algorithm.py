
"""
Genetic Algorithm implementation for Transportation Route Optimization.
Solves the Traveling Salesman Problem using evolutionary computation.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import copy
from dataclasses import dataclass

@dataclass
class GAConfig:
    """Configuration parameters for Genetic Algorithm."""
    population_size: int = 100
    generations: int = 500
    elite_size: int = 5
    tournament_size: int = 5
    mutation_rate: float = 0.02
    crossover_rate: float = 0.8
    convergence_threshold: float = 1e-6
    max_stagnation: int = 50

class GeneticAlgorithm:
    """
    Genetic Algorithm for solving the Traveling Salesman Problem.
    
    Uses tournament selection, order crossover (OX1), and swap mutation
    with elitism to find optimal or near-optimal routes.
    """
    
    def __init__(self, cities: List[Dict], config: Optional[GAConfig] = None):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            cities: List of city dictionaries with 'latitude', 'longitude', 'name'
            config: GA configuration parameters
        """
        self.cities = cities
        self.config = config or GAConfig()
        self.num_cities = len(cities)
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Evolution tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_times = []
        self.best_route = None
        self.best_distance = float('inf')
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all city pairs using Haversine formula."""
        n = len(self.cities)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._haversine_distance(
                    self.cities[i]['latitude'], self.cities[i]['longitude'],
                    self.cities[j]['latitude'], self.cities[j]['longitude']
                )
                matrix[i][j] = matrix[j][i] = dist
                
        return matrix
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a given route."""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
    
    def _fitness(self, route: List[int]) -> float:
        """Calculate fitness (inverse of distance) for a route."""
        distance = self._calculate_route_distance(route)
        return 1 / (1 + distance)  # Higher fitness for shorter routes
    
    def _initialize_population(self) -> List[List[int]]:
        """Create initial population with random routes."""
        population = []
        city_indices = list(range(self.num_cities))
        
        for _ in range(self.config.population_size):
            route = city_indices.copy()
            random.shuffle(route)
            population.append(route)
            
        return population
    
    def _tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Select parent using tournament selection."""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=self._fitness)
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform Order Crossover (OX1) to create offspring.
        Preserves the relative order of cities from parents.
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child with segment from parent1
        child = [None] * size
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining positions with cities from parent2 in order
        remaining = [city for city in parent2 if city not in child]
        remaining_idx = 0
        
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[remaining_idx]
                remaining_idx += 1
                
        return child
    
    def _swap_mutation(self, route: List[int]) -> List[int]:
        """Perform swap mutation by exchanging two random cities."""
        if random.random() < self.config.mutation_rate:
            route = route.copy()
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def _two_opt_improvement(self, route: List[int]) -> List[int]:
        """Apply 2-opt local search improvement."""
        best_route = route.copy()
        best_distance = self._calculate_route_distance(best_route)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue  # Skip adjacent edges
                    
                    # Create new route by reversing segment between i and j
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        
            route = best_route
            
        return best_route
    
    def optimize(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run the genetic algorithm optimization.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (best_route, best_distance)
        """
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        stagnation_counter = 0
        previous_best = float('inf')
        
        if verbose:
            print(f"Starting GA optimization with {self.num_cities} cities")
            print(f"Population size: {self.config.population_size}, Generations: {self.config.generations}")
        
        for generation in range(self.config.generations):
            gen_start_time = time.time()
            
            # Evaluate fitness and sort population
            population.sort(key=self._fitness, reverse=True)
            
            # Track statistics
            current_best_distance = self._calculate_route_distance(population[0])
            avg_fitness = np.mean([self._fitness(route) for route in population])
            
            self.best_fitness_history.append(1 / (1 + current_best_distance))
            self.avg_fitness_history.append(avg_fitness)
            
            # Update global best
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_route = population[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Check for convergence
            if abs(previous_best - current_best_distance) < self.config.convergence_threshold:
                stagnation_counter += 1
            
            if stagnation_counter >= self.config.max_stagnation:
                if verbose:
                    print(f"Converged at generation {generation}")
                break
            
            previous_best = current_best_distance
            
            # Create new population
            new_population = population[:self.config.elite_size]  # Elitism
            
            while len(new_population) < self.config.population_size:
                if random.random() < self.config.crossover_rate:
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = self._tournament_selection(population).copy()
                
                child = self._swap_mutation(child)
                
                # Apply 2-opt improvement occasionally
                if random.random() < 0.1:
                    child = self._two_opt_improvement(child)
                
                new_population.append(child)
            
            population = new_population
            
            gen_time = time.time() - gen_start_time
            self.generation_times.append(gen_time)
            
            if verbose and generation % 50 == 0:
                print(f"Generation {generation:3d}: Best distance = {current_best_distance:.2f} km, "
                      f"Avg fitness = {avg_fitness:.6f}, Time = {gen_time:.3f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            print(f"Best route distance: {self.best_distance:.2f} km")
            print(f"Best route: {[self.cities[i]['name'] for i in self.best_route]}")
        
        return self.best_route, self.best_distance
    
    def get_optimization_stats(self) -> Dict:
        """Get detailed optimization statistics."""
        return {
            'best_distance': self.best_distance,
            'best_route': self.best_route,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generation_times': self.generation_times,
            'total_generations': len(self.best_fitness_history),
            'avg_generation_time': np.mean(self.generation_times) if self.generation_times else 0,
            'convergence_generation': len(self.best_fitness_history)
        }
    
    def get_route_coordinates(self, route: Optional[List[int]] = None) -> List[Tuple[float, float]]:
        """Get latitude/longitude coordinates for a route."""
        if route is None:
            route = self.best_route
        
        if route is None:
            return []
        
        coordinates = []
        for city_idx in route:
            city = self.cities[city_idx]
            coordinates.append((city['latitude'], city['longitude']))
        
        # Add return to start
        if coordinates:
            coordinates.append(coordinates[0])
            
        return coordinates
