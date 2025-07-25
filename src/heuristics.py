
"""
Alternative optimization algorithms for route optimization.
Includes nearest neighbor, 2-opt, simulated annealing, and other heuristics.
"""

import random
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class SAConfig:
    """Configuration for Simulated Annealing."""
    initial_temperature: float = 1000.0
    cooling_rate: float = 0.95
    min_temperature: float = 1.0
    max_iterations: int = 10000

@dataclass
class TwoOptConfig:
    """Configuration for 2-opt algorithm."""
    max_iterations: int = 1000
    improvement_threshold: float = 0.01

class RouteOptimizer:
    """Collection of route optimization algorithms."""
    
    def __init__(self, cities: List[Dict]):
        """
        Initialize the optimizer with city data.
        
        Args:
            cities: List of city dictionaries with 'latitude', 'longitude', 'name'
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all city pairs."""
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
    
    def nearest_neighbor(self, start_city: int = 0, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Nearest Neighbor heuristic for TSP.
        
        Args:
            start_city: Index of starting city
            verbose: Whether to print progress
            
        Returns:
            Tuple of (route, total_distance)
        """
        start_time = time.time()
        
        unvisited = set(range(self.num_cities))
        route = [start_city]
        unvisited.remove(start_city)
        current_city = start_city
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, 
                             key=lambda city: self.distance_matrix[current_city][city])
            route.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        total_distance = self._calculate_route_distance(route)
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"Nearest Neighbor completed in {execution_time:.3f} seconds")
            print(f"Route distance: {total_distance:.2f} km")
        
        return route, total_distance
    
    def two_opt(self, initial_route: Optional[List[int]] = None, 
                config: Optional[TwoOptConfig] = None, verbose: bool = True) -> Tuple[List[int], float]:
        """
        2-opt local search improvement algorithm.
        
        Args:
            initial_route: Starting route (uses nearest neighbor if None)
            config: 2-opt configuration
            verbose: Whether to print progress
            
        Returns:
            Tuple of (improved_route, total_distance)
        """
        start_time = time.time()
        config = config or TwoOptConfig()
        
        if initial_route is None:
            initial_route, _ = self.nearest_neighbor(verbose=False)
        
        best_route = initial_route.copy()
        best_distance = self._calculate_route_distance(best_route)
        
        if verbose:
            print(f"2-opt starting with distance: {best_distance:.2f} km")
        
        iteration = 0
        improved = True
        
        while improved and iteration < config.max_iterations:
            improved = False
            
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route)):
                    if j - i == 1: continue  # Skip adjacent edges
                    
                    # Create new route by reversing segment between i and j
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    new_distance = self._calculate_route_distance(new_route)
                    
                    # Check for improvement
                    if new_distance < best_distance - config.improvement_threshold:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        
                        if verbose and iteration % 100 == 0:
                            print(f"Iteration {iteration}: New best distance = {best_distance:.2f} km")
            
            iteration += 1
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"2-opt completed in {execution_time:.3f} seconds after {iteration} iterations")
            print(f"Final distance: {best_distance:.2f} km")
        
        return best_route, best_distance
    
    def simulated_annealing(self, initial_route: Optional[List[int]] = None,
                          config: Optional[SAConfig] = None, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Simulated Annealing algorithm for TSP.
        
        Args:
            initial_route: Starting route (uses nearest neighbor if None)
            config: SA configuration parameters
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_route, best_distance)
        """
        start_time = time.time()
        config = config or SAConfig()
        
        if initial_route is None:
            initial_route, _ = self.nearest_neighbor(verbose=False)
        
        current_route = initial_route.copy()
        current_distance = self._calculate_route_distance(current_route)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = config.initial_temperature
        
        if verbose:
            print(f"Simulated Annealing starting with distance: {current_distance:.2f} km")
        
        for iteration in range(config.max_iterations):
            # Generate neighbor by swapping two random cities
            neighbor_route = current_route.copy()
            i, j = random.sample(range(len(neighbor_route)), 2)
            neighbor_route[i], neighbor_route[j] = neighbor_route[j], neighbor_route[i]
            
            neighbor_distance = self._calculate_route_distance(neighbor_route)
            
            # Accept or reject the neighbor
            delta = neighbor_distance - current_distance
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = neighbor_route
                current_distance = neighbor_distance
                
                # Update best solution
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            # Cool down
            temperature *= config.cooling_rate
            
            if temperature < config.min_temperature:
                break
            
            if verbose and iteration % 1000 == 0:
                print(f"Iteration {iteration}: Current = {current_distance:.2f} km, "
                      f"Best = {best_distance:.2f} km, Temp = {temperature:.2f}")
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"Simulated Annealing completed in {execution_time:.3f} seconds")
            print(f"Final best distance: {best_distance:.2f} km")
        
        return best_route, best_distance
    
    def random_search(self, num_iterations: int = 10000, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Random search baseline algorithm.
        
        Args:
            num_iterations: Number of random routes to try
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_route, best_distance)
        """
        start_time = time.time()
        
        city_indices = list(range(self.num_cities))
        best_route = city_indices.copy()
        best_distance = self._calculate_route_distance(best_route)
        
        if verbose:
            print(f"Random search starting with {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            route = city_indices.copy()
            random.shuffle(route)
            distance = self._calculate_route_distance(route)
            
            if distance < best_distance:
                best_route = route
                best_distance = distance
                
                if verbose and iteration % 1000 == 0:
                    print(f"Iteration {iteration}: New best distance = {best_distance:.2f} km")
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"Random search completed in {execution_time:.3f} seconds")
            print(f"Best distance found: {best_distance:.2f} km")
        
        return best_route, best_distance
    
    def christofides_approximation(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Christofides algorithm approximation for TSP.
        Provides a 1.5-approximation to the optimal solution.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Tuple of (route, total_distance)
        """
        start_time = time.time()
        
        if verbose:
            print("Running Christofides approximation algorithm")
        
        # Step 1: Find minimum spanning tree using Prim's algorithm
        mst_edges = self._minimum_spanning_tree()
        
        # Step 2: Find vertices with odd degree in MST
        odd_vertices = self._find_odd_degree_vertices(mst_edges)
        
        # Step 3: Find minimum weight perfect matching on odd vertices
        matching_edges = self._minimum_weight_matching(odd_vertices)
        
        # Step 4: Combine MST and matching to form Eulerian graph
        eulerian_edges = mst_edges + matching_edges
        
        # Step 5: Find Eulerian tour
        eulerian_tour = self._find_eulerian_tour(eulerian_edges)
        
        # Step 6: Convert to Hamiltonian tour by skipping repeated vertices
        hamiltonian_tour = self._eulerian_to_hamiltonian(eulerian_tour)
        
        total_distance = self._calculate_route_distance(hamiltonian_tour)
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"Christofides completed in {execution_time:.3f} seconds")
            print(f"Route distance: {total_distance:.2f} km")
        
        return hamiltonian_tour, total_distance
    
    def _minimum_spanning_tree(self) -> List[Tuple[int, int]]:
        """Find minimum spanning tree using Prim's algorithm."""
        visited = {0}
        edges = []
        
        while len(visited) < self.num_cities:
            min_edge = None
            min_weight = float('inf')
            
            for u in visited:
                for v in range(self.num_cities):
                    if v not in visited and self.distance_matrix[u][v] < min_weight:
                        min_weight = self.distance_matrix[u][v]
                        min_edge = (u, v)
            
            if min_edge:
                edges.append(min_edge)
                visited.add(min_edge[1])
        
        return edges
    
    def _find_odd_degree_vertices(self, edges: List[Tuple[int, int]]) -> List[int]:
        """Find vertices with odd degree in the graph."""
        degree = [0] * self.num_cities
        
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        return [i for i in range(self.num_cities) if degree[i] % 2 == 1]
    
    def _minimum_weight_matching(self, vertices: List[int]) -> List[Tuple[int, int]]:
        """Find minimum weight perfect matching (simplified greedy approach)."""
        vertices = vertices.copy()
        matching = []
        
        while len(vertices) >= 2:
            min_weight = float('inf')
            best_pair = None
            
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    weight = self.distance_matrix[vertices[i]][vertices[j]]
                    if weight < min_weight:
                        min_weight = weight
                        best_pair = (i, j)
            
            if best_pair:
                u_idx, v_idx = best_pair
                u, v = vertices[u_idx], vertices[v_idx]
                matching.append((u, v))
                
                # Remove matched vertices (remove higher index first)
                vertices.pop(max(u_idx, v_idx))
                vertices.pop(min(u_idx, v_idx))
        
        return matching
    
    def _find_eulerian_tour(self, edges: List[Tuple[int, int]]) -> List[int]:
        """Find Eulerian tour in the graph (simplified implementation)."""
        # Build adjacency list
        graph = {i: [] for i in range(self.num_cities)}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Start from vertex 0
        tour = []
        stack = [0]
        
        while stack:
            curr = stack[-1]
            if graph[curr]:
                next_vertex = graph[curr].pop()
                graph[next_vertex].remove(curr)
                stack.append(next_vertex)
            else:
                tour.append(stack.pop())
        
        return tour[::-1]
    
    def _eulerian_to_hamiltonian(self, eulerian_tour: List[int]) -> List[int]:
        """Convert Eulerian tour to Hamiltonian tour by skipping repeated vertices."""
        visited = set()
        hamiltonian_tour = []
        
        for vertex in eulerian_tour:
            if vertex not in visited:
                hamiltonian_tour.append(vertex)
                visited.add(vertex)
        
        return hamiltonian_tour
    
    def compare_algorithms(self, verbose: bool = True) -> Dict:
        """
        Compare performance of different optimization algorithms.
        
        Args:
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with algorithm comparison results
        """
        results = {}
        
        if verbose:
            print("Comparing route optimization algorithms...")
            print("=" * 50)
        
        # Nearest Neighbor
        route_nn, dist_nn = self.nearest_neighbor(verbose=verbose)
        results['nearest_neighbor'] = {
            'route': route_nn,
            'distance': dist_nn,
            'route_names': [self.cities[i]['name'] for i in route_nn]
        }
        
        if verbose:
            print("-" * 50)
        
        # 2-opt improvement
        route_2opt, dist_2opt = self.two_opt(initial_route=route_nn, verbose=verbose)
        results['two_opt'] = {
            'route': route_2opt,
            'distance': dist_2opt,
            'route_names': [self.cities[i]['name'] for i in route_2opt]
        }
        
        if verbose:
            print("-" * 50)
        
        # Simulated Annealing
        route_sa, dist_sa = self.simulated_annealing(initial_route=route_nn, verbose=verbose)
        results['simulated_annealing'] = {
            'route': route_sa,
            'distance': dist_sa,
            'route_names': [self.cities[i]['name'] for i in route_sa]
        }
        
        if verbose:
            print("-" * 50)
        
        # Random Search (shorter run for comparison)
        route_random, dist_random = self.random_search(num_iterations=1000, verbose=verbose)
        results['random_search'] = {
            'route': route_random,
            'distance': dist_random,
            'route_names': [self.cities[i]['name'] for i in route_random]
        }
        
        if verbose:
            print("\nAlgorithm Comparison Summary:")
            print("=" * 50)
            for alg_name, result in results.items():
                print(f"{alg_name:20}: {result['distance']:8.2f} km")
        
        return results
