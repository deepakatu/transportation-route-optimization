
"""
Main CLI interface for the Transportation Route Optimization system.
"""

import argparse
import sys
import os
from typing import Optional, List, Dict
import json

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm import GeneticAlgorithm, GAConfig
from heuristics import RouteOptimizer, SAConfig, TwoOptConfig
from visualization import RouteVisualizer
from utils import load_cities, load_config, save_optimization_results, PerformanceTimer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transportation Route Optimization using various algorithms"
    )
    
    parser.add_argument(
        '--cities', '-c',
        type=str,
        default='data/sample_cities.json',
        help='Path to cities data file (JSON or CSV)'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['ga', 'nn', '2opt', 'sa', 'random', 'christofides'],
        default='ga',
        help='Optimization algorithm to use'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple algorithms'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results and visualizations'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization maps'
    )
    
    parser.add_argument(
        '--export-gpx',
        action='store_true',
        help='Export route to GPX format'
    )
    
    # Algorithm-specific parameters
    parser.add_argument(
        '--population-size',
        type=int,
        help='Population size for genetic algorithm'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        help='Number of generations for genetic algorithm'
    )
    
    parser.add_argument(
        '--mutation-rate',
        type=float,
        help='Mutation rate for genetic algorithm'
    )
    
    return parser.parse_args()

def run_genetic_algorithm(cities: List[Dict], config: Dict, args) -> tuple:
    """Run genetic algorithm optimization."""
    ga_config = GAConfig(
        population_size=args.population_size or config['algorithms']['genetic_algorithm']['population_size'],
        generations=args.generations or config['algorithms']['genetic_algorithm']['generations'],
        elite_size=config['algorithms']['genetic_algorithm']['elite_size'],
        tournament_size=config['algorithms']['genetic_algorithm']['tournament_size'],
        mutation_rate=args.mutation_rate or config['algorithms']['genetic_algorithm']['mutation_rate'],
        crossover_rate=config['algorithms']['genetic_algorithm']['crossover_rate']
    )
    
    ga = GeneticAlgorithm(cities, ga_config)
    
    with PerformanceTimer("Genetic Algorithm Optimization"):
        best_route, best_distance = ga.optimize(verbose=args.verbose)
    
    stats = ga.get_optimization_stats()
    return best_route, best_distance, stats

def run_single_algorithm(cities: List[Dict], algorithm: str, config: Dict, args) -> tuple:
    """Run a single optimization algorithm."""
    optimizer = RouteOptimizer(cities)
    
    if algorithm == 'ga':
        return run_genetic_algorithm(cities, config, args)
    
    elif algorithm == 'nn':
        with PerformanceTimer("Nearest Neighbor"):
            route, distance = optimizer.nearest_neighbor(verbose=args.verbose)
        return route, distance, {}
    
    elif algorithm == '2opt':
        two_opt_config = TwoOptConfig(
            max_iterations=config['algorithms']['two_opt']['max_iterations'],
            improvement_threshold=config['algorithms']['two_opt']['improvement_threshold']
        )
        with PerformanceTimer("2-opt Optimization"):
            route, distance = optimizer.two_opt(config=two_opt_config, verbose=args.verbose)
        return route, distance, {}
    
    elif algorithm == 'sa':
        sa_config = SAConfig(
            initial_temperature=config['algorithms']['simulated_annealing']['initial_temperature'],
            cooling_rate=config['algorithms']['simulated_annealing']['cooling_rate'],
            min_temperature=config['algorithms']['simulated_annealing']['min_temperature'],
            max_iterations=config['algorithms']['simulated_annealing']['max_iterations']
        )
        with PerformanceTimer("Simulated Annealing"):
            route, distance = optimizer.simulated_annealing(config=sa_config, verbose=args.verbose)
        return route, distance, {}
    
    elif algorithm == 'random':
        with PerformanceTimer("Random Search"):
            route, distance = optimizer.random_search(num_iterations=10000, verbose=args.verbose)
        return route, distance, {}
    
    elif algorithm == 'christofides':
        with PerformanceTimer("Christofides Algorithm"):
            route, distance = optimizer.christofides_approximation(verbose=args.verbose)
        return route, distance, {}
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def compare_algorithms(cities: List[Dict], config: Dict, args) -> Dict:
    """Compare multiple optimization algorithms."""
    print("Running algorithm comparison...")
    print("=" * 60)
    
    algorithms = ['nn', '2opt', 'sa', 'ga']
    results = {}
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm.upper()}...")
        print("-" * 40)
        
        try:
            route, distance, stats = run_single_algorithm(cities, algorithm, config, args)
            results[algorithm] = {
                'route': route,
                'distance': distance,
                'stats': stats,
                'route_names': [cities[i]['name'] for i in route]
            }
        except Exception as e:
            print(f"Error running {algorithm}: {str(e)}")
            continue
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 60)
    
    if results:
        best_distance = min(result['distance'] for result in results.values())
        
        for algorithm, result in results.items():
            efficiency = (best_distance / result['distance']) * 100
            print(f"{algorithm.upper():15}: {result['distance']:8.2f} km ({efficiency:5.1f}% efficiency)")
    
    return results

def generate_visualizations(cities: List[Dict], results: Dict, output_dir: str):
    """Generate visualization maps and charts."""
    visualizer = RouteVisualizer(cities)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if len(results) == 1:
        # Single algorithm visualization
        algorithm, result = next(iter(results.items()))
        route = result['route']
        distance = result['distance']
        
        # Create route map
        map_path = os.path.join(output_dir, f'{algorithm}_route_map.html')
        route_map = visualizer.create_route_map(
            route, 
            title=f"{algorithm.upper()} Optimized Route ({distance:.2f} km)",
            save_path=map_path
        )
        print(f"Route map saved to: {map_path}")
        
        # Create optimization progress chart (if available)
        if 'stats' in result and 'best_fitness_history' in result['stats']:
            stats = result['stats']
            progress_fig = visualizer.plot_optimization_progress(
                stats['best_fitness_history'],
                stats['avg_fitness_history'],
                title=f"{algorithm.upper()} Optimization Progress"
            )
            progress_path = os.path.join(output_dir, f'{algorithm}_progress.html')
            progress_fig.write_html(progress_path)
            print(f"Progress chart saved to: {progress_path}")
    
    else:
        # Multi-algorithm comparison
        routes_data = {alg: (result['route'], result['distance']) 
                      for alg, result in results.items()}
        
        # Create comparison map
        comparison_map_path = os.path.join(output_dir, 'algorithm_comparison_map.html')
        comparison_map = visualizer.compare_routes_map(
            routes_data,
            save_path=comparison_map_path
        )
        print(f"Comparison map saved to: {comparison_map_path}")
        
        # Create comparison chart
        comparison_chart = visualizer.plot_algorithm_comparison(results)
        chart_path = os.path.join(output_dir, 'algorithm_comparison_chart.html')
        comparison_chart.write_html(chart_path)
        print(f"Comparison chart saved to: {chart_path}")
        
        # Create comprehensive dashboard
        dashboard = visualizer.create_route_statistics_dashboard(results)
        dashboard_path = os.path.join(output_dir, 'optimization_dashboard.html')
        dashboard.write_html(dashboard_path)
        print(f"Dashboard saved to: {dashboard_path}")

def export_results(results: Dict, cities: List[Dict], output_dir: str, export_gpx: bool):
    """Export results to various formats."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results as JSON
    results_path = os.path.join(output_dir, 'optimization_results.json')
    save_optimization_results(results, results_path)
    print(f"Results saved to: {results_path}")
    
    # Export GPX files if requested
    if export_gpx:
        from utils import export_route_to_gpx
        
        for algorithm, result in results.items():
            gpx_path = os.path.join(output_dir, f'{algorithm}_route.gpx')
            export_route_to_gpx(result['route'], cities, gpx_path)
            print(f"GPX route saved to: {gpx_path}")

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Load cities data
        print(f"Loading cities from: {args.cities}")
        cities = load_cities(args.cities)
        print(f"Loaded {len(cities)} cities")
        
        # Load configuration
        config = load_config(args.config)
        
        # Set output directory
        output_dir = args.output or f"results_{args.algorithm}"
        
        # Run optimization
        if args.compare:
            results = compare_algorithms(cities, config, args)
        else:
            route, distance, stats = run_single_algorithm(cities, args.algorithm, config, args)
            results = {
                args.algorithm: {
                    'route': route,
                    'distance': distance,
                    'stats': stats,
                    'route_names': [cities[i]['name'] for i in route]
                }
            }
            
            print(f"\nOptimization completed!")
            print(f"Best route distance: {distance:.2f} km")
            print(f"Route: {' -> '.join([cities[i]['name'] for i in route])}")
        
        # Generate visualizations
        if args.visualize:
            print("\nGenerating visualizations...")
            generate_visualizations(cities, results, output_dir)
        
        # Export results
        print("\nExporting results...")
        export_results(results, cities, output_dir, args.export_gpx)
        
        print(f"\nAll outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
