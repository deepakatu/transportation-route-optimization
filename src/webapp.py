
"""
Streamlit web application for Transportation Route Optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import List, Dict, Optional
import plotly.graph_objects as go
import time

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm import GeneticAlgorithm, GAConfig
from heuristics import RouteOptimizer, SAConfig, TwoOptConfig
from visualization import RouteVisualizer
from utils import load_cities, generate_random_cities, haversine_distance

# Page configuration
st.set_page_config(
    page_title="Transportation Route Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .algorithm-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_cities():
    """Load sample cities data."""
    try:
        return load_cities('data/sample_cities.json')
    except:
        # Fallback to generated cities if sample file not found
        return generate_random_cities(15)

@st.cache_data
def calculate_distance_matrix_cached(cities_json: str):
    """Cached distance matrix calculation."""
    cities = json.loads(cities_json)
    n = len(cities)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(
                cities[i]['latitude'], cities[i]['longitude'],
                cities[j]['latitude'], cities[j]['longitude']
            )
            matrix[i][j] = matrix[j][i] = dist
    
    return matrix

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚛 Transportation Route Optimizer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Optimize delivery routes using advanced algorithms including Genetic Algorithm, 
    Simulated Annealing, and various heuristics. Compare performance and visualize results 
    on interactive maps.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Sample Cities", "Upload File", "Generate Random"]
    )
    
    cities = None
    
    if data_source == "Sample Cities":
        cities = load_sample_cities()
        st.sidebar.success(f"Loaded {len(cities)} sample cities")
    
    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a cities file",
            type=['json', 'csv'],
            help="Upload a JSON or CSV file with city data (name, latitude, longitude)"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    cities = data.get('cities', data)
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    cities = df.to_dict('records')
                
                st.sidebar.success(f"Loaded {len(cities)} cities from file")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
    
    elif data_source == "Generate Random":
        num_cities = st.sidebar.slider("Number of Cities", 5, 50, 15)
        region = st.sidebar.selectbox(
            "Region",
            ["Continental US", "Europe", "Custom"]
        )
        
        if region == "Continental US":
            bounds = (25.0, 49.0, -125.0, -66.0)
        elif region == "Europe":
            bounds = (35.0, 71.0, -10.0, 40.0)
        else:
            st.sidebar.subheader("Custom Bounds")
            min_lat = st.sidebar.number_input("Min Latitude", -90.0, 90.0, 25.0)
            max_lat = st.sidebar.number_input("Max Latitude", -90.0, 90.0, 49.0)
            min_lon = st.sidebar.number_input("Min Longitude", -180.0, 180.0, -125.0)
            max_lon = st.sidebar.number_input("Max Longitude", -180.0, 180.0, -66.0)
            bounds = (min_lat, max_lat, min_lon, max_lon)
        
        if st.sidebar.button("Generate Cities"):
            cities = generate_random_cities(num_cities, bounds)
            st.sidebar.success(f"Generated {len(cities)} random cities")
    
    if cities is None:
        st.warning("Please select a data source and load cities to continue.")
        return
    
    # Algorithm selection
    st.sidebar.header("🧮 Algorithm Selection")
    
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ["Genetic Algorithm", "Simulated Annealing", "2-opt", "Nearest Neighbor", "Compare All"]
    )
    
    # Algorithm parameters
    if algorithm == "Genetic Algorithm":
        st.sidebar.subheader("GA Parameters")
        population_size = st.sidebar.slider("Population Size", 20, 200, 100)
        generations = st.sidebar.slider("Generations", 50, 1000, 500)
        mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.1, 0.02)
        elite_size = st.sidebar.slider("Elite Size", 1, 20, 5)
        
    elif algorithm == "Simulated Annealing":
        st.sidebar.subheader("SA Parameters")
        initial_temp = st.sidebar.slider("Initial Temperature", 100, 2000, 1000)
        cooling_rate = st.sidebar.slider("Cooling Rate", 0.8, 0.99, 0.95)
        max_iterations = st.sidebar.slider("Max Iterations", 1000, 20000, 10000)
    
    elif algorithm == "2-opt":
        st.sidebar.subheader("2-opt Parameters")
        max_iterations = st.sidebar.slider("Max Iterations", 100, 2000, 1000)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 City Data")
        
        # Display cities table
        cities_df = pd.DataFrame(cities)
        st.dataframe(cities_df[['name', 'latitude', 'longitude']], height=300)
        
        # Basic statistics
        st.subheader("📈 Statistics")
        st.metric("Total Cities", len(cities))
        
        if len(cities) > 1:
            # Calculate some basic stats
            distances = []
            for i in range(len(cities)):
                for j in range(i + 1, len(cities)):
                    dist = haversine_distance(
                        cities[i]['latitude'], cities[i]['longitude'],
                        cities[j]['latitude'], cities[j]['longitude']
                    )
                    distances.append(dist)
            
            st.metric("Avg Distance", f"{np.mean(distances):.1f} km")
            st.metric("Max Distance", f"{np.max(distances):.1f} km")
    
    with col1:
        st.subheader("🗺️ Route Optimization")
        
        # Run optimization button
        if st.button("🚀 Optimize Route", type="primary"):
            
            with st.spinner("Optimizing route..."):
                start_time = time.time()
                
                try:
                    if algorithm == "Compare All":
                        # Run comparison
                        optimizer = RouteOptimizer(cities)
                        
                        results = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        algorithms = [
                            ("Nearest Neighbor", "nn"),
                            ("2-opt", "2opt"),
                            ("Simulated Annealing", "sa"),
                            ("Genetic Algorithm", "ga")
                        ]
                        
                        for i, (name, alg_code) in enumerate(algorithms):
                            status_text.text(f"Running {name}...")
                            
                            if alg_code == "nn":
                                route, distance = optimizer.nearest_neighbor(verbose=False)
                            elif alg_code == "2opt":
                                route, distance = optimizer.two_opt(verbose=False)
                            elif alg_code == "sa":
                                sa_config = SAConfig(
                                    initial_temperature=1000,
                                    cooling_rate=0.95,
                                    max_iterations=5000
                                )
                                route, distance = optimizer.simulated_annealing(
                                    config=sa_config, verbose=False
                                )
                            elif alg_code == "ga":
                                ga_config = GAConfig(
                                    population_size=50,
                                    generations=200,
                                    mutation_rate=0.02
                                )
                                ga = GeneticAlgorithm(cities, ga_config)
                                route, distance = ga.optimize(verbose=False)
                            
                            results[name] = {
                                'route': route,
                                'distance': distance,
                                'route_names': [cities[j]['name'] for j in route]
                            }
                            
                            progress_bar.progress((i + 1) / len(algorithms))
                        
                        status_text.text("Optimization complete!")
                        
                        # Display comparison results
                        st.subheader("📊 Algorithm Comparison")
                        
                        comparison_df = pd.DataFrame([
                            {
                                'Algorithm': alg,
                                'Distance (km)': f"{result['distance']:.2f}",
                                'Efficiency': f"{(min(r['distance'] for r in results.values()) / result['distance'] * 100):.1f}%"
                            }
                            for alg, result in results.items()
                        ])
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Create comparison visualization
                        visualizer = RouteVisualizer(cities)
                        
                        # Bar chart comparison
                        fig_comparison = visualizer.plot_algorithm_comparison(results)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Show best route map
                        best_algorithm = min(results.keys(), key=lambda k: results[k]['distance'])
                        best_route = results[best_algorithm]['route']
                        best_distance = results[best_algorithm]['distance']
                        
                        st.subheader(f"🏆 Best Route: {best_algorithm}")
                        
                        route_map = visualizer.create_route_map(
                            best_route,
                            title=f"{best_algorithm} - {best_distance:.2f} km"
                        )
                        
                        # Display map
                        st.components.v1.html(route_map._repr_html_(), height=500)
                    
                    else:
                        # Run single algorithm
                        if algorithm == "Genetic Algorithm":
                            ga_config = GAConfig(
                                population_size=population_size,
                                generations=generations,
                                mutation_rate=mutation_rate,
                                elite_size=elite_size
                            )
                            
                            ga = GeneticAlgorithm(cities, ga_config)
                            
                            # Create progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            chart_placeholder = st.empty()
                            
                            # Custom optimization with progress updates
                            population = ga._initialize_population()
                            best_fitness_history = []
                            
                            for generation in range(generations):
                                # Update progress
                                progress = generation / generations
                                progress_bar.progress(progress)
                                status_text.text(f"Generation {generation}/{generations}")
                                
                                # Run one generation
                                population.sort(key=ga._fitness, reverse=True)
                                current_best_distance = ga._calculate_route_distance(population[0])
                                best_fitness_history.append(1 / (1 + current_best_distance))
                                
                                # Update chart every 10 generations
                                if generation % 10 == 0 and generation > 0:
                                    fig_progress = go.Figure()
                                    fig_progress.add_trace(go.Scatter(
                                        x=list(range(len(best_fitness_history))),
                                        y=best_fitness_history,
                                        mode='lines',
                                        name='Best Fitness'
                                    ))
                                    fig_progress.update_layout(
                                        title="Optimization Progress",
                                        xaxis_title="Generation",
                                        yaxis_title="Fitness",
                                        height=300
                                    )
                                    chart_placeholder.plotly_chart(fig_progress, use_container_width=True)
                                
                                # Create new population
                                if generation < generations - 1:
                                    new_population = population[:ga.config.elite_size]
                                    
                                    while len(new_population) < ga.config.population_size:
                                        parent1 = ga._tournament_selection(population)
                                        parent2 = ga._tournament_selection(population)
                                        child = ga._order_crossover(parent1, parent2)
                                        child = ga._swap_mutation(child)
                                        new_population.append(child)
                                    
                                    population = new_population
                            
                            # Final results
                            best_route = population[0]
                            best_distance = ga._calculate_route_distance(best_route)
                            
                            status_text.text("Optimization complete!")
                            
                        elif algorithm == "Simulated Annealing":
                            sa_config = SAConfig(
                                initial_temperature=initial_temp,
                                cooling_rate=cooling_rate,
                                max_iterations=max_iterations
                            )
                            
                            optimizer = RouteOptimizer(cities)
                            best_route, best_distance = optimizer.simulated_annealing(
                                config=sa_config, verbose=False
                            )
                            
                        elif algorithm == "2-opt":
                            two_opt_config = TwoOptConfig(max_iterations=max_iterations)
                            optimizer = RouteOptimizer(cities)
                            best_route, best_distance = optimizer.two_opt(
                                config=two_opt_config, verbose=False
                            )
                            
                        elif algorithm == "Nearest Neighbor":
                            optimizer = RouteOptimizer(cities)
                            best_route, best_distance = optimizer.nearest_neighbor(verbose=False)
                        
                        # Display results
                        execution_time = time.time() - start_time
                        
                        st.success(f"Optimization completed in {execution_time:.2f} seconds!")
                        
                        # Metrics
                        col1_metrics, col2_metrics, col3_metrics = st.columns(3)
                        
                        with col1_metrics:
                            st.metric("Total Distance", f"{best_distance:.2f} km")
                        
                        with col2_metrics:
                            st.metric("Execution Time", f"{execution_time:.2f} s")
                        
                        with col3_metrics:
                            avg_distance = best_distance / len(cities)
                            st.metric("Avg Distance/City", f"{avg_distance:.2f} km")
                        
                        # Route details
                        st.subheader("📋 Route Details")
                        route_names = [cities[i]['name'] for i in best_route]
                        route_text = " → ".join(route_names) + f" → {route_names[0]}"
                        st.text_area("Route Order", route_text, height=100)
                        
                        # Visualization
                        st.subheader("🗺️ Route Visualization")
                        
                        visualizer = RouteVisualizer(cities)
                        route_map = visualizer.create_route_map(
                            best_route,
                            title=f"{algorithm} Optimized Route - {best_distance:.2f} km"
                        )
                        
                        # Display map
                        st.components.v1.html(route_map._repr_html_(), height=500)
                        
                        # Download options
                        st.subheader("💾 Download Results")
                        
                        # Prepare download data
                        results_data = {
                            'algorithm': algorithm,
                            'route': best_route,
                            'distance': best_distance,
                            'execution_time': execution_time,
                            'route_names': route_names,
                            'cities': cities
                        }
                        
                        results_json = json.dumps(results_data, indent=2)
                        
                        col1_download, col2_download = st.columns(2)
                        
                        with col1_download:
                            st.download_button(
                                label="📄 Download Results (JSON)",
                                data=results_json,
                                file_name=f"route_optimization_{algorithm.lower().replace(' ', '_')}.json",
                                mime="application/json"
                            )
                        
                        with col2_download:
                            # Create GPX content
                            gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1">\n<trk>\n<trkseg>\n'
                            for city_idx in best_route:
                                city = cities[city_idx]
                                gpx_content += f'<trkpt lat="{city["latitude"]}" lon="{city["longitude"]}"><name>{city["name"]}</name></trkpt>\n'
                            # Return to start
                            start_city = cities[best_route[0]]
                            gpx_content += f'<trkpt lat="{start_city["latitude"]}" lon="{start_city["longitude"]}"><name>{start_city["name"]}</name></trkpt>\n'
                            gpx_content += '</trkseg>\n</trk>\n</gpx>'
                            
                            st.download_button(
                                label="🗺️ Download Route (GPX)",
                                data=gpx_content,
                                file_name=f"route_{algorithm.lower().replace(' ', '_')}.gpx",
                                mime="application/gpx+xml"
                            )
                
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Transportation Route Optimizer | Built with Streamlit and advanced optimization algorithms
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
