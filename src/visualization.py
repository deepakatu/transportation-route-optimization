
"""
Visualization module for route optimization using Folium for interactive maps.
"""

import folium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class RouteVisualizer:
    """Interactive visualization for route optimization results."""
    
    def __init__(self, cities: List[Dict]):
        """
        Initialize visualizer with city data.
        
        Args:
            cities: List of city dictionaries with 'latitude', 'longitude', 'name'
        """
        self.cities = cities
        self.center_lat = np.mean([city['latitude'] for city in cities])
        self.center_lon = np.mean([city['longitude'] for city in cities])
    
    def create_route_map(self, route: List[int], title: str = "Optimized Route",
                        save_path: Optional[str] = None) -> folium.Map:
        """
        Create an interactive Folium map showing the optimized route.
        
        Args:
            route: List of city indices in route order
            title: Map title
            save_path: Path to save HTML file (optional)
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add city markers
        for i, city_idx in enumerate(route):
            city = self.cities[city_idx]
            
            # Different colors for start/end vs intermediate cities
            if i == 0:
                color = 'green'
                icon = 'play'
                popup_text = f"START: {city['name']}"
            elif i == len(route) - 1:
                color = 'red'
                icon = 'stop'
                popup_text = f"END: {city['name']}"
            else:
                color = 'blue'
                icon = 'info-sign'
                popup_text = f"{i}: {city['name']}"
            
            folium.Marker(
                location=[city['latitude'], city['longitude']],
                popup=popup_text,
                tooltip=city['name'],
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
        
        # Add route lines
        route_coordinates = []
        for city_idx in route:
            city = self.cities[city_idx]
            route_coordinates.append([city['latitude'], city['longitude']])
        
        # Add return to start
        if route_coordinates:
            route_coordinates.append(route_coordinates[0])
        
        folium.PolyLine(
            locations=route_coordinates,
            color='red',
            weight=3,
            opacity=0.8,
            popup='Optimized Route'
        ).add_to(m)
        
        # Add distance annotations
        total_distance = self._calculate_total_distance(route)
        distance_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 60px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Total Distance:</b><br>
        {total_distance:.2f} km
        </div>
        '''
        m.get_root().html.add_child(folium.Element(distance_html))
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def compare_routes_map(self, routes_data: Dict[str, Tuple[List[int], float]], 
                          save_path: Optional[str] = None) -> folium.Map:
        """
        Create a map comparing multiple routes.
        
        Args:
            routes_data: Dictionary with algorithm names as keys and (route, distance) tuples
            save_path: Path to save HTML file (optional)
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:20px"><b>Route Comparison</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Color palette for different routes
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
                 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        # Add city markers (only once)
        for i, city in enumerate(self.cities):
            folium.Marker(
                location=[city['latitude'], city['longitude']],
                popup=f"{city['name']} (Index: {i})",
                tooltip=city['name'],
                icon=folium.Icon(color='gray', icon='info-sign')
            ).add_to(m)
        
        # Add routes with different colors
        legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 300px; background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px"><b>Routes:</b><br>'
        
        for i, (algorithm, (route, distance)) in enumerate(routes_data.items()):
            color = colors[i % len(colors)]
            
            # Create route coordinates
            route_coordinates = []
            for city_idx in route:
                city = self.cities[city_idx]
                route_coordinates.append([city['latitude'], city['longitude']])
            
            # Add return to start
            if route_coordinates:
                route_coordinates.append(route_coordinates[0])
            
            # Add route line
            folium.PolyLine(
                locations=route_coordinates,
                color=color,
                weight=2,
                opacity=0.7,
                popup=f'{algorithm}: {distance:.2f} km'
            ).add_to(m)
            
            # Add to legend
            legend_html += f'<span style="color:{color}">■</span> {algorithm}: {distance:.2f} km<br>'
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def plot_optimization_progress(self, fitness_history: List[float], 
                                 avg_fitness_history: List[float],
                                 title: str = "Optimization Progress") -> go.Figure:
        """
        Plot optimization progress using Plotly.
        
        Args:
            fitness_history: Best fitness values over generations
            avg_fitness_history: Average fitness values over generations
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        generations = list(range(len(fitness_history)))
        
        fig = go.Figure()
        
        # Add best fitness trace
        fig.add_trace(go.Scatter(
            x=generations,
            y=fitness_history,
            mode='lines',
            name='Best Fitness',
            line=dict(color='red', width=2)
        ))
        
        # Add average fitness trace
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_fitness_history,
            mode='lines',
            name='Average Fitness',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title='Fitness',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_algorithm_comparison(self, results: Dict) -> go.Figure:
        """
        Create a bar chart comparing algorithm performance.
        
        Args:
            results: Dictionary with algorithm results
            
        Returns:
            Plotly figure object
        """
        algorithms = list(results.keys())
        distances = [results[alg]['distance'] for alg in algorithms]
        
        # Create color scale
        colors = px.colors.qualitative.Set3[:len(algorithms)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=algorithms,
                y=distances,
                marker_color=colors,
                text=[f'{d:.1f} km' for d in distances],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Algorithm Performance Comparison',
            xaxis_title='Algorithm',
            yaxis_title='Total Distance (km)',
            template='plotly_white'
        )
        
        return fig
    
    def create_distance_matrix_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap of the distance matrix.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Calculate distance matrix
        n = len(self.cities)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = self._haversine_distance(
                        self.cities[i]['latitude'], self.cities[i]['longitude'],
                        self.cities[j]['latitude'], self.cities[j]['longitude']
                    )
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        city_names = [city['name'] for city in self.cities]
        
        sns.heatmap(
            distance_matrix,
            xticklabels=city_names,
            yticklabels=city_names,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Distance (km)'}
        )
        
        plt.title('Distance Matrix Between Cities')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_route_statistics_dashboard(self, results: Dict) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple statistics.
        
        Args:
            results: Dictionary with algorithm results
            
        Returns:
            Plotly figure with subplots
        """
        algorithms = list(results.keys())
        distances = [results[alg]['distance'] for alg in algorithms]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distance Comparison', 'Route Efficiency', 
                          'City Visit Order', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "table"}]]
        )
        
        # Distance comparison bar chart
        fig.add_trace(
            go.Bar(x=algorithms, y=distances, name='Distance'),
            row=1, col=1
        )
        
        # Route efficiency scatter plot
        best_distance = min(distances)
        efficiency = [(best_distance / d) * 100 for d in distances]
        
        fig.add_trace(
            go.Scatter(
                x=algorithms, 
                y=efficiency, 
                mode='markers+lines',
                name='Efficiency %',
                marker=dict(size=10)
            ),
            row=1, col=2
        )
        
        # Create a simple route visualization matrix
        if results:
            first_route = list(results.values())[0]['route']
            route_matrix = np.zeros((len(first_route), len(algorithms)))
            
            for j, (alg, data) in enumerate(results.items()):
                for i, city_idx in enumerate(data['route']):
                    route_matrix[i][j] = city_idx
            
            fig.add_trace(
                go.Heatmap(
                    z=route_matrix,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=2, col=1
            )
        
        # Performance metrics table
        table_data = []
        for alg, data in results.items():
            table_data.append([
                alg,
                f"{data['distance']:.2f} km",
                f"{(best_distance / data['distance']) * 100:.1f}%"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Algorithm', 'Distance', 'Efficiency']),
                cells=dict(values=list(zip(*table_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Route Optimization Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def _calculate_total_distance(self, route: List[int]) -> float:
        """Calculate total distance for a route."""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self._haversine_distance(
                self.cities[from_city]['latitude'], self.cities[from_city]['longitude'],
                self.cities[to_city]['latitude'], self.cities[to_city]['longitude']
            )
        return total_distance
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
