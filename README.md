
# Transportation Route Optimization

A comprehensive Python application for solving vehicle routing problems using genetic algorithms and other optimization techniques. Features interactive visualization, real-world city data integration, and performance comparison tools.

## Features

- **Multiple Optimization Algorithms**:
  - Genetic Algorithm (GA) with customizable parameters
  - Nearest Neighbor heuristic
  - 2-opt local search improvement
  - Simulated Annealing

- **Interactive Visualization**:
  - Folium-based interactive maps
  - Route comparison and analysis
  - Real-time optimization progress tracking
  - Performance metrics dashboard

- **Real-World Integration**:
  - Support for city coordinate data
  - Distance matrix calculations
  - Custom waypoint management
  - Export/import route configurations

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using Docker
```bash
docker build -t route-optimizer .
docker run -p 8501:8501 route-optimizer
```

## Quick Start

### Command Line Interface
```bash
# Run genetic algorithm optimization
python src/main.py --algorithm ga --cities data/sample_cities.json

# Compare multiple algorithms
python src/main.py --compare --cities data/sample_cities.json

# Run web interface
python src/webapp.py
```

### Python API
```python
from src.genetic_algorithm import GeneticAlgorithm
from src.utils import load_cities

cities = load_cities('data/sample_cities.json')
ga = GeneticAlgorithm(cities, population_size=100, generations=500)
best_route, best_distance = ga.optimize()
```

## Web Interface

Launch the Streamlit web application:
```bash
streamlit run src/webapp.py
```

Access at `http://localhost:8501`

## Project Structure

```
transportation-route-optimization/
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── genetic_algorithm.py    # GA implementation
│   ├── heuristics.py          # Other optimization algorithms
│   ├── visualization.py       # Folium map generation
│   ├── webapp.py              # Streamlit web interface
│   └── utils.py               # Utility functions
├── tests/
│   ├── test_genetic_algorithm.py
│   ├── test_heuristics.py
│   └── test_utils.py
├── data/
│   ├── sample_cities.json     # Sample city coordinates
│   └── world_cities.csv       # Extended city database
├── docker/
│   └── Dockerfile
└── docs/
    └── algorithm_comparison.md
```

## Algorithm Details

### Genetic Algorithm
- **Selection**: Tournament selection with configurable size
- **Crossover**: Order crossover (OX1) for permutation preservation
- **Mutation**: Swap mutation with adaptive rates
- **Elitism**: Configurable elite preservation

### Performance Metrics
- Total route distance
- Execution time
- Convergence rate
- Solution quality comparison

## Configuration

Edit `config.yaml` to customize:
- Algorithm parameters
- Visualization settings
- Data sources
- Performance thresholds

## Testing

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
