# pyRouteForge

**A clean and fast Vehicle Routing Problem solver — powered by Genetic Algorithms.**

pyRouteForge solves a wide family of routing problems with a single function call:

- **Capacitated VRP** —  ([ Colab Demo ](https://colab.research.google.com/drive/1UvOtkRnjPF5PUi9A4yd2Yg1EZ3EBJv-F?usp=sharing)) — Assign customers to a vehicle fleet with capacity limits
- **Multi-Depot VRP** —  ([ Colab Demo ](https://colab.research.google.com/drive/1kQvpiv_oCNK3aq1-BJtIq7Rshs6qIZK1?usp=sharing)) — Multiple starting depots, depot auto-assignment per route
- **VRP with Time Windows** —  ([ Colab Demo ](https://colab.research.google.com/drive/1OMiw2l-SR4w8CFrkPF80BdcVz13UpDMe?usp=sharing)) — Earliest/latest arrival, service times, waiting costs
- **Heterogeneous Fleet** —  ([ Colab Demo ](https://colab.research.google.com/drive/16vgnJQHq4HtQmRFvCWIUCDUuiU0BVD8l?usp=sharing)) — Mix vehicle types with different capacities, costs, speeds
- **Finite or Infinite Fleet** —  ([ Colab Demo ](https://colab.research.google.com/drive/1pQHW2qvaXFQzvdMdJjQzifYs1BOtKfoh?usp=sharing)) — Set hard limits on vehicle counts or leave them open
- **TSP** —  ([ Colab Demo ](https://colab.research.google.com/drive/1X_MIsZHJum4-zBOm64inQ_bOK5oeFVSC?usp=sharing)) — Classical Travelling Salesman
- **mTSP** —  ([ Colab Demo ](https://colab.research.google.com/drive/1k0KJVfuP2rStTA10JOEDfEVNjBUBQqm9?usp=sharing)) — Multi-Depot Travelling Salesman
- **Open or Closed Routes** —  ([ Colab Demo ](https://colab.research.google.com/drive/1cA19MoR2j8slce0iiN21iSP4HFg8Tx-7?usp=sharing)) — Return to depot or finish at the last customer

---

## Features

- **One function**: `solve()` — that accepts a pandas DataFrame, a numpy array, or just a distance matrix
- **One result object**: `Solution` — with `.report`, `.routes`, `.total_distance`, and `.plot()`
- **Plotly**: that render natively in Google Colab, Jupyter, and as standalone HTML

---

## Installation

```bash
pip install pyrouteforge
```

---

## Quick start

```python
import pandas as pd
from pyrouteforge import solve

# Row 0 is the depot, the rest are customers.
df = pd.DataFrame({
					"x":      [40, 25, 22, 22, 20, 20, 18, 15, 15],
					"y":      [50, 85, 75, 85, 80, 85, 75, 75, 80],
					"demand": [ 0, 20, 30, 10, 40, 20, 20, 20, 10],
				  })

result = solve(
					locations     = df,
					n_depots      = 1,
					capacity      = 150,
					fixed_cost    = 30,
					variable_cost = 2,
					velocity      = 70,
					generations   = 300,
					seed          = 42,
			   )

print(f"Total Distance: {result.total_distance:.2f}")
result.plot().show()         
result.plot_convergence().show()
```

Output:

```
Total distance: 263.41

Route #1: depot = 0, vehicle = 0, load = 130, distance = 152.66, stops = [5, 3, 1, 8, 7, 6, 2, 4]
```

---

## Input formats

`solve()` is deliberately flexible. Whatever you have on hand, it'll work.

### a. A pandas DataFrame (recommended)

The most natural format. pyRouteForge auto-detects column names:

| What you mean       | Recognised column aliases                                     |
|---------------------|---------------------------------------------------------------|
| X coordinate        | `x`, `lon`, `lng`, `longitude`                                |
| Y coordinate        | `y`, `lat`, `latitude`                                        |
| Demand              | `demand`, `weight`, `load`, `qty`, `quantity`                 |
| Time window start   | `tw_early`, `ready_time`, `earliest`, `open_time`             |
| Time window end     | `tw_late`, `due_time`, `latest`, `close_time`                 |
| Service time        | `tw_service_time`, `service_time`, `service`                  |
| Waiting cost        | `tw_wait_cost`, `wait_cost`, `waiting_cost`                   |
| Display label       | `name`, `label`, `id`                                         |


### b. A numpy array of coordinates

```python
import numpy as np
arr    = np.array([
                    [40, 50], 
					[25, 85], 
					[22, 75], 
					[22, 85]
			      ])
result = solve(locations = arr, demand = [0, 20, 30, 10], capacity = 100)
```

### c. A precomputed distance matrix

When your network isn't Euclidean (real-world driving distances, sea routes, etc.):

```python
result = solve(
				distance_matrix = dm,            
				demand          = [0, 20, 30, 10],
				capacity        = 100,
			  )
```

---

## Vehicle specification

### Homogeneous fleet — simple kwargs

```python
result = solve(
				locations     = df,
				capacity      = 150,
				fixed_cost    = 30,
				variable_cost = 2,
				velocity      = 70,
				fleet_size    = 5,  # omit for infinite
			)
```

### Heterogeneous fleet — list of dicts

```python
result = solve(
					locations = df,
					vehicles  = [
									{"capacity": 50,  "fixed_cost": 10, "variable_cost": 0.5,
									 "velocity": 60,  "count": 3},     
									{"capacity": 100, "fixed_cost": 30, "variable_cost": 1.2,
									 "velocity": 80,  "count": 2},    
								],
				)
```

---

## Working with the result

`solve()` returns a `Solution` with everything you need:

```python
result.total_distance      # float
result.total_cost          # float
result.n_routes            # int
result.routes              # list of dicts: route_id, vehicle_type, depot, stops, load, distance
result.report              # pandas DataFrame: per-stop schedule (load, arrival/leave times, etc.)
result.history             # list[float] — best distance per generation
result.coordinates         # np.ndarray — the (n, 2) layout used for plotting
result.raw                 # internal [depots, routes, vehicles] structure (advanced use)

# Plotting
result.plot()              # main route map
result.plot_convergence()  # GA fitness curve
result.plot_loads(         # bar chart of load vs capacity per route
					capacity   = [150],
					parameters = problem.parameters,
				 )

# Export
result.to_csv("routes.csv")
```

---

## Plotting

All figures are `plotly.graph_objects.Figure` instances:

```python
fig = result.plot(title = "Solution", width = 1100, height = 750)
fig.show()                                 # Jupyter / Colab inline
fig.write_html("solution.html")            # standalone HTML
fig.write_image("solution.png", scale = 2) # requires kaleido
```

The default styling is a dark theme with:

- **Glow underlay + crisp main line** for each route, distinct colors per route
- **Arrowheads** on every leg so direction is obvious
- **Square depot markers** in amber, **circular client markers** in slate
- **Rich hover tooltips** with stop number, vehicle type, coordinates
- **Equal-aspect axes** so geometry isn't distorted

Customize anything by accessing `fig.layout` / `fig.data` directly — it's just Plotly underneath.

---

## GA hyper-parameters

All optional with sensible defaults. Larger instances generally want bigger populations and more generations.

| Parameter           | Default | Meaning                                                  |
|---------------------|---------|----------------------------------------------------------|
| `population_size`   | 50      | Number of candidate solutions per generation             |
| `generations`       | 200     | Total iterations                                         |
| `mutation_rate`     | 0.10    | Probability of mutation per individual                   |
| `elite`             | 1       | Best individuals carried over unchanged each generation  |
| `penalty_value`     | 10000   | Penalty for violating capacity / time-window constraints |
| `selection`         | `'rw'`  | `'rw'` (roulette wheel) or `'rank'`                      |
| `seed`              | `None`  | Reproducibility                                          |
| `verbose`           | `False` | Print per-generation progress                            |

You can also pass an `on_generation` callback for live monitoring:

```python
def progress(gen, distance, cost):
    print(f"Gen {gen}: distance = {distance:.2f}")

solve(locations = df, capacity = 100, on_generation = progress)
```

---

