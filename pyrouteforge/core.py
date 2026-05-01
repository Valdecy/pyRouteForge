"""
The single public entry point: :func:`solve`.

This module ties together :mod:`routeforge.data` (input normalization),
:mod:`routeforge.ga` (genetic algorithm) and :mod:`routeforge.plotting`
(Plotly visualizations) behind one function.

Typical usage
-------------
.. code-block:: python

    import pandas as pd
    from routeforge import solve

    locations = pd.DataFrame({
        "x":      [40, 25, 22, 22, 20, 20, 18, 15, 15],
        "y":      [50, 85, 75, 85, 80, 85, 75, 75, 80],
        "demand": [ 0, 20, 30, 10, 40, 20, 20, 20, 10],
    })

    result = solve(
        locations=locations,
        n_depots=1,
        capacity=150,
        fixed_cost=30,
        variable_cost=2,
        velocity=70,
        generations=300,
        seed=42,
    )

    print(result.report)
    result.plot().show()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data import LocationsInput, Problem, VehicleSpec
from .ga import run_genetic_algorithm
from .plotting import plot_convergence, plot_routes, plot_vehicle_loads


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    """The result of :func:`solve`.

    Attributes
    ----------
    routes
        A list of routes. Each route is a dict with keys
        ``vehicle_type``, ``depot``, ``stops``, ``load``, ``distance``.
    report
        Detailed per-stop pandas DataFrame (load, arrival/leave times,
        wait time, leg distance, costs).
    total_distance, total_cost
        Headline numbers, also at the bottom of ``report``.
    history
        Best total distance per GA generation (length ``generations + 1``).
    coordinates
        The 2D coordinates used for plotting.
    n_depots
        Number of depot rows at the start of ``coordinates``.
    raw
        The internal ``[depots, routes, vehicles]`` solution object.
        Only needed for advanced workflows.
    """

    routes: List[dict]
    report: pd.DataFrame
    total_distance: float
    total_cost: float
    history: List[float]
    coordinates: np.ndarray
    n_depots: int
    route_mode: str
    location_labels: Optional[List[str]] = None
    raw: Any = field(default=None, repr=False)

    # ---- convenience properties -------------------------------------------

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    @property
    def vehicles_used(self) -> int:
        return sum(1 for r in self.routes if r["stops"])

    # ---- visualization shortcuts ------------------------------------------

    def plot(self, **kwargs):
        """Plotly figure of the routed solution. See :func:`plot_routes`."""
        return plot_routes(
            self.coordinates, self.raw,
            n_depots=self.n_depots, route=self.route_mode,
            labels=self.location_labels,
            **kwargs,
        )

    def plot_convergence(self, **kwargs):
        """Plotly figure of best-distance vs GA generation."""
        return plot_convergence(self.history, **kwargs)

    def plot_loads(self, capacity: Sequence[float], parameters: np.ndarray, **kwargs):
        """Plotly bar chart of route load vs capacity."""
        return plot_vehicle_loads(self.raw, parameters, capacity, **kwargs)

    # ---- convenience IO ----------------------------------------------------

    def to_csv(self, path: str, sep: str = ";") -> None:
        """Save the per-stop report to CSV."""
        self.report.to_csv(path, sep=sep, index=False)


# ---------------------------------------------------------------------------
# Building human-friendly route summaries
# ---------------------------------------------------------------------------

def _summarize_routes(problem: Problem, raw_solution: list) -> List[dict]:
    demand_col = problem.parameters[:, 0]
    dm = problem.distance_matrix
    closed = problem.route == "closed"
    summary = []
    for j, sub in enumerate(raw_solution[1]):
        if not sub:
            continue
        depot = raw_solution[0][j][0]
        v_type = raw_solution[2][j][0]
        path = [depot] + list(sub) + ([depot] if closed else [])
        dist = float(sum(dm[path[k], path[k + 1]] for k in range(len(path) - 1)))
        load = float(sum(demand_col[i] for i in sub))
        summary.append({
            "route_id": j + 1,
            "vehicle_type": int(v_type),
            "depot": int(depot),
            "stops": [int(i) for i in sub],
            "load": load,
            "distance": dist,
        })
    return summary


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------

def solve(
    locations: Optional[LocationsInput] = None,
    *,
    distance_matrix: Optional[np.ndarray] = None,
    n_depots: int = 1,
    # ---- vehicles --------------------------------------------------------
    vehicles: Optional[VehicleSpec] = None,
    capacity: Optional[Union[float, Sequence[float]]] = None,
    fixed_cost: Optional[Union[float, Sequence[float]]] = None,
    variable_cost: Optional[Union[float, Sequence[float]]] = None,
    velocity: Optional[Union[float, Sequence[float]]] = None,
    fleet_size: Optional[Union[int, Sequence[int]]] = None,
    # ---- demands & TWs ---------------------------------------------------
    demand: Optional[Sequence[float]] = None,
    time_windows: Optional[pd.DataFrame] = None,
    # ---- problem variant -------------------------------------------------
    model: str = "vrp",
    route: str = "closed",
    time_window: str = "auto",
    # ---- GA hyper-parameters ---------------------------------------------
    population_size: int = 50,
    generations: int = 200,
    mutation_rate: float = 0.1,
    elite: int = 1,
    penalty_value: float = 10000.0,
    selection: str = "rw",
    # ---- misc ------------------------------------------------------------
    labels: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    on_generation: Optional[Callable[[int, float, float], None]] = None,
) -> Solution:
    """Solve a VRP / TSP / mTSP and return a :class:`Solution`.

    The function is intentionally broad — pass ``locations`` (a DataFrame
    or array of coordinates) **or** a precomputed ``distance_matrix``,
    plus whatever vehicle spec fits your problem.

    Parameters
    ----------
    locations
        DataFrame (or array) with at least x/y columns. Optional columns
        recognised automatically: ``demand``, ``tw_early``, ``tw_late``,
        ``tw_service_time``, ``tw_wait_cost``, and a ``name`` column for
        human-readable hover labels. Many aliases work — e.g. ``lat``/``lon``
        for coordinates, ``ready_time``/``due_time`` for time windows.
    distance_matrix
        Square numpy array. Use this when you don't have coordinates;
        a 2D embedding is reconstructed for plotting.
    n_depots
        The first ``n_depots`` rows of ``locations`` (or
        ``distance_matrix``) are treated as depots.
    vehicles
        Either a single dict (homogeneous fleet) or a list of dicts
        (heterogeneous). Each dict accepts ``capacity``, ``fixed_cost``,
        ``variable_cost``, ``velocity``, and ``count``. As an alternative
        you can pass the simple kwargs (``capacity``, ``fixed_cost``, …)
        directly.
    fleet_size
        If you want a finite fleet, pass either an int (homogeneous) or a
        list of ints (one per vehicle type). Empty / ``None`` -> infinite.
    model
        ``'vrp'`` (default), ``'tsp'``, or ``'mtsp'``.
    route
        ``'closed'`` (return to depot) or ``'open'``.
    time_window
        ``'auto'`` (default — detected from ``locations`` columns),
        ``'with'``, or ``'without'``.
    seed
        Reproducibility.

    Returns
    -------
    Solution
    """
    # 1. Normalize inputs into a Problem
    if locations is not None and distance_matrix is not None:
        raise ValueError("Pass either locations or distance_matrix, not both.")
    if locations is None and distance_matrix is None:
        raise ValueError("Either locations or distance_matrix is required.")

    if locations is not None:
        problem = Problem.from_locations(
            locations,
            n_depots=n_depots,
            vehicles=vehicles,
            capacity=capacity, fixed_cost=fixed_cost,
            variable_cost=variable_cost, velocity=velocity,
            fleet_size=fleet_size,
            demand=demand, model=model, route=route,
            time_window=time_window, labels=labels,
        )
    else:
        problem = Problem.from_distance_matrix(
            distance_matrix,
            n_depots=n_depots,
            vehicles=vehicles,
            capacity=capacity, fixed_cost=fixed_cost,
            variable_cost=variable_cost, velocity=velocity,
            fleet_size=fleet_size,
            demand=demand, time_windows=time_windows,
            model=model, route=route, labels=labels,
        )

    # 2. Run the GA
    report, raw, history = run_genetic_algorithm(
        problem.coordinates,
        problem.distance_matrix,
        problem.parameters,
        velocity=problem.velocity,
        fixed_cost=problem.fixed_cost,
        variable_cost=problem.variable_cost,
        capacity=problem.capacity,
        population_size=population_size,
        vehicle_types=problem.vehicle_types,
        n_depots=problem.n_depots,
        route=problem.route,
        model=problem.model,
        time_window=problem.time_window,
        fleet_size=problem.fleet_size,
        mutation_rate=mutation_rate,
        elite=elite,
        generations=generations,
        penalty_value=penalty_value,
        selection=selection,
        seed=seed,
        verbose=verbose,
        on_generation=on_generation,
    )

    # 3. Build a friendly Solution
    routes_summary = _summarize_routes(problem, raw)
    total_row = report.iloc[-1]
    total_distance = float(total_row.get("Distance", 0.0) or 0.0)
    total_cost = float(total_row.get("Costs", 0.0) or 0.0)

    return Solution(
        routes=routes_summary,
        report=report,
        total_distance=total_distance,
        total_cost=total_cost,
        history=history,
        coordinates=problem.coordinates,
        n_depots=problem.n_depots,
        route_mode=problem.route,
        location_labels=problem.location_labels,
        raw=raw,
    )
