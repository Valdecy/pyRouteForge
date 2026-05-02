"""
routeforge — a clean, fast, and beautiful Vehicle Routing Problem solver.

Public API
----------
* :func:`solve` — the one-stop function that builds a problem from your
  data, runs the genetic algorithm, and returns a :class:`Solution`.
* :class:`Solution` — the result, with ``.report``, ``.routes``,
  ``.total_distance``, ``.plot()``, ``.plot_convergence()``.
* :class:`Problem` — exposed for advanced workflows.
* :func:`build_distance_matrix`, :func:`build_coordinates` — geometry helpers.
* :func:`plot_routes`, :func:`plot_convergence`, :func:`plot_vehicle_loads` —
  Plotly figures, callable directly without going through Solution.

Example
-------
.. code-block:: python

    import pandas as pd
    from pyrouteforge import solve

    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 5, 3, 1],
                       "demand": [0, 10, 15, 5]})

    result = solve(df, capacity=30, generations=100, seed=42)
    result.plot().show()
"""

from ._kernels import build_coordinates, build_distance_matrix
from .core import Solution, solve
from .data import Problem
from .plotting import plot_convergence, plot_routes, plot_vehicle_loads

__version__ = "1.4.0"

__all__ = [
    "solve",
    "Solution",
    "Problem",
    "build_distance_matrix",
    "build_coordinates",
    "plot_routes",
    "plot_convergence",
    "plot_vehicle_loads",
    "__version__",
]
