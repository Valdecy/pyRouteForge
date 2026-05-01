"""
Data input layer.

The goal here is to make ``routeforge.solve(...)`` accept whatever shape
of data the user has on hand — pandas DataFrame, dict, numpy arrays, or
even just a distance matrix — and normalize everything to the internal
arrays the genetic algorithm expects.

Two main entry points:

* ``Problem.from_locations(...)``  — locations as DataFrame or array of (x, y) plus demand
* ``Problem.from_distance_matrix(...)`` — when the user already has a matrix

All other inputs (vehicles, depots, time windows) are normalized via
``_normalize_vehicles`` and ``_normalize_parameters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._kernels import build_coordinates, build_distance_matrix


VehicleSpec = Union[int, float, Mapping[str, Any], Sequence[Mapping[str, Any]]]
LocationsInput = Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]]


# Column aliases — users hate having to remember exact column names.
_X_COLS = {"x", "lon", "lng", "longitude"}
_Y_COLS = {"y", "lat", "latitude"}
_DEMAND_COLS = {"demand", "weight", "load", "qty", "quantity"}
_TW_EARLY_COLS = {"tw_early", "ready_time", "earliest", "open_time"}
_TW_LATE_COLS = {"tw_late", "due_time", "latest", "close_time"}
_TW_SERVICE_COLS = {"tw_service_time", "service_time", "service"}
_TW_WAIT_COST_COLS = {"tw_wait_cost", "wait_cost", "waiting_cost"}


def _find_column(df: pd.DataFrame, candidates: set) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _coerce_locations_to_df(locations: LocationsInput) -> pd.DataFrame:
    """Accept a DataFrame, array-like, or list of sequences and return a DataFrame
    with at least ``x`` and ``y`` columns.
    """
    if isinstance(locations, pd.DataFrame):
        return locations.copy()

    arr = np.asarray(locations, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            "locations must have shape (n, 2+) with at least x and y columns; "
            f"got shape {arr.shape}."
        )
    cols = ["x", "y"] + [f"col_{i}" for i in range(arr.shape[1] - 2)]
    return pd.DataFrame(arr, columns=cols)


def _normalize_parameters(
    n_nodes: int,
    df: Optional[pd.DataFrame],
    demand: Optional[Sequence[float]],
    has_time_windows: bool,
) -> np.ndarray:
    """Build the (n, 5) parameter matrix expected by the GA.

    Columns: [demand, tw_early, tw_late, tw_service_time, tw_wait_cost].
    """
    params = np.zeros((n_nodes, 5), dtype=np.float64)
    # Defaults keep time-window-disabled problems behaving correctly:
    # tw_late = +inf so no late arrivals are flagged.
    params[:, 2] = np.inf

    if df is not None:
        d_col = _find_column(df, _DEMAND_COLS)
        if d_col is not None:
            params[:, 0] = df[d_col].to_numpy(dtype=float)

        if has_time_windows:
            e_col = _find_column(df, _TW_EARLY_COLS)
            l_col = _find_column(df, _TW_LATE_COLS)
            s_col = _find_column(df, _TW_SERVICE_COLS)
            w_col = _find_column(df, _TW_WAIT_COST_COLS)
            if e_col is not None:
                params[:, 1] = df[e_col].to_numpy(dtype=float)
            if l_col is not None:
                params[:, 2] = df[l_col].to_numpy(dtype=float)
            if s_col is not None:
                params[:, 3] = df[s_col].to_numpy(dtype=float)
            if w_col is not None:
                params[:, 4] = df[w_col].to_numpy(dtype=float)

    if demand is not None:
        demand_arr = np.asarray(demand, dtype=float)
        if demand_arr.shape[0] != n_nodes:
            raise ValueError(
                f"demand has length {demand_arr.shape[0]} but expected {n_nodes}."
            )
        params[:, 0] = demand_arr

    return params


def _normalize_vehicles(
    vehicles: Optional[VehicleSpec],
    capacity: Optional[Union[float, Sequence[float]]] = None,
    fixed_cost: Optional[Union[float, Sequence[float]]] = None,
    variable_cost: Optional[Union[float, Sequence[float]]] = None,
    velocity: Optional[Union[float, Sequence[float]]] = None,
    fleet_size: Optional[Union[int, Sequence[int]]] = None,
) -> Dict[str, list]:
    """Normalize the multitude of ways to specify vehicles into the
    canonical 5-list form: capacity / fixed_cost / variable_cost / velocity / fleet_size.

    Accepted shapes for ``vehicles``:

    * ``None`` — fall back to the simple-kwarg path
    * ``dict`` — single homogeneous vehicle type
    * ``list[dict]`` — heterogeneous fleet, one dict per type

    A vehicle dict can use any of the keys: capacity, fixed_cost,
    variable_cost, velocity, count (or fleet_size).
    """
    types: List[Dict[str, Any]] = []

    if vehicles is None:
        # Build a single homogeneous spec from individual kwargs.
        cap = _ensure_iterable(capacity, default=[float("inf")])
        fc = _ensure_iterable(fixed_cost, default=[0.0] * len(cap))
        vc = _ensure_iterable(variable_cost, default=[1.0] * len(cap))
        vel = _ensure_iterable(velocity, default=[1.0] * len(cap))
        fs = _ensure_iterable(fleet_size, default=[])

        for i in range(len(cap)):
            types.append({
                "capacity": float(cap[i]),
                "fixed_cost": float(fc[i]) if i < len(fc) else 0.0,
                "variable_cost": float(vc[i]) if i < len(vc) else 1.0,
                "velocity": float(vel[i]) if i < len(vel) else 1.0,
            })
        fleet_size_list = [int(x) for x in fs]
    else:
        if isinstance(vehicles, Mapping):
            vehicles_list = [vehicles]
        else:
            vehicles_list = list(vehicles)
        fleet_size_list = []
        for v in vehicles_list:
            types.append({
                "capacity": float(v.get("capacity", float("inf"))),
                "fixed_cost": float(v.get("fixed_cost", 0.0)),
                "variable_cost": float(v.get("variable_cost", 1.0)),
                "velocity": float(v.get("velocity", 1.0)),
            })
            count = v.get("count", v.get("fleet_size", None))
            if count is not None:
                fleet_size_list.append(int(count))
        # If only some types specified count, fall back to infinite for everyone.
        if len(fleet_size_list) != len(types):
            fleet_size_list = []

    return {
        "capacity": [t["capacity"] for t in types],
        "fixed_cost": [t["fixed_cost"] for t in types],
        "variable_cost": [t["variable_cost"] for t in types],
        "velocity": [t["velocity"] for t in types],
        "fleet_size": fleet_size_list,
        "vehicle_types": len(types),
    }


def _ensure_iterable(x, default):
    if x is None:
        return list(default)
    if isinstance(x, (int, float)):
        return [float(x)]
    return list(x)


@dataclass
class Problem:
    """Internal representation of a routing problem.

    Users typically don't construct this directly — they call
    :func:`routeforge.solve`, which builds a Problem under the hood.
    The class is exposed for advanced workflows (custom plotting,
    serialization, debugging).
    """

    coordinates: np.ndarray
    distance_matrix: np.ndarray
    parameters: np.ndarray  # (n, 5) — demand, tw_early, tw_late, tw_service, tw_wait_cost
    n_depots: int
    capacity: List[float]
    fixed_cost: List[float]
    variable_cost: List[float]
    velocity: List[float]
    fleet_size: List[int]
    vehicle_types: int
    time_window: str          # 'with' or 'without'
    route: str                # 'open' or 'closed'
    model: str                # 'tsp', 'mtsp', 'vrp'
    location_labels: Optional[List[str]] = field(default=None)

    @classmethod
    def from_locations(
        cls,
        locations: LocationsInput,
        n_depots: int = 1,
        vehicles: Optional[VehicleSpec] = None,
        *,
        capacity: Optional[Union[float, Sequence[float]]] = None,
        fixed_cost: Optional[Union[float, Sequence[float]]] = None,
        variable_cost: Optional[Union[float, Sequence[float]]] = None,
        velocity: Optional[Union[float, Sequence[float]]] = None,
        fleet_size: Optional[Union[int, Sequence[int]]] = None,
        demand: Optional[Sequence[float]] = None,
        model: str = "vrp",
        route: str = "closed",
        time_window: str = "auto",
        labels: Optional[Sequence[str]] = None,
    ) -> "Problem":
        df = _coerce_locations_to_df(locations)

        x_col = _find_column(df, _X_COLS) or df.columns[0]
        y_col = _find_column(df, _Y_COLS) or df.columns[1]
        coords = df[[x_col, y_col]].to_numpy(dtype=np.float64)

        # Auto-detect time-window mode if requested.
        has_tw_cols = any(_find_column(df, s) is not None for s in
                          (_TW_EARLY_COLS, _TW_LATE_COLS,
                           _TW_SERVICE_COLS, _TW_WAIT_COST_COLS))
        if time_window == "auto":
            tw = "with" if has_tw_cols else "without"
        else:
            tw = time_window

        params = _normalize_parameters(
            n_nodes=coords.shape[0],
            df=df,
            demand=demand,
            has_time_windows=(tw == "with"),
        )
        dm = build_distance_matrix(coords)

        veh = _normalize_vehicles(vehicles, capacity, fixed_cost,
                                  variable_cost, velocity, fleet_size)

        # Resolve labels from a 'name'/'label'/'id' column if the user didn't
        # pass them explicitly.
        if labels is None:
            for cand in ("name", "label", "id", "Name", "Label", "ID"):
                if cand in df.columns:
                    labels = df[cand].astype(str).tolist()
                    break
        return cls(
            coordinates=coords,
            distance_matrix=dm,
            parameters=params,
            n_depots=n_depots,
            time_window=tw,
            route=route,
            model=model,
            location_labels=list(labels) if labels is not None else None,
            **veh,
        )

    @classmethod
    def from_distance_matrix(
        cls,
        distance_matrix: np.ndarray,
        n_depots: int = 1,
        vehicles: Optional[VehicleSpec] = None,
        *,
        capacity: Optional[Union[float, Sequence[float]]] = None,
        fixed_cost: Optional[Union[float, Sequence[float]]] = None,
        variable_cost: Optional[Union[float, Sequence[float]]] = None,
        velocity: Optional[Union[float, Sequence[float]]] = None,
        fleet_size: Optional[Union[int, Sequence[int]]] = None,
        demand: Optional[Sequence[float]] = None,
        time_windows: Optional[pd.DataFrame] = None,
        model: str = "vrp",
        route: str = "closed",
        labels: Optional[Sequence[str]] = None,
    ) -> "Problem":
        dm = np.asarray(distance_matrix, dtype=np.float64)
        if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
            raise ValueError(f"distance_matrix must be square; got shape {dm.shape}.")

        coords = build_coordinates(dm)  # 2D embedding for plotting
        has_tw = time_windows is not None
        params = _normalize_parameters(
            n_nodes=dm.shape[0],
            df=time_windows,
            demand=demand,
            has_time_windows=has_tw,
        )

        veh = _normalize_vehicles(vehicles, capacity, fixed_cost,
                                  variable_cost, velocity, fleet_size)

        return cls(
            coordinates=coords,
            distance_matrix=dm,
            parameters=params,
            n_depots=n_depots,
            time_window="with" if has_tw else "without",
            route=route,
            model=model,
            location_labels=list(labels) if labels is not None else None,
            **veh,
        )
