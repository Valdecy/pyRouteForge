"""
Low-level route-cost kernels.

These are the only hot-path functions in routeforge — the GA evaluates a
single route's penalised cost millions of times per run, so we keep the
work tight and let numba JIT-compile when available. When numba is not
installed everything still runs (just slower); the public API is identical.
"""

from __future__ import annotations

import numpy as np

try:  # numba is an optional accelerator
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op fallback decorator with the same call signatures as numba.njit."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func):
            return func
        return _decorator


@njit(cache=True)
def route_cost_basic(distance_matrix, demand, depot_idx, subroute,
                     fixed_cost, variable_cost, capacity_limit,
                     penalty_value, route_closed):
    """Penalised cost of a single route (no time windows).

    Parameters
    ----------
    distance_matrix : 2D float array
    demand : 1D float array, indexed by node
    depot_idx : int
    subroute : 1D int array of client indices in visit order
    fixed_cost, variable_cost : float (for the chosen vehicle)
    capacity_limit : float (vehicle capacity)
    penalty_value : float (per-violation penalty)
    route_closed : int (1 -> return to depot, 0 -> open route)
    """
    n = subroute.shape[0]
    if n == 0:
        return 0.0

    total_dist = 0.0
    cum_cap = 0.0
    pnlt = 0
    prev = depot_idx

    for i in range(n):
        node = subroute[i]
        total_dist += distance_matrix[prev, node]
        cum_cap += demand[node]
        if cum_cap > capacity_limit:
            pnlt += 1
        prev = node

    if route_closed == 1:
        total_dist += distance_matrix[prev, depot_idx]

    return fixed_cost + total_dist * variable_cost + pnlt * penalty_value


@njit(cache=True)
def route_cost_time_windows(distance_matrix, demand, tw_early, tw_late, tw_st, tw_wc,
                            depot_idx, subroute, velocity, fixed_cost,
                            variable_cost, capacity_limit, penalty_value, route_closed):
    """Penalised cost of a single route with time windows.

    Adds waiting cost (proportional to wait time at each node) and a
    per-violation penalty when the vehicle arrives after ``tw_late``.
    """
    n = subroute.shape[0]
    if n == 0:
        return 0.0

    total_dist = 0.0
    wait_total_cost = 0.0
    cum_cap = 0.0
    pnlt = 0
    t = 0.0
    prev = depot_idx

    for i in range(n):
        node = subroute[i]
        leg = distance_matrix[prev, node]
        total_dist += leg
        t += leg / velocity

        if t < tw_early[node]:
            wait = tw_early[node] - t
            wait_total_cost += wait * tw_wc[node]
            t = tw_early[node]

        t += tw_st[node]
        if t > tw_late[node] + tw_st[node]:
            pnlt += 1

        cum_cap += demand[node]
        if cum_cap > capacity_limit:
            pnlt += 1

        prev = node

    if route_closed == 1:
        leg = distance_matrix[prev, depot_idx]
        total_dist += leg
        t += leg / velocity

        if t < tw_early[depot_idx]:
            wait = tw_early[depot_idx] - t
            wait_total_cost += wait * tw_wc[depot_idx]
            t = tw_early[depot_idx]

        t += tw_st[depot_idx]
        if t > tw_late[depot_idx] + tw_st[depot_idx]:
            pnlt += 1

    return (fixed_cost + total_dist * variable_cost
            + wait_total_cost + pnlt * penalty_value)


def build_distance_matrix(coordinates):
    """Euclidean distance matrix from a coordinate array of shape (n, 2)."""
    coords = np.asarray(coordinates, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))


def build_coordinates(distance_matrix):
    """Reconstruct a 2D embedding from a distance matrix (classical MDS).

    Useful when the user only has a distance matrix — we still want a
    plottable 2D layout for visualization. This is an approximation; if
    the matrix is non-Euclidean, negative eigenvalues are clipped to 0.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    a = dm[0, :].reshape(-1, 1)
    b = dm[:, 0].reshape(1, -1)
    m = 0.5 * (a ** 2 + b ** 2 - dm ** 2)
    w, u = np.linalg.eig(m.T @ m)
    # Clip tiny negative eigenvalues from numerical noise before taking sqrt.
    w_sorted = np.sort(w.real)[::-1]
    w_clipped = np.clip(w_sorted, 0.0, None)
    s = np.diag(w_clipped) ** 0.5
    return (u @ (s ** 0.5)).real[:, :2]
