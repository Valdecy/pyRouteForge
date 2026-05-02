"""
Hybrid genetic search core for routing problems (optimized).

Public API
----------
``run_genetic_algorithm`` keeps its previous signature and return type
(report DataFrame, raw [[depots], [stops], [vehicles]] layout, history).

Generic by construction
-----------------------
A single decoder + local search handles every variant the user asked for:

    * TSP / mTSP          — capacities forced infinite, fleet count optionally fixed
    * Capacitated VRP     — per-vehicle capacity enforced via penalty + repair
    * Multi-Depot VRP     — segment cost matrix vectorised over depots; each
                            split picks the best depot for that segment
    * VRP with Time Windows — earliest/latest/service/wait-cost from `parameters`
    * Heterogeneous Fleet — segment cost matrix vectorised over vehicle types
    * Finite / Infinite Fleet — empty `fleet_size` -> infinite; otherwise a
                                bounded-K dynamic program plus a swap-repair
                                pass enforces per-type counts
    * Open / Closed routes — the closing leg back to the depot is added (or not)
                              uniformly through every cost computation

Speed strategy
--------------
* Pre-bundled `_Ctx` of contiguous numpy arrays: no `parameters[:, k]` in hot
  loops, no `math.isfinite` per stop, no `round()` for comparison keys.
* `_segment_cost_matrix_no_tw` is fully vectorised over (depot, vehicle) and
  uses prefix sums; cost of every segment is found without allocating a
  RouteState per cell. The TW variant walks per (depot, vehicle) but
  broadcasts the inner update with numpy.
* Neighbour lists (k-nearest customers) prune inter-route relocate / swap.
* 2-opt for non-TW uses an O(L) numpy delta vector per pivot, not a full
  re-evaluation per pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time as tm
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Legacy report helpers (used by plotting, kept byte-compatible)
# ---------------------------------------------------------------------------

def _evaluate_distance(distance_matrix, depot, subroute):
    d = depot[0]
    n = len(subroute)
    if n == 0:
        return [0.0, 0.0]
    path = np.empty(n + 2, dtype=np.int64)
    path[0] = d
    path[1:n + 1] = subroute
    path[-1] = d
    seg = distance_matrix[path[:-1], path[1:]]
    return [0.0] + np.cumsum(seg).tolist()


def _evaluate_time(distance_matrix, parameters, depot, subroute, velocity):
    tw_early = parameters[:, 1]
    tw_st = parameters[:, 3]
    d = depot[0]
    vel = max(velocity[0], 1e-12)
    nodes = [d] + list(subroute) + [d]
    L = len(nodes)
    wait = [0.0] * L
    time = [0.0] * L
    for i in range(1, L):
        prev = nodes[i - 1]
        cur = nodes[i]
        t = time[i - 1] + distance_matrix[prev, cur] / vel
        if t < tw_early[cur]:
            wait[i] = tw_early[cur] - t
            t = tw_early[cur]
        time[i] = t + tw_st[cur]
    return wait, time


def _evaluate_capacity(parameters, depot, subroute):
    demand = parameters[:, 0]
    if not subroute:
        return [0.0, 0.0]
    idx = np.asarray([depot[0]] + list(subroute) + [depot[0]], dtype=np.int64)
    return np.cumsum(demand[idx]).tolist()


def _evaluate_cost(dist, wait, parameters, depot, subroute,
                   fixed_cost, variable_cost, time_window):
    tw_wc = parameters[:, 4]
    subroute_ = depot + subroute + depot
    fc, vc = fixed_cost[0], variable_cost[0]
    if time_window == "with":
        return [
            fc + wait[i] * tw_wc[subroute_[i]] if dist[i] == 0
            else fc + dist[i] * vc + wait[i] * tw_wc[subroute_[i]]
            for i in range(len(subroute_))
        ]
    return [fc if x == 0 else fc + x * vc for x in dist]


def _build_report(solution, distance_matrix, parameters, velocity, fixed_cost,
                  variable_cost, route, time_window):
    columns = ["Route", "Vehicle", "Activity", "Job", "Arrive_Load", "Leave_Load",
               "Wait_Time", "Arrive_Time", "Leave_Time", "Distance", "Costs"]
    tt = td = tc = 0.0
    tw_st = parameters[:, 3]
    rows: list = []

    for i in range(len(solution[1])):
        dist = _evaluate_distance(distance_matrix, solution[0][i], solution[1][i])
        wait, time = _evaluate_time(distance_matrix, parameters, solution[0][i],
                                    solution[1][i],
                                    velocity=[velocity[solution[2][i][0]]])
        rev = solution[1][i][::-1]
        cap = _evaluate_capacity(parameters, solution[0][i], rev); cap.reverse()
        leave_cap = cap[:]
        for n in range(1, len(leave_cap) - 1):
            leave_cap[n] = cap[n + 1]
        cost = _evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i],
                              fixed_cost=[fixed_cost[solution[2][i][0]]],
                              variable_cost=[variable_cost[solution[2][i][0]]],
                              time_window=time_window)
        if route == "closed":
            subroute = [solution[0][i] + solution[1][i] + solution[0][i]]
        else:
            subroute = [solution[0][i] + solution[1][i]]
        for j in range(len(subroute[0])):
            if j == 0:
                activity = "start"
                arrive_time = round(time[j], 2)
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j], 2)
            if 0 < j < len(subroute[0]) - 1:
                activity = "service"
            if j == len(subroute[0]) - 1:
                activity = "finish"
                if time[j] > tt:
                    tt = time[j]
                td += dist[j]
                tc += cost[j]
            rows.append([f"#{i + 1}", solution[2][i][0], activity, subroute[0][j],
                         cap[j], leave_cap[j], round(wait[j], 2),
                         arrive_time, round(time[j], 2),
                         round(dist[j], 2), round(cost[j], 2)])
        rows.append(["-//-"] * 11)
    rows.append(["MAX TIME", "", "", "", "", "", "", "", round(tt, 2), "", ""])
    rows.append(["TOTAL", "", "", "", "", "", "", "", "", round(td, 2), round(tc, 2)])
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Internal context & data classes
# ---------------------------------------------------------------------------

class _Ctx:
    """Bundle of preprocessed arrays + scalar settings.

    Built once per `run_genetic_algorithm` call. Avoids re-fetching the
    same column slices and re-checking the same booleans inside hot loops.
    """
    __slots__ = (
        "D", "demand", "tw_early", "tw_late", "tw_service", "tw_wait_cost",
        "velocity", "fixed_cost", "variable_cost", "capacity", "cap_finite",
        "n_depots", "depots", "n_vehicle_types",
        "route_mode", "tw_enabled", "closed",
        "cap_penalty", "tw_penalty",
        "model", "fleet_size",
        "neighbors",
    )

    def __init__(self, distance_matrix, parameters, velocity, fixed_cost,
                 variable_cost, capacity, n_depots, route_mode, tw_enabled,
                 model, fleet_size, cap_penalty, tw_penalty):
        self.D = np.ascontiguousarray(distance_matrix, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        self.demand = np.ascontiguousarray(params[:, 0])
        self.tw_early = np.ascontiguousarray(params[:, 1])
        self.tw_late = np.ascontiguousarray(params[:, 2])
        self.tw_service = np.ascontiguousarray(params[:, 3])
        self.tw_wait_cost = np.ascontiguousarray(params[:, 4])
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.fixed_cost = np.asarray(fixed_cost, dtype=np.float64)
        self.variable_cost = np.asarray(variable_cost, dtype=np.float64)
        self.capacity = np.asarray(capacity, dtype=np.float64)
        self.cap_finite = np.isfinite(self.capacity)
        self.n_depots = int(n_depots)
        self.depots = np.arange(self.n_depots, dtype=np.int64)
        self.n_vehicle_types = len(capacity)
        self.route_mode = route_mode
        self.closed = (route_mode == "closed")
        self.tw_enabled = bool(tw_enabled)
        self.model = model
        self.fleet_size = list(fleet_size) if fleet_size else []
        self.cap_penalty = float(cap_penalty)
        self.tw_penalty = float(tw_penalty)
        self.neighbors: dict = {}

    def update_penalties(self, cap_penalty: float, tw_penalty: float):
        self.cap_penalty = float(cap_penalty)
        self.tw_penalty = float(tw_penalty)


class RouteState:
    __slots__ = ("depot", "vehicle", "stops", "distance", "cost",
                 "wait_cost", "cap_violation", "tw_violation", "feasible")

    def __init__(self, depot, vehicle, stops, distance, cost,
                 wait_cost, cap_violation, tw_violation, feasible):
        self.depot = depot
        self.vehicle = vehicle
        self.stops = stops
        self.distance = distance
        self.cost = cost
        self.wait_cost = wait_cost
        self.cap_violation = cap_violation
        self.tw_violation = tw_violation
        self.feasible = feasible


@dataclass
class Candidate:
    perm: List[int]
    raw: List[List[List[int]]]
    total_distance: float
    total_cost: float
    cap_violation: float
    tw_violation: float
    fleet_violation: float
    feasible: bool
    n_routes: int

    @property
    def violation(self) -> float:
        return self.cap_violation + self.tw_violation + self.fleet_violation

    @property
    def sort_key(self):
        # Use plain values; avoid round() in hot paths by relying on the
        # later total_cost / total_distance dominance.
        return (
            0 if self.feasible else 1,
            self.violation,
            self.total_cost,
            self.total_distance,
            len(self.perm),
        )


# ---------------------------------------------------------------------------
# Fast single-route evaluation
# ---------------------------------------------------------------------------

def _eval_route(stops, depot: int, vehicle: int, ctx: _Ctx) -> RouteState:
    """Evaluate a single route. `stops` may be list/tuple/ndarray."""
    if isinstance(stops, np.ndarray):
        stops_list = stops.tolist()
    else:
        stops_list = list(stops)
    n = len(stops_list)
    if n == 0:
        return RouteState(int(depot), int(vehicle), [], 0.0, 0.0, 0.0, 0.0, 0.0, True)

    D = ctx.D
    closed = ctx.closed
    vel = max(float(ctx.velocity[vehicle]), 1e-12)
    cap_limit = float(ctx.capacity[vehicle])
    cap_finite = bool(ctx.cap_finite[vehicle])

    # Distance + load: pure-Python scalar loop avoids numpy roundtrip overhead
    # which dominates per-call cost for typical route sizes (<~80 stops).
    prev = depot
    total_distance = 0.0
    load_total = 0.0
    demand = ctx.demand
    if cap_finite:
        for s in stops_list:
            total_distance += float(D[prev, s])
            load_total += float(demand[s])
            prev = s
    else:
        for s in stops_list:
            total_distance += float(D[prev, s])
            prev = s
    if closed:
        total_distance += float(D[prev, depot])

    cap_violation = max(0.0, load_total - cap_limit) if cap_finite else 0.0

    wait_cost_total = 0.0
    tw_violation = 0.0
    if ctx.tw_enabled:
        tw_early = ctx.tw_early
        tw_late = ctx.tw_late
        tw_service = ctx.tw_service
        tw_wait_cost = ctx.tw_wait_cost
        cur = 0.0
        prev = depot
        for s in stops_list:
            cur += float(D[prev, s]) / vel
            ear = float(tw_early[s])
            if cur < ear:
                wait_cost_total += (ear - cur) * float(tw_wait_cost[s])
                cur = ear
            late = float(tw_late[s])
            if cur > late:
                tw_violation += cur - late
            cur += float(tw_service[s])
            prev = s
        if closed:
            cur += float(D[prev, depot]) / vel
            ear = float(tw_early[depot])
            if cur < ear:
                wait_cost_total += (ear - cur) * float(tw_wait_cost[depot])
                cur = ear
            late = float(tw_late[depot])
            if cur > late:
                tw_violation += cur - late

    total_cost = (
        float(ctx.fixed_cost[vehicle])
        + total_distance * float(ctx.variable_cost[vehicle])
        + wait_cost_total
        + ctx.cap_penalty * cap_violation
        + ctx.tw_penalty * tw_violation
    )
    feasible = (cap_violation <= 1e-9 and tw_violation <= 1e-9)
    return RouteState(
        depot, vehicle, stops_list,
        total_distance, total_cost, wait_cost_total,
        cap_violation, tw_violation, feasible,
    )


def _route_better(a: RouteState, b: RouteState) -> bool:
    ka = (0 if a.feasible else 1, a.cap_violation + a.tw_violation, a.cost, a.distance)
    kb = (0 if b.feasible else 1, b.cap_violation + b.tw_violation, b.cost, b.distance)
    return ka < kb


def _pair_better(new_a, new_b, old_a, old_b) -> bool:
    new_feas = new_a.feasible and new_b.feasible
    old_feas = old_a.feasible and old_b.feasible
    nv = new_a.cap_violation + new_a.tw_violation + new_b.cap_violation + new_b.tw_violation
    ov = old_a.cap_violation + old_a.tw_violation + old_b.cap_violation + old_b.tw_violation
    nc = new_a.cost + new_b.cost
    oc = old_a.cost + old_b.cost
    return (0 if new_feas else 1, nv, nc) < (0 if old_feas else 1, ov, oc)


# ---------------------------------------------------------------------------
# Vectorised segment cost matrix
# ---------------------------------------------------------------------------

def _segment_cost_no_tw(perm_arr: np.ndarray, ctx: _Ctx):
    """O(n^2) work over (i, j); the (depot, vehicle) selection per segment is
    fully vectorised. Returns
        cost[i, j], best_d[i, j], best_v[i, j]
    where the segment is perm[i..j] (inclusive, i <= j).
    """
    n = perm_arr.size
    D = ctx.D
    nD = ctx.n_depots
    nV = ctx.n_vehicle_types

    # cum_inner[k] = sum_{m=0..k-1} D[perm[m], perm[m+1]]
    cum_inner = np.zeros(n, dtype=np.float64)
    if n >= 2:
        inner = D[perm_arr[:-1], perm_arr[1:]]
        cum_inner[1:] = np.cumsum(inner)

    # cum_dem[k] = sum demand[perm[0..k-1]]
    cum_dem = np.empty(n + 1, dtype=np.float64)
    cum_dem[0] = 0.0
    if n >= 1:
        cum_dem[1:] = np.cumsum(ctx.demand[perm_arr])

    # entry leg D[depot_d, perm[i]] -> shape (nD, n)
    d_first = D[ctx.depots[:, None], perm_arr[None, :]]
    if ctx.closed:
        d_last = D[perm_arr[None, :], ctx.depots[:, None]]  # (nD, n) last->depot
    else:
        d_last = np.zeros_like(d_first)

    cap = ctx.capacity                  # (nV,)
    cap_finite = ctx.cap_finite         # (nV,) bool
    fc = ctx.fixed_cost                 # (nV,)
    vc = ctx.variable_cost              # (nV,)
    cap_pen = ctx.cap_penalty

    cost = np.full((n, n), np.inf, dtype=np.float64)
    best_d = np.zeros((n, n), dtype=np.int32)
    best_v = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        idxs = np.arange(i, n)
        m = idxs.size

        # Distance per (depot, j) candidate
        internal = cum_inner[idxs] - cum_inner[i]               # (m,)
        seg_dist = d_first[:, i:i + 1] + internal[None, :] + d_last[:, idxs]  # (nD, m)

        # Demand per j candidate; capacity violation per (vehicle, j)
        demand_sum = cum_dem[idxs + 1] - cum_dem[i]             # (m,)
        diff = demand_sum[None, :] - cap[:, None]               # (nV, m)
        cap_v = np.where(cap_finite[:, None], np.maximum(0.0, diff), 0.0)

        # cost shape (nD, nV, m). Memory is bounded — typical fleets are small.
        cost_dvm = (
            fc[None, :, None]
            + seg_dist[:, None, :] * vc[None, :, None]
            + cap_pen * cap_v[None, :, :]
        )
        flat = cost_dvm.reshape(nD * nV, m)
        am = flat.argmin(axis=0)
        chosen = flat[am, np.arange(m)]
        cd = (am // nV).astype(np.int32)
        cv = (am % nV).astype(np.int32)

        cost[i, idxs] = chosen
        best_d[i, idxs] = cd
        best_v[i, idxs] = cv

    return cost, best_d, best_v


def _segment_cost_tw(perm_arr: np.ndarray, ctx: _Ctx):
    """O(n^2) outer work; per-(depot, vehicle) walk because waiting at TW
    introduces a non-additive recurrence.

    Two implementations:
      * fast scalar path for nD == 1 and nV == 1 (the common case for TSP,
        mTSP, CVRP, VRPTW, open-route, finite-fleet single-depot)
      * flat vectorised path for nD * nV > 1 (MDVRP, heterogeneous fleet)
    """
    n = perm_arr.size
    D = ctx.D
    nD = ctx.n_depots
    nV = ctx.n_vehicle_types
    closed = ctx.closed
    cost = np.full((n, n), np.inf, dtype=np.float64)
    best_d = np.zeros((n, n), dtype=np.int32)
    best_v = np.zeros((n, n), dtype=np.int32)

    # Pull arrays / scalars locally
    cap = ctx.capacity
    cap_finite = ctx.cap_finite
    fc = ctx.fixed_cost
    vc = ctx.variable_cost
    vel_arr = np.maximum(ctx.velocity, 1e-12)
    cap_pen = ctx.cap_penalty
    tw_pen = ctx.tw_penalty
    tw_early = ctx.tw_early
    tw_late = ctx.tw_late
    tw_service = ctx.tw_service
    tw_wait_cost = ctx.tw_wait_cost
    depots = ctx.depots
    demand = ctx.demand

    # Pre-pull D rows for nodes we will visit -- single fancy index
    perm_int = perm_arr.astype(np.int64, copy=False)

    if nD == 1 and nV == 1:
        # ---------- Scalar fast path ----------
        depot = int(depots[0])
        inv_vel = 1.0 / float(vel_arr[0])
        cap_lim = float(cap[0])
        cap_fin = bool(cap_finite[0])
        fc0 = float(fc[0])
        vc0 = float(vc[0])
        depot_back = float(D[depot, depot]) if False else None  # unused; keeps depot leg below
        ear_d = float(tw_early[depot])
        late_d = float(tw_late[depot])
        wc_d = float(tw_wait_cost[depot])

        # Pre-extract the row D[depot, :] and per-node arrays
        D_from_depot = D[depot]
        D_to_depot = D[:, depot]
        # Convert numpy scalar arrays to .item-ready python lookups
        for i in range(n):
            time_s = 0.0
            load_s = 0.0
            dist_s = 0.0
            wait_cost_s = 0.0
            tw_viol_s = 0.0
            prev = depot
            for j in range(i, n):
                node = int(perm_int[j])
                leg = float(D[prev, node])
                arrival = time_s + leg * inv_vel
                ear = float(tw_early[node])
                if arrival < ear:
                    wait_cost_s += (ear - arrival) * float(tw_wait_cost[node])
                    start_service = ear
                else:
                    start_service = arrival
                late = float(tw_late[node])
                if start_service > late:
                    tw_viol_s += start_service - late
                time_s = start_service + float(tw_service[node])
                dist_s += leg
                load_s += float(demand[node])
                prev = node

                if closed:
                    back_leg = float(D_to_depot[node])
                    back_arrival = time_s + back_leg * inv_vel
                    if back_arrival < ear_d:
                        bwc = (ear_d - back_arrival) * wc_d
                        start_back = ear_d
                    else:
                        bwc = 0.0
                        start_back = back_arrival
                    blate = max(0.0, start_back - late_d)
                    total_dist = dist_s + back_leg
                    total_wc = wait_cost_s + bwc
                    total_tw = tw_viol_s + blate
                else:
                    total_dist = dist_s
                    total_wc = wait_cost_s
                    total_tw = tw_viol_s

                cap_v = max(0.0, load_s - cap_lim) if cap_fin else 0.0
                tot = (
                    fc0
                    + total_dist * vc0
                    + total_wc
                    + cap_pen * cap_v
                    + tw_pen * total_tw
                )
                cost[i, j] = tot
                # best_d, best_v already 0
        return cost, best_d, best_v

    # ---------- Multi-track flat-vector path ----------
    T = nD * nV
    # Build per-track depot index and inv_velocity
    track_depot = np.repeat(depots, nV).astype(np.int64)            # (T,)
    track_invvel = np.tile(1.0 / vel_arr, nD)                       # (T,)
    track_fc = np.tile(fc, nD)                                       # (T,)
    track_vc = np.tile(vc, nD)                                       # (T,)
    track_cap = np.tile(cap, nD)                                     # (T,)
    track_cap_fin = np.tile(cap_finite, nD)                          # (T,)
    # depot-tw arrays per track
    track_d_early = tw_early[track_depot]
    track_d_late = tw_late[track_depot]
    track_d_wc = tw_wait_cost[track_depot]

    prev = np.empty(T, dtype=np.int64)
    time_s = np.empty(T, dtype=np.float64)
    load_s = np.empty(T, dtype=np.float64)
    dist_s = np.empty(T, dtype=np.float64)
    wait_cost_s = np.empty(T, dtype=np.float64)
    tw_viol_s = np.empty(T, dtype=np.float64)

    for i in range(n):
        prev[:] = track_depot
        time_s[:] = 0.0
        load_s[:] = 0.0
        dist_s[:] = 0.0
        wait_cost_s[:] = 0.0
        tw_viol_s[:] = 0.0

        for j in range(i, n):
            node = int(perm_int[j])
            leg = D[prev, node]                              # (T,)
            arrival = time_s + leg * track_invvel
            ear = tw_early[node]
            late = tw_late[node]
            wait = np.maximum(0.0, ear - arrival)
            wait_cost_s = wait_cost_s + wait * tw_wait_cost[node]
            start_service = arrival + wait
            tw_viol_s = tw_viol_s + np.maximum(0.0, start_service - late)
            time_s = start_service + tw_service[node]
            dist_s = dist_s + leg
            load_s = load_s + demand[node]
            prev[:] = node

            if closed:
                back_leg = D[node, track_depot]              # (T,)
                back_arrival = time_s + back_leg * track_invvel
                back_wait = np.maximum(0.0, track_d_early - back_arrival)
                back_wait_cost = back_wait * track_d_wc
                start_back = back_arrival + back_wait
                back_late = np.maximum(0.0, start_back - track_d_late)
                total_dist = dist_s + back_leg
                total_wc = wait_cost_s + back_wait_cost
                total_tw = tw_viol_s + back_late
            else:
                total_dist = dist_s
                total_wc = wait_cost_s
                total_tw = tw_viol_s

            cap_v = np.where(track_cap_fin,
                             np.maximum(0.0, load_s - track_cap),
                             0.0)
            tot = (track_fc + total_dist * track_vc + total_wc
                   + cap_pen * cap_v + tw_pen * total_tw)
            am = int(tot.argmin())
            cost[i, j] = float(tot[am])
            best_d[i, j] = am // nV
            best_v[i, j] = am % nV

    return cost, best_d, best_v


# ---------------------------------------------------------------------------
# Split decoder via DP over the segment cost matrix
# ---------------------------------------------------------------------------

def _decode_split(perm: Sequence[int], ctx: _Ctx) -> List[RouteState]:
    perm_arr = np.asarray(perm, dtype=np.int64)
    n = perm_arr.size
    if n == 0:
        return []

    if ctx.tw_enabled:
        seg_cost, best_d, best_v = _segment_cost_tw(perm_arr, ctx)
    else:
        seg_cost, best_d, best_v = _segment_cost_no_tw(perm_arr, ctx)

    fleet_size = ctx.fleet_size
    model = ctx.model

    exact_routes: Optional[int] = None
    max_routes: Optional[int] = None
    if fleet_size:
        max_routes = max(1, min(int(sum(fleet_size)), n))
        if model == "mtsp":
            exact_routes = max_routes
    if model == "tsp":
        exact_routes = 1
        max_routes = 1

    if max_routes is None:
        # Unbounded number of routes
        dp = np.full(n + 1, np.inf, dtype=np.float64)
        prev_idx = np.full(n + 1, -1, dtype=np.int64)
        dp[0] = 0.0
        for j in range(1, n + 1):
            costs = dp[:j] + seg_cost[:j, j - 1]
            i = int(np.argmin(costs))
            dp[j] = costs[i]
            prev_idx[j] = i
        cuts: List[Tuple[int, int]] = []
        cur = n
        while cur > 0:
            i = int(prev_idx[cur])
            cuts.append((i, cur - 1))
            cur = i
        cuts.reverse()
    else:
        # Bounded-K DP. dp[k][j] = min cost using exactly k routes covering
        # the first j perm entries.
        dp = np.full((max_routes + 1, n + 1), np.inf, dtype=np.float64)
        prev_i = np.full((max_routes + 1, n + 1), -1, dtype=np.int64)
        dp[0, 0] = 0.0
        for k in range(1, max_routes + 1):
            prev_layer = dp[k - 1, :]
            for j in range(1, n + 1):
                vals = prev_layer[:j] + seg_cost[:j, j - 1]
                ii = int(np.argmin(vals))
                v = vals[ii]
                if np.isfinite(v):
                    dp[k, j] = v
                    prev_i[k, j] = ii

        if exact_routes is not None:
            best_k = exact_routes
        else:
            ks = np.arange(1, max_routes + 1)
            col = dp[1:max_routes + 1, n]
            valid = np.isfinite(col)
            if valid.any():
                masked = np.where(valid, col, np.inf)
                best_k = int(ks[int(np.argmin(masked))])
            else:
                best_k = max_routes

        cuts = []
        cur = n
        k = best_k
        while cur > 0 and k >= 1:
            ii = int(prev_i[k, cur])
            if ii < 0:
                break
            cuts.append((ii, cur - 1))
            cur = ii
            k -= 1
        cuts.reverse()
        if not cuts:
            return [_eval_route(perm_arr, int(ctx.depots[0]), 0, ctx)]

    routes: List[RouteState] = []
    for (i, j) in cuts:
        sub = perm_arr[i:j + 1]
        d = int(best_d[i, j])
        v = int(best_v[i, j])
        routes.append(_eval_route(sub, d, v, ctx))
    return routes


# ---------------------------------------------------------------------------
# Local search: 2-opt (vectorised when no TW), inter-route relocate / swap
# ---------------------------------------------------------------------------

def _two_opt_no_tw(route: RouteState, ctx: _Ctx) -> RouteState:
    """Best-improvement 2-opt with O(L) numpy delta per pivot.
    Capacity is invariant under reversal, so we only check distance.
    """
    stops = list(route.stops)
    n = len(stops)
    if n < 2:
        return route
    D = ctx.D
    depot = route.depot
    if ctx.closed:
        path = np.empty(n + 2, dtype=np.int64)
        path[0] = depot
        path[1:n + 1] = stops
        path[-1] = depot
    else:
        path = np.empty(n + 1, dtype=np.int64)
        path[0] = depot
        path[1:] = stops
    L = path.size
    closed = ctx.closed

    while True:
        best_delta = -1e-12
        best_i = best_j = -1
        last_i = (L - 3) if closed else (L - 2)
        for i in range(1, last_i + 1):
            a = int(path[i - 1])
            b = int(path[i])
            d_ab = float(D[a, b])
            if closed:
                js = np.arange(i + 1, L - 1)
            else:
                js = np.arange(i + 1, L)
            if js.size == 0:
                continue
            c = path[js]
            # delta when there's a successor edge
            if closed or js[-1] != L - 1:
                d_nodes = path[js + 1]
                delta = D[a, c] + D[b, d_nodes] - d_ab - D[c, d_nodes]
            else:
                # open route: last j has no successor; treat that j separately
                d_nodes_full = path[js[:-1] + 1]
                delta_full = D[a, c[:-1]] + D[b, d_nodes_full] - d_ab - D[c[:-1], d_nodes_full]
                delta_tail = float(D[a, c[-1]]) - d_ab
                delta = np.empty(js.size, dtype=np.float64)
                delta[:-1] = delta_full
                delta[-1] = delta_tail
            jm = int(np.argmin(delta))
            if delta[jm] < best_delta:
                best_delta = float(delta[jm])
                best_i = i
                best_j = int(js[jm])
        if best_i < 0:
            break
        path[best_i:best_j + 1] = path[best_i:best_j + 1][::-1]

    new_stops = path[1:-1].tolist() if closed else path[1:].tolist()
    if new_stops == stops:
        return route
    return _eval_route(new_stops, depot, route.vehicle, ctx)


def _two_opt_tw(route: RouteState, ctx: _Ctx) -> RouteState:
    """First-improvement 2-opt that calls full _eval_route — needed when TWs
    or per-stop wait costs make moves non-local."""
    best = route
    n = len(best.stops)
    if n < 2:
        return best
    improved = True
    while improved:
        improved = False
        stops = list(best.stops)
        m = len(stops)
        for i in range(m - 1):
            for j in range(i + 1, m):
                cand = stops[:i] + stops[i:j + 1][::-1] + stops[j + 1:]
                cnd = _eval_route(cand, best.depot, best.vehicle, ctx)
                if _route_better(cnd, best):
                    best = cnd
                    improved = True
                    break
            if improved:
                break
    return best


def _insert_positions(target_stops, node, ctx: _Ctx, limit: int = 6) -> List[int]:
    n = len(target_stops)
    if n == 0:
        return [0]
    if n + 1 <= limit:
        return list(range(n + 1))
    D = ctx.D
    arr = np.asarray(target_stops, dtype=np.int64)
    dists = D[node, arr]
    order = np.argsort(dists)[:limit]
    positions = {0, n}
    for r in order.tolist():
        positions.add(r)
        positions.add(r + 1)
    return sorted(p for p in positions if 0 <= p <= n)


def _build_neighbors(ctx: _Ctx, k: int = 15) -> dict:
    """k-nearest customer neighbour list."""
    N = ctx.D.shape[0]
    nD = ctx.n_depots
    if N - nD <= 1:
        return {i: [] for i in range(nD, N)}
    customers = np.arange(nD, N, dtype=np.int64)
    sub = ctx.D[customers[:, None], customers[None, :]].copy()
    np.fill_diagonal(sub, np.inf)
    K = min(k, customers.size - 1)
    part = np.argpartition(sub, K - 1, axis=1)[:, :K]
    rows = np.arange(customers.size)[:, None]
    order = np.argsort(np.take_along_axis(sub, part, axis=1), axis=1)
    sorted_part = np.take_along_axis(part, order, axis=1)
    neighbors = customers[sorted_part]
    return {int(c): neighbors[r].tolist() for r, c in enumerate(customers)}


def _local_search(routes: List[RouteState], ctx: _Ctx) -> List[RouteState]:
    if not routes:
        return routes
    routes = [RouteState(r.depot, r.vehicle, list(r.stops), r.distance, r.cost,
                         r.wait_cost, r.cap_violation, r.tw_violation, r.feasible)
              for r in routes]

    # Intra-route 2-opt
    two_opt_cap = 35 if ctx.tw_enabled else 200
    for idx in range(len(routes)):
        if not routes[idx].stops:
            continue
        if len(routes[idx].stops) > two_opt_cap:
            continue
        if ctx.tw_enabled:
            routes[idx] = _two_opt_tw(routes[idx], ctx)
        else:
            routes[idx] = _two_opt_no_tw(routes[idx], ctx)

    neighbors = ctx.neighbors or {}
    nb_pool = 8 if not ctx.tw_enabled else 6
    pos_limit = 4 if ctx.tw_enabled else 6
    max_passes = 1 if ctx.tw_enabled else 2

    for _ in range(max_passes):
        improved = False

        # ---- Relocate node u from route a (pos i) into route b at best pos
        for a_idx in range(len(routes)):
            if improved:
                break
            ra = routes[a_idx]
            i = 0
            while i < len(ra.stops):
                u = ra.stops[i]
                base_a = ra.stops[:i] + ra.stops[i + 1:]
                # candidate destinations: routes containing a near-neighbour of u
                cand_b = [a_idx]
                for nb in neighbors.get(u, ())[:nb_pool]:
                    for bi, rb in enumerate(routes):
                        if bi == a_idx:
                            continue
                        if nb in rb.stops:
                            if bi not in cand_b:
                                cand_b.append(bi)
                            break
                if len(cand_b) < 3:
                    for bi in range(len(routes)):
                        if bi not in cand_b:
                            cand_b.append(bi)
                            if len(cand_b) >= 3:
                                break
                done = False
                for b_idx in cand_b:
                    rb = routes[b_idx]
                    if a_idx == b_idx:
                        for pos in range(len(base_a) + 1):
                            if pos == i:
                                continue
                            new_a = base_a[:pos] + [u] + base_a[pos:]
                            cnd = _eval_route(new_a, ra.depot, ra.vehicle, ctx)
                            if _route_better(cnd, ra):
                                routes[a_idx] = cnd
                                ra = cnd
                                improved = True
                                done = True
                                break
                        if done:
                            break
                    else:
                        positions = _insert_positions(rb.stops, u, ctx, limit=pos_limit)
                        cand_a = _eval_route(base_a, ra.depot, ra.vehicle, ctx)
                        for pos in positions:
                            new_b = rb.stops[:pos] + [u] + rb.stops[pos:]
                            cand_b_state = _eval_route(new_b, rb.depot, rb.vehicle, ctx)
                            if _pair_better(cand_a, cand_b_state, ra, rb):
                                routes[a_idx] = cand_a
                                routes[b_idx] = cand_b_state
                                ra = cand_a
                                improved = True
                                done = True
                                break
                        if done:
                            break
                if improved:
                    break
                i += 1
        if improved:
            continue

        # ---- Swap nodes between routes via neighbour pairs
        for a_idx in range(len(routes)):
            if improved:
                break
            ra = routes[a_idx]
            for i in range(len(ra.stops)):
                if improved:
                    break
                u = ra.stops[i]
                for nb in neighbors.get(u, ())[:nb_pool]:
                    if improved:
                        break
                    for b_idx, rb in enumerate(routes):
                        if b_idx == a_idx:
                            continue
                        if nb in rb.stops:
                            j = rb.stops.index(nb)
                            new_a = list(ra.stops); new_a[i] = nb
                            new_b = list(rb.stops); new_b[j] = u
                            cand_a = _eval_route(new_a, ra.depot, ra.vehicle, ctx)
                            cand_b = _eval_route(new_b, rb.depot, rb.vehicle, ctx)
                            if _pair_better(cand_a, cand_b, ra, rb):
                                routes[a_idx] = cand_a
                                routes[b_idx] = cand_b
                                improved = True
                                break
        if not improved:
            break

    return [r for r in routes if r.stops]


# ---------------------------------------------------------------------------
# Vehicle-count repair (finite fleet enforcement)
# ---------------------------------------------------------------------------

def _repair_fleet(routes: List[RouteState], ctx: _Ctx) -> Tuple[List[RouteState], float]:
    fleet_size = ctx.fleet_size
    if not fleet_size:
        return routes, 0.0
    counts = [0] * len(fleet_size)
    for r in routes:
        if r.vehicle < len(counts):
            counts[r.vehicle] += 1
    routes = list(routes)
    total_excess = 0.0
    while True:
        over = [i for i, c in enumerate(counts) if c > fleet_size[i]]
        under = [i for i, c in enumerate(counts) if c < fleet_size[i]]
        if not over:
            break
        if not under:
            total_excess = sum(max(0, counts[i] - fleet_size[i]) for i in range(len(counts)))
            break
        best_key = None
        best_idx = -1
        best_new = None
        for vt in over:
            for idx, r in enumerate(routes):
                if r.vehicle != vt:
                    continue
                for alt in under:
                    cand = _eval_route(r.stops, r.depot, alt, ctx)
                    delta = cand.cost - r.cost
                    key = ((0 if cand.feasible else 1), delta, cand.distance)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = idx
                        best_new = cand
        if best_idx < 0 or best_new is None:
            total_excess = sum(max(0, counts[i] - fleet_size[i]) for i in range(len(counts)))
            break
        old_v = routes[best_idx].vehicle
        counts[old_v] -= 1
        counts[best_new.vehicle] += 1
        routes[best_idx] = best_new
    return routes, total_excess


# ---------------------------------------------------------------------------
# Candidate construction & utilities
# ---------------------------------------------------------------------------

def _routes_to_raw(routes: Sequence[RouteState]) -> List[List[List[int]]]:
    depots = [[int(r.depot)] for r in routes if r.stops]
    stops = [[int(x) for x in r.stops] for r in routes if r.stops]
    vehicles = [[int(r.vehicle)] for r in routes if r.stops]
    return [depots, stops, vehicles]


def _flatten_raw(raw):
    perm: List[int] = []
    for route in raw[1]:
        perm.extend(route)
    return perm


def _clone_raw(raw):
    return [[d[:] for d in raw[0]], [r[:] for r in raw[1]], [v[:] for v in raw[2]]]


def _candidate_from_perm(perm: Sequence[int], ctx: _Ctx,
                         apply_local_search: bool = True) -> Candidate:
    routes = _decode_split(list(perm), ctx)
    if apply_local_search:
        routes = _local_search(routes, ctx)
        if ctx.model == "mtsp" and ctx.fleet_size:
            # Force exactly fleet_size routes to remain after collapsing in LS
            routes = _decode_split(_flatten_raw(_routes_to_raw(routes)), ctx)
    routes, fleet_violation = _repair_fleet(routes, ctx)

    total_distance = sum(r.distance for r in routes)
    total_cost = sum(r.cost for r in routes)
    cap_violation = sum(r.cap_violation for r in routes)
    tw_violation = sum(r.tw_violation for r in routes)
    feasible = (cap_violation <= 1e-9 and tw_violation <= 1e-9 and fleet_violation <= 1e-9)
    raw = _routes_to_raw(routes)
    return Candidate(
        perm=list(_flatten_raw(raw)),
        raw=raw,
        total_distance=float(total_distance),
        total_cost=float(total_cost + fleet_violation * ctx.cap_penalty),
        cap_violation=float(cap_violation),
        tw_violation=float(tw_violation),
        fleet_violation=float(fleet_violation),
        feasible=bool(feasible),
        n_routes=len(raw[1]),
    )


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def _tournament(population, size: int = 3) -> Candidate:
    choices = random.sample(range(len(population)), k=min(size, len(population)))
    return min((population[i] for i in choices), key=lambda x: x.sort_key)


def _roulette(population, rank_mode: bool = False) -> Candidate:
    ordered = sorted(population, key=lambda x: x.sort_key)
    if rank_mode:
        weights = np.arange(len(ordered), 0, -1, dtype=float)
    else:
        scores = np.array([c.total_cost + c.violation * 1000.0 for c in ordered],
                          dtype=float)
        scores -= scores.min()
        weights = 1.0 / (1.0 + scores)
    probs = weights / weights.sum()
    idx = int(np.random.choice(len(ordered), p=probs))
    return ordered[idx]


def _select_parent(population, selection: str) -> Candidate:
    if selection == "rw":
        return _roulette(population, rank_mode=False)
    if selection == "rank":
        return _roulette(population, rank_mode=True)
    return _tournament(population)


def _order_crossover(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
    n = len(p1)
    if n <= 1:
        return list(p1)
    a, b = sorted(random.sample(range(n), 2))
    child: List[Optional[int]] = [None] * n
    child[a:b + 1] = list(p1[a:b + 1])
    used = set(p1[a:b + 1])
    fill = [x for x in p2 if x not in used]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return [int(x) for x in child]  # type: ignore[arg-type]


def _route_based_crossover(c1: Candidate, c2: Candidate) -> List[int]:
    if not c1.raw[1] or not c2.raw[1]:
        return _order_crossover(c1.perm, c2.perm)
    donor = random.choice(c1.raw[1])
    donor_set = set(donor)
    base = [x for x in c2.perm if x not in donor_set]
    if not base:
        return list(donor)
    insert_pos = random.randint(0, len(base))
    return base[:insert_pos] + list(donor) + base[insert_pos:]


def _mutate_perm(perm: List[int]) -> List[int]:
    n = len(perm)
    if n <= 1:
        return perm
    r = random.random()
    if r < 0.34:
        i, j = random.sample(range(n), 2)
        perm[i], perm[j] = perm[j], perm[i]
    elif r < 0.67:
        i, j = random.sample(range(n), 2)
        item = perm.pop(i)
        perm.insert(j, item)
    else:
        i, j = sorted(random.sample(range(n), 2))
        perm[i:j + 1] = reversed(perm[i:j + 1])
    return perm


# ---------------------------------------------------------------------------
# Initial population heuristics
# ---------------------------------------------------------------------------

def _sweep_perm(customers, coordinates: np.ndarray, depot_idx: int = 0) -> List[int]:
    if not customers:
        return []
    depot = coordinates[depot_idx]
    angles = []
    for c in customers:
        dx = coordinates[c, 0] - depot[0]
        dy = coordinates[c, 1] - depot[1]
        angles.append((math.atan2(dy, dx), c))
    angles.sort()
    return [c for _, c in angles]


def _greedy_perm(customers, distance_matrix: np.ndarray, n_depots: int) -> List[int]:
    customers = list(customers)
    if not customers:
        return []
    current = random.choice(range(n_depots))
    remaining = set(customers)
    perm: List[int] = []
    D = distance_matrix
    while remaining:
        ranked = sorted(remaining, key=lambda x: D[current, x])
        rcl = ranked[:max(1, min(5, len(ranked)))]
        nxt = random.choice(rcl)
        perm.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return perm


def _initial_population(customers, coordinates, distance_matrix,
                        population_size, n_depots) -> List[List[int]]:
    customers = list(customers)
    pop: List[List[int]] = []
    if customers:
        pop.append(_sweep_perm(customers, coordinates, 0))
        pop.append(list(reversed(pop[0])))
        pop.append(_greedy_perm(customers, distance_matrix, n_depots))
    while len(pop) < population_size:
        perm = customers[:]
        random.shuffle(perm)
        if random.random() < 0.35:
            perm = _greedy_perm(customers, distance_matrix, n_depots)
        pop.append(perm)
    return pop[:population_size]


# ---------------------------------------------------------------------------
# Public driver — signature & semantics preserved
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    coordinates: np.ndarray,
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    velocity: List[float],
    fixed_cost: List[float],
    variable_cost: List[float],
    capacity: List[float],
    *,
    population_size: int = 50,
    vehicle_types: int = 1,
    n_depots: int = 1,
    route: str = "closed",
    model: str = "vrp",
    time_window: str = "without",
    fleet_size: List[int] = (),
    mutation_rate: float = 0.1,
    elite: int = 1,
    generations: int = 200,
    penalty_value: float = 10000,
    selection: str = "rw",
    seed: Optional[int] = None,
    verbose: bool = False,
    on_generation: Optional[Callable[[int, float, float], None]] = None,
) -> Tuple[pd.DataFrame, list, List[float]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    parameters = parameters.copy()
    for i in range(n_depots):
        parameters[i, 0] = 0.0

    tw_enabled = (time_window == "with")
    if model == "tsp":
        n_depots = 1
        fleet_size = [1]
        capacity = [float("inf")] * len(capacity)
    elif model == "mtsp":
        capacity = [float("inf")] * len(capacity)

    customers = list(range(n_depots, distance_matrix.shape[0]))
    if not customers:
        raw = [[], [], []]
        report = _build_report(raw, distance_matrix, parameters, velocity,
                               fixed_cost, variable_cost, route=route,
                               time_window=time_window)
        return report, raw, [0.0]

    cap_penalty = float(penalty_value)
    tw_penalty = float(penalty_value)

    ctx = _Ctx(distance_matrix, parameters, velocity, fixed_cost, variable_cost,
               capacity, n_depots, route, tw_enabled, model, list(fleet_size),
               cap_penalty, tw_penalty)
    ctx.neighbors = _build_neighbors(ctx, k=15)

    perms = _initial_population(customers, coordinates, distance_matrix,
                                population_size, n_depots)
    population: List[Candidate] = []
    init_ls_budget = 2 if tw_enabled else min(3, max(1, population_size // 10))
    for idx, p in enumerate(perms):
        population.append(_candidate_from_perm(p, ctx,
                                               apply_local_search=(idx < init_ls_budget)))
    population.sort(key=lambda x: x.sort_key)
    best = population[0]
    history = [round(best.total_distance, 2)]

    if verbose:
        print(f"Generation 0  Distance = {best.total_distance:.2f}  f(x) = {best.total_cost:.2f}")
    if on_generation:
        on_generation(0, best.total_distance, best.total_cost)

    start = tm.time()
    for gen in range(1, generations + 1):
        feasible_ratio = sum(1 for c in population if c.feasible) / max(1, len(population))
        if tw_enabled or model == "vrp":
            if feasible_ratio < 0.2:
                cap_penalty *= 1.12
                tw_penalty *= 1.12
            elif feasible_ratio > 0.8:
                cap_penalty *= 0.94
                tw_penalty *= 0.94
            cap_penalty = max(10.0, min(cap_penalty, penalty_value * 100.0))
            tw_penalty = max(10.0, min(tw_penalty, penalty_value * 100.0))
            ctx.update_penalties(cap_penalty, tw_penalty)

        population.sort(key=lambda x: x.sort_key)
        next_pop: List[Candidate] = population[:max(0, elite)]

        while len(next_pop) < population_size:
            p1 = _select_parent(population, selection)
            p2 = _select_parent(population, selection)
            if random.random() < 0.5:
                child_perm = _order_crossover(p1.perm, p2.perm)
            else:
                child_perm = _route_based_crossover(p1, p2)
            if random.random() <= mutation_rate:
                child_perm = _mutate_perm(child_perm[:])
            if (tw_enabled or model == "mtsp") and random.random() <= mutation_rate * 0.5:
                child_perm = _mutate_perm(child_perm[:])
            ls_rate = 1.0 if model == "tsp" else (
                0.35 if model == "mtsp" else (0.15 if tw_enabled else 0.22)
            )
            do_ls = (len(next_pop) <= max(1, elite) or random.random() < ls_rate)
            child = _candidate_from_perm(child_perm, ctx, apply_local_search=do_ls)
            next_pop.append(child)

        population = sorted(next_pop, key=lambda x: x.sort_key)[:population_size]

        intensify_every = 4 if tw_enabled else 3
        if gen % intensify_every == 0:
            intensify_k = min((3 if tw_enabled else 4), len(population))
            for idx in range(intensify_k):
                population[idx] = _candidate_from_perm(population[idx].perm, ctx,
                                                       apply_local_search=True)
            population.sort(key=lambda x: x.sort_key)
            population = population[:population_size]

        if population[0].sort_key < best.sort_key:
            best = population[0]

        history.append(round(best.total_distance, 2))
        if verbose:
            print(f"Generation {gen}  Distance = {best.total_distance:.2f}  f(x) = {best.total_cost:.2f}")
        if on_generation:
            on_generation(gen, best.total_distance, best.total_cost)

    report = _build_report(best.raw, distance_matrix, parameters, velocity,
                           fixed_cost, variable_cost, route=route,
                           time_window=time_window)
    if verbose:
        print(f"Algorithm time: {round(tm.time() - start, 2)} s")
    return report, _clone_raw(best.raw), history
