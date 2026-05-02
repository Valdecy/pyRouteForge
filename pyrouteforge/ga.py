
"""
Hybrid genetic search core for routing problems.

Public API compatibility is preserved through ``run_genetic_algorithm``.
Internally this module now uses case-aware giant-tour decoding plus local
search instead of the legacy route-list crossover/mutation scheme.

Design by case
--------------
* TSP   : OX crossover + inversion/swap mutation + 2-opt local search
* mTSP  : giant-tour + split decoder (+ exact route count when fleet_size given)
* VRP   : giant-tour + split decoder + relocate/swap/2-opt local search
* VRPTW : same as VRP, but route evaluation and repairs are time-window aware

The returned solution still uses the legacy raw format expected by plotting:
    [ [[depot], ...], [[client,...], ...], [[vehicle_type], ...] ]
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time as tm
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Legacy-compatible report helpers
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
    vel = velocity[0]
    nodes = [d] + list(subroute) + [d]
    L = len(nodes)
    wait = [0.0] * L
    time = [0.0] * L
    for i in range(1, L):
        prev = nodes[i - 1]
        cur = nodes[i]
        t = time[i - 1] + distance_matrix[prev, cur] / max(vel, 1e-12)
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
# Hybrid GA internals
# ---------------------------------------------------------------------------

@dataclass
class RouteState:
    depot: int
    vehicle: int
    stops: List[int]
    distance: float
    cost: float
    wait_cost: float
    cap_violation: float
    tw_violation: float
    feasible: bool


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
        return (
            0 if self.feasible else 1,
            round(self.violation, 8),
            round(self.total_cost, 8),
            round(self.total_distance, 8),
            len(self.perm),
        )


def _clone_raw(raw):
    return [[d[:] for d in raw[0]], [r[:] for r in raw[1]], [v[:] for v in raw[2]]]


def _flatten_raw(raw: List[List[List[int]]]) -> List[int]:
    perm: List[int] = []
    for route in raw[1]:
        perm.extend(route)
    return perm


def _route_eval(
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    route_mode: str,
    depot: int,
    vehicle: int,
    stops: Sequence[int],
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
) -> RouteState:
    stops = list(stops)
    if not stops:
        return RouteState(depot, vehicle, [], 0.0, 0.0, 0.0, 0.0, 0.0, True)

    demand = parameters[:, 0]
    tw_early = parameters[:, 1]
    tw_late = parameters[:, 2]
    tw_service = parameters[:, 3]
    tw_wait_cost = parameters[:, 4]

    prev = depot
    current_time = 0.0
    load = 0.0
    total_distance = 0.0
    wait_cost_total = 0.0
    cap_violation = 0.0
    tw_violation = 0.0
    vel = max(float(velocity[vehicle]), 1e-12)
    cap_limit = float(capacity[vehicle])

    for node in stops:
        leg = float(distance_matrix[prev, node])
        total_distance += leg
        arrival = current_time + leg / vel
        start_service = arrival
        if tw_enabled and arrival < tw_early[node]:
            wait = tw_early[node] - arrival
            wait_cost_total += wait * tw_wait_cost[node]
            start_service = tw_early[node]
        if tw_enabled and start_service > tw_late[node]:
            tw_violation += start_service - tw_late[node]
        current_time = start_service + tw_service[node]
        load += demand[node]
        if math.isfinite(cap_limit) and load > cap_limit:
            cap_violation += load - cap_limit
        prev = node

    if route_mode == "closed":
        leg = float(distance_matrix[prev, depot])
        total_distance += leg
        arrival = current_time + leg / vel
        start_service = arrival
        if tw_enabled and arrival < tw_early[depot]:
            wait = tw_early[depot] - arrival
            wait_cost_total += wait * tw_wait_cost[depot]
            start_service = tw_early[depot]
        if tw_enabled and start_service > tw_late[depot]:
            tw_violation += start_service - tw_late[depot]

    total_cost = (
        float(fixed_cost[vehicle])
        + total_distance * float(variable_cost[vehicle])
        + wait_cost_total
        + cap_penalty * cap_violation
        + tw_penalty * tw_violation
    )
    feasible = (cap_violation <= 1e-9 and tw_violation <= 1e-9)
    return RouteState(
        depot=depot,
        vehicle=vehicle,
        stops=stops,
        distance=total_distance,
        cost=total_cost,
        wait_cost=wait_cost_total,
        cap_violation=cap_violation,
        tw_violation=tw_violation,
        feasible=feasible,
    )


def _route_eval_all_types(
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    route_mode: str,
    depot: int,
    stops: Sequence[int],
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
) -> List[RouteState]:
    return [
        _route_eval(distance_matrix, parameters, route_mode, depot, v, stops,
                    velocity, fixed_cost, variable_cost, capacity,
                    tw_enabled, cap_penalty, tw_penalty)
        for v in range(len(capacity))
    ]


def _best_route_for_segment(
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    route_mode: str,
    depots: Sequence[int],
    vehicle_types: Sequence[int],
    stops: Sequence[int],
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
) -> RouteState:
    best = None
    for depot in depots:
        for vehicle in vehicle_types:
            route = _route_eval(distance_matrix, parameters, route_mode, depot, vehicle, stops,
                                velocity, fixed_cost, variable_cost, capacity,
                                tw_enabled, cap_penalty, tw_penalty)
            if best is None or (
                (0 if route.feasible else 1, route.cap_violation + route.tw_violation, route.cost, route.distance)
                < (0 if best.feasible else 1, best.cap_violation + best.tw_violation, best.cost, best.distance)
            ):
                best = route
    assert best is not None
    return best


def _segment_cache(
    perm: Sequence[int],
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    n_depots: int,
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    route_mode: str,
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
) -> List[List[RouteState]]:
    n = len(perm)
    depots = list(range(n_depots))
    vehicle_types = list(range(len(capacity)))
    cache: List[List[RouteState]] = [[None] * n for _ in range(n)]  # type: ignore[list-item]
    # O(n^2 * depots * vehicle_types * avg segment length). Simpler and robust.
    for i in range(n):
        stops: List[int] = []
        for j in range(i, n):
            stops.append(perm[j])
            cache[i][j] = _best_route_for_segment(
                distance_matrix, parameters, route_mode, depots, vehicle_types, stops,
                velocity, fixed_cost, variable_cost, capacity,
                tw_enabled, cap_penalty, tw_penalty,
            )
    return cache


def _decode_split(
    perm: Sequence[int],
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    n_depots: int,
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    fleet_size: Sequence[int],
    route_mode: str,
    model: str,
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
) -> List[RouteState]:
    n = len(perm)
    if n == 0:
        return []
    seg = _segment_cache(perm, distance_matrix, parameters, n_depots,
                         velocity, fixed_cost, variable_cost, capacity,
                         route_mode, tw_enabled, cap_penalty, tw_penalty)

    exact_routes = None
    max_routes = None
    if fleet_size:
        max_routes = max(1, min(sum(int(x) for x in fleet_size), n))
        if model == "mtsp":
            exact_routes = max_routes
    if model == "tsp":
        exact_routes = 1
        max_routes = 1

    if max_routes is None:
        dp = [float("inf")] * (n + 1)
        prev = [-1] * (n + 1)
        dp[0] = 0.0
        for j in range(1, n + 1):
            best_val = float("inf")
            best_i = -1
            for i in range(j):
                state = seg[i][j - 1]
                val = dp[i] + state.cost
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[j] = best_val
            prev[j] = best_i
        cuts: List[Tuple[int, int]] = []
        cur = n
        while cur > 0:
            i = prev[cur]
            cuts.append((i, cur - 1))
            cur = i
        cuts.reverse()
        return [seg[i][j] for (i, j) in cuts]

    # bounded-route DP
    dp = [[float("inf")] * (n + 1) for _ in range(max_routes + 1)]
    prev: List[List[Tuple[int, int]]] = [[(-1, -1)] * (n + 1) for _ in range(max_routes + 1)]
    dp[0][0] = 0.0
    for k in range(1, max_routes + 1):
        for j in range(1, n + 1):
            best_val = float("inf")
            best_i = -1
            for i in range(j):
                if dp[k - 1][i] == float("inf"):
                    continue
                state = seg[i][j - 1]
                val = dp[k - 1][i] + state.cost
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[k][j] = best_val
            prev[k][j] = (k - 1, best_i)

    if exact_routes is not None:
        best_k = exact_routes
    else:
        candidates = [(dp[k][n], k) for k in range(1, max_routes + 1)]
        best_k = min(candidates)[1]

    cuts: List[Tuple[int, int]] = []
    k = best_k
    cur = n
    while cur > 0 and k >= 1:
        pk, i = prev[k][cur]
        if i < 0:
            break
        cuts.append((i, cur - 1))
        cur = i
        k = pk
    cuts.reverse()
    if not cuts:
        return [seg[0][n - 1]]
    return [seg[i][j] for (i, j) in cuts]


def _repair_vehicle_counts(
    routes: List[RouteState],
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    route_mode: str,
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    fleet_size: Sequence[int],
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
) -> Tuple[List[RouteState], float]:
    if not fleet_size:
        return routes, 0.0
    counts = [0] * len(fleet_size)
    for r in routes:
        if r.vehicle < len(counts):
            counts[r.vehicle] += 1

    total_excess = 0.0
    routes = list(routes)
    while True:
        over_types = [i for i, c in enumerate(counts) if c > fleet_size[i]]
        under_types = [i for i, c in enumerate(counts) if c < fleet_size[i]]
        if not over_types:
            break
        if not under_types:
            total_excess = sum(max(0, counts[i] - fleet_size[i]) for i in range(len(counts)))
            break

        best_delta = None
        best_idx = None
        best_new = None
        for vt in over_types:
            for idx, route in enumerate(routes):
                if route.vehicle != vt:
                    continue
                for alt in under_types:
                    candidate = _route_eval(distance_matrix, parameters, route_mode,
                                            route.depot, alt, route.stops,
                                            velocity, fixed_cost, variable_cost, capacity,
                                            tw_enabled, cap_penalty, tw_penalty)
                    delta = candidate.cost - route.cost
                    key = ((0 if candidate.feasible else 1), delta, candidate.distance)
                    if best_delta is None or key < best_delta:
                        best_delta = key
                        best_idx = idx
                        best_new = candidate
        if best_idx is None or best_new is None:
            total_excess = sum(max(0, counts[i] - fleet_size[i]) for i in range(len(counts)))
            break
        old_v = routes[best_idx].vehicle
        counts[old_v] -= 1
        counts[best_new.vehicle] += 1
        routes[best_idx] = best_new
    return routes, total_excess


def _routes_to_raw(routes: Sequence[RouteState]) -> List[List[List[int]]]:
    depots = [[int(r.depot)] for r in routes if r.stops]
    stops = [[int(x) for x in r.stops] for r in routes if r.stops]
    vehicles = [[int(r.vehicle)] for r in routes if r.stops]
    return [depots, stops, vehicles]


def _candidate_from_perm(
    perm: Sequence[int],
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    velocity: Sequence[float],
    fixed_cost: Sequence[float],
    variable_cost: Sequence[float],
    capacity: Sequence[float],
    n_depots: int,
    route_mode: str,
    model: str,
    fleet_size: Sequence[int],
    tw_enabled: bool,
    cap_penalty: float,
    tw_penalty: float,
    apply_local_search: bool = True,
) -> Candidate:
    routes = _decode_split(perm, distance_matrix, parameters, n_depots,
                           velocity, fixed_cost, variable_cost, capacity,
                           fleet_size, route_mode, model, tw_enabled,
                           cap_penalty, tw_penalty)
    if apply_local_search:
        routes = _local_search(routes, distance_matrix, parameters, velocity,
                               fixed_cost, variable_cost, capacity,
                               route_mode, tw_enabled, cap_penalty, tw_penalty,
                               model)
        # For mTSP with an explicit fleet size, every salesman should correspond
        # to a non-empty route. Local search may temporarily collapse routes, so
        # we re-decode the improved customer order under the exact-route split.
        if model == "mtsp" and fleet_size:
            routes = _decode_split(_flatten_raw(_routes_to_raw(routes)), distance_matrix,
                                   parameters, n_depots, velocity, fixed_cost,
                                   variable_cost, capacity, fleet_size,
                                   route_mode, model, tw_enabled,
                                   cap_penalty, tw_penalty)
    routes, fleet_violation = _repair_vehicle_counts(
        routes, distance_matrix, parameters, route_mode,
        velocity, fixed_cost, variable_cost, capacity,
        fleet_size, tw_enabled, cap_penalty, tw_penalty,
    )
    total_distance = sum(r.distance for r in routes)
    total_cost = sum(r.cost for r in routes)
    cap_violation = sum(r.cap_violation for r in routes)
    tw_violation = sum(r.tw_violation for r in routes)
    feasible = (cap_violation <= 1e-9 and tw_violation <= 1e-9 and fleet_violation <= 1e-9)
    raw = _routes_to_raw(routes)
    return Candidate(
        perm=list(_flatten_raw(raw)),
        raw=raw,
        total_distance=total_distance,
        total_cost=total_cost + (fleet_violation * cap_penalty),
        cap_violation=cap_violation,
        tw_violation=tw_violation,
        fleet_violation=fleet_violation,
        feasible=feasible,
        n_routes=len(raw[1]),
    )


def _tournament(population: Sequence[Candidate], size: int = 3) -> Candidate:
    choices = random.sample(range(len(population)), k=min(size, len(population)))
    best = min((population[i] for i in choices), key=lambda x: x.sort_key)
    return best


def _roulette(population: Sequence[Candidate], rank_mode: bool = False) -> Candidate:
    ordered = sorted(population, key=lambda x: x.sort_key)
    if rank_mode:
        weights = np.arange(len(ordered), 0, -1, dtype=float)
    else:
        scores = np.array([c.total_cost + c.violation * 1000.0 for c in ordered], dtype=float)
        scores -= scores.min()
        weights = 1.0 / (1.0 + scores)
    probs = weights / weights.sum()
    idx = np.random.choice(len(ordered), p=probs)
    return ordered[int(idx)]


def _select_parent(population: Sequence[Candidate], selection: str) -> Candidate:
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
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = set(p1[a:b + 1])
    fill = [x for x in p2 if x not in used]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return [int(x) for x in child]


def _route_based_crossover(c1: Candidate, c2: Candidate) -> List[int]:
    if not c1.raw[1] or not c2.raw[1]:
        return _order_crossover(c1.perm, c2.perm)
    donor = random.choice(c1.raw[1])
    donor_set = set(donor)
    base = [x for x in c2.perm if x not in donor_set]
    if not base:
        return list(donor)
    insert_pos = random.randint(0, len(base))
    child = base[:insert_pos] + donor[:] + base[insert_pos:]
    return child


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


def _try_two_opt(route: RouteState, distance_matrix, parameters, velocity,
                 fixed_cost, variable_cost, capacity, route_mode,
                 tw_enabled, cap_penalty, tw_penalty) -> RouteState:
    best = route
    stops = route.stops
    n = len(stops)
    if n < 4:
        return best
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                cand_stops = stops[:i] + list(reversed(stops[i:j + 1])) + stops[j + 1:]
                cand = _route_eval(distance_matrix, parameters, route_mode, route.depot, route.vehicle,
                                   cand_stops, velocity, fixed_cost, variable_cost, capacity,
                                   tw_enabled, cap_penalty, tw_penalty)
                if ((0 if cand.feasible else 1, cand.cap_violation + cand.tw_violation, cand.cost, cand.distance)
                        < (0 if best.feasible else 1, best.cap_violation + best.tw_violation, best.cost, best.distance)):
                    best = cand
                    stops = cand.stops
                    n = len(stops)
                    improved = True
                    break
            if improved:
                break
    return best


def _local_search(routes: List[RouteState], distance_matrix, parameters, velocity,
                  fixed_cost, variable_cost, capacity, route_mode, tw_enabled,
                  cap_penalty, tw_penalty, model: str) -> List[RouteState]:
    if not routes:
        return routes
    routes = [RouteState(r.depot, r.vehicle, r.stops[:], r.distance, r.cost, r.wait_cost,
                         r.cap_violation, r.tw_violation, r.feasible)
              for r in routes]

    # Intra-route improvement
    for idx, route in enumerate(routes):
        if model == "tsp" or len(route.stops) <= 60:
            routes[idx] = _try_two_opt(route, distance_matrix, parameters, velocity,
                                       fixed_cost, variable_cost, capacity, route_mode,
                                       tw_enabled, cap_penalty, tw_penalty)

    # Inter-route relocate / swap (first improvement, bounded effort)
    max_passes = 3
    for _ in range(max_passes):
        improved = False
        current_score = sum(r.cost for r in routes), sum(r.cap_violation + r.tw_violation for r in routes)
        # relocate
        for a in range(len(routes)):
            if improved:
                break
            for b in range(len(routes)):
                if a == b and len(routes[a].stops) <= 1:
                    continue
                for i in range(len(routes[a].stops)):
                    node = routes[a].stops[i]
                    base_a = routes[a].stops[:i] + routes[a].stops[i + 1:]
                    b_positions = range(len(routes[b].stops) + 1)
                    for pos in b_positions:
                        if a == b and (pos == i or pos == i + 1):
                            continue
                        cand_a_stops = base_a if a != b else None
                        if a == b:
                            temp = routes[a].stops[:]
                            moved = temp.pop(i)
                            temp.insert(pos if pos <= len(temp) else len(temp), moved)
                            cand = _route_eval(distance_matrix, parameters, route_mode,
                                               routes[a].depot, routes[a].vehicle, temp,
                                               velocity, fixed_cost, variable_cost, capacity,
                                               tw_enabled, cap_penalty, tw_penalty)
                            if ((0 if cand.feasible else 1, cand.cap_violation + cand.tw_violation, cand.cost)
                                    < (0 if routes[a].feasible else 1, routes[a].cap_violation + routes[a].tw_violation, routes[a].cost)):
                                routes[a] = cand
                                improved = True
                                break
                        else:
                            cand_a = _route_eval(distance_matrix, parameters, route_mode,
                                                 routes[a].depot, routes[a].vehicle, base_a,
                                                 velocity, fixed_cost, variable_cost, capacity,
                                                 tw_enabled, cap_penalty, tw_penalty)
                            temp_b = routes[b].stops[:]
                            temp_b.insert(pos, node)
                            cand_b = _route_eval(distance_matrix, parameters, route_mode,
                                                 routes[b].depot, routes[b].vehicle, temp_b,
                                                 velocity, fixed_cost, variable_cost, capacity,
                                                 tw_enabled, cap_penalty, tw_penalty)
                            old_key = (0 if routes[a].feasible and routes[b].feasible else 1,
                                       routes[a].cap_violation + routes[a].tw_violation + routes[b].cap_violation + routes[b].tw_violation,
                                       routes[a].cost + routes[b].cost)
                            new_key = (0 if cand_a.feasible and cand_b.feasible else 1,
                                       cand_a.cap_violation + cand_a.tw_violation + cand_b.cap_violation + cand_b.tw_violation,
                                       cand_a.cost + cand_b.cost)
                            if new_key < old_key:
                                routes[a] = cand_a
                                routes[b] = cand_b
                                if not routes[a].stops:
                                    del routes[a]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        # swap
        for a in range(len(routes)):
            if improved:
                break
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a].stops)):
                    for j in range(len(routes[b].stops)):
                        ra = routes[a].stops[:]
                        rb = routes[b].stops[:]
                        ra[i], rb[j] = rb[j], ra[i]
                        cand_a = _route_eval(distance_matrix, parameters, route_mode,
                                             routes[a].depot, routes[a].vehicle, ra,
                                             velocity, fixed_cost, variable_cost, capacity,
                                             tw_enabled, cap_penalty, tw_penalty)
                        cand_b = _route_eval(distance_matrix, parameters, route_mode,
                                             routes[b].depot, routes[b].vehicle, rb,
                                             velocity, fixed_cost, variable_cost, capacity,
                                             tw_enabled, cap_penalty, tw_penalty)
                        old_key = (0 if routes[a].feasible and routes[b].feasible else 1,
                                   routes[a].cap_violation + routes[a].tw_violation + routes[b].cap_violation + routes[b].tw_violation,
                                   routes[a].cost + routes[b].cost)
                        new_key = (0 if cand_a.feasible and cand_b.feasible else 1,
                                   cand_a.cap_violation + cand_a.tw_violation + cand_b.cap_violation + cand_b.tw_violation,
                                   cand_a.cost + cand_b.cost)
                        if new_key < old_key:
                            routes[a] = cand_a
                            routes[b] = cand_b
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return [r for r in routes if r.stops]


def _randomized_greedy_perm(customers: Sequence[int], distance_matrix: np.ndarray, n_depots: int) -> List[int]:
    customers = list(customers)
    if not customers:
        return []
    current = random.choice(list(range(n_depots)))
    remaining = set(customers)
    perm: List[int] = []
    while remaining:
        ranked = sorted(remaining, key=lambda x: distance_matrix[current, x])
        rcl = ranked[:max(1, min(5, len(ranked)))]
        nxt = random.choice(rcl)
        perm.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return perm


def _sweep_perm(customers: Sequence[int], coordinates: np.ndarray, depot_idx: int = 0) -> List[int]:
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


def _initial_population(customers: Sequence[int], coordinates: np.ndarray, distance_matrix: np.ndarray,
                        population_size: int, n_depots: int) -> List[List[int]]:
    customers = list(customers)
    pop: List[List[int]] = []
    if customers:
        pop.append(_sweep_perm(customers, coordinates, 0))
        pop.append(list(reversed(pop[0])))
        pop.append(_randomized_greedy_perm(customers, distance_matrix, n_depots))
    while len(pop) < population_size:
        perm = customers[:]
        random.shuffle(perm)
        if random.random() < 0.35:
            perm = _randomized_greedy_perm(customers, distance_matrix, n_depots)
        pop.append(perm)
    return pop[:population_size]


# ---------------------------------------------------------------------------
# Public driver
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
                               fixed_cost, variable_cost, route=route, time_window=time_window)
        return report, raw, [0.0]

    cap_penalty = float(penalty_value)
    tw_penalty = float(penalty_value)

    perms = _initial_population(customers, coordinates, distance_matrix, population_size, n_depots)
    population = [
        _candidate_from_perm(p, distance_matrix, parameters, velocity, fixed_cost, variable_cost,
                             capacity, n_depots, route, model, list(fleet_size), tw_enabled,
                             cap_penalty, tw_penalty, apply_local_search=True)
        for p in perms
    ]
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

        population.sort(key=lambda x: x.sort_key)
        next_population: List[Candidate] = population[:max(0, elite)]

        while len(next_population) < population_size:
            p1 = _select_parent(population, selection)
            p2 = _select_parent(population, selection)
            if random.random() < 0.5:
                child_perm = _order_crossover(p1.perm, p2.perm)
            else:
                child_perm = _route_based_crossover(p1, p2)
            if random.random() <= mutation_rate:
                child_perm = _mutate_perm(child_perm[:])
            # light second mutation on harder constrained cases
            if (tw_enabled or model == "mtsp") and random.random() <= mutation_rate * 0.5:
                child_perm = _mutate_perm(child_perm[:])
            child = _candidate_from_perm(child_perm, distance_matrix, parameters,
                                         velocity, fixed_cost, variable_cost, capacity,
                                         n_depots, route, model, list(fleet_size),
                                         tw_enabled, cap_penalty, tw_penalty,
                                         apply_local_search=True)
            next_population.append(child)

        population = sorted(next_population, key=lambda x: x.sort_key)[:population_size]
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
