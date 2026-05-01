"""
Genetic algorithm core for routing problems.

Internal module — users interact with :func:`routeforge.solve`. The
algorithm is a port of Valdecy Pereira's well-tested implementation,
restructured for readability and packaged so the heavy inner loops
benefit from numba (when installed) via ``_kernels``.

An "individual" is a triple of equal-length lists::

    [
        [[depot_idx], ...],         # one per route
        [[client_idx, ...], ...],   # client visit order, one list per route
        [[vehicle_type_idx], ...],  # one per route
    ]
"""

from __future__ import annotations

import random
import time as tm
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._kernels import route_cost_basic, route_cost_time_windows


# ---------------------------------------------------------------------------
# Individual cloning (replaces deepcopy on the hot path)
# ---------------------------------------------------------------------------

def _clone_individual(ind):
    return [
        [lst[:] for lst in ind[0]],
        [lst[:] for lst in ind[1]],
        [lst[:] for lst in ind[2]],
    ]


# ---------------------------------------------------------------------------
# Route-level evaluators (used in reporting)
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


def _route_demand_sum(subroute, demand_col):
    return sum(demand_col[node] for node in subroute)


def _single_route_penalised_cost(distance_matrix, parameters, velocity, fixed_cost,
                                 variable_cost, capacity, penalty_value, time_window,
                                 route, depot, subroute, v_type):
    if not subroute:
        return 0.0
    demand = parameters[:, 0]
    sub_arr = np.asarray(subroute, dtype=np.int64)
    closed = 0 if route == "open" else 1
    if time_window == "with":
        return float(route_cost_time_windows(
            distance_matrix, demand,
            parameters[:, 1], parameters[:, 2], parameters[:, 3], parameters[:, 4],
            depot[0], sub_arr, velocity[v_type], fixed_cost[v_type],
            variable_cost[v_type], capacity[v_type], penalty_value, closed,
        ))
    return float(route_cost_basic(
        distance_matrix, demand, depot[0], sub_arr,
        fixed_cost[v_type], variable_cost[v_type], capacity[v_type],
        penalty_value, closed,
    ))


# ---------------------------------------------------------------------------
# Greedy fixers between generations
# ---------------------------------------------------------------------------

def _reassign_depots(n_depots, individual, distance_matrix):
    for j in range(len(individual[1])):
        subroute = individual[1][j]
        best_d = float("inf")
        best = individual[0][j][0]
        for i in range(n_depots):
            d = _evaluate_distance(distance_matrix, [i], subroute)[-1]
            if d < best_d:
                best_d, best = d, i
        individual[0][j] = [best]
    return individual


def _reassign_vehicles(vehicle_types, individual, distance_matrix, parameters,
                       velocity, fixed_cost, variable_cost, capacity, penalty_value,
                       time_window, route):
    for i in range(len(individual[0])):
        depot = individual[0][i]
        subroute = individual[1][i]
        current = individual[2][i][0]
        best_v = current
        best_c = _single_route_penalised_cost(
            distance_matrix, parameters, velocity, fixed_cost, variable_cost,
            capacity, penalty_value, time_window, route, depot, subroute, current,
        )
        for j in range(vehicle_types):
            if j == current:
                continue
            c = _single_route_penalised_cost(
                distance_matrix, parameters, velocity, fixed_cost, variable_cost,
                capacity, penalty_value, time_window, route, depot, subroute, j,
            )
            if c < best_c:
                best_c, best_v = c, j
        individual[2][i] = [best_v]
    return individual


def _split_overcapacity_routes(individual, parameters, capacity, max_iter=64):
    """Split routes whose cumulative load exceeds capacity into feasible+overflow halves."""
    for _ in range(max_iter):
        solution = [[], [], []]
        changed = False
        for i in range(len(individual[0])):
            cap = _evaluate_capacity(parameters, individual[0][i], individual[1][i])
            cap_core = cap[1:-1]
            cap_i = capacity[individual[2][i][0]]
            if not cap_core or max(cap_core) <= cap_i:
                solution[0].append(individual[0][i])
                solution[1].append(individual[1][i])
                solution[2].append(individual[2][i])
                continue
            sep = [x > cap_i for x in cap_core]
            sub = individual[1][i]
            sep_f = [sub[x] for x in range(len(sub)) if not sep[x]]
            sep_t = [sub[x] for x in range(len(sub)) if sep[x]]
            if sep_f and sep_t:
                changed = True
                solution[0].append(individual[0][i]); solution[0].append(individual[0][i])
                solution[1].append(sep_f);            solution[1].append(sep_t)
                solution[2].append(individual[2][i]); solution[2].append(individual[2][i])
            elif sep_t:
                solution[0].append(individual[0][i])
                solution[1].append(sep_t)
                solution[2].append(individual[2][i])
            elif sep_f:
                solution[0].append(individual[0][i])
                solution[1].append(sep_f)
                solution[2].append(individual[2][i])
        individual = solution
        if not changed:
            break
    return individual


# ---------------------------------------------------------------------------
# Population fitness + initial generation
# ---------------------------------------------------------------------------

def _target_function(population, distance_matrix, parameters, velocity, fixed_cost,
                     variable_cost, capacity, penalty_value, time_window, route,
                     fleet_size):
    demand = parameters[:, 0]
    tw_early = parameters[:, 1]; tw_late = parameters[:, 2]
    tw_st = parameters[:, 3];   tw_wc = parameters[:, 4]
    closed = 0 if route == "open" else 1
    use_tw = (time_window == "with")
    cost = [[0.0] for _ in population]

    for k, individual in enumerate(population):
        total = 0.0
        flt_cnt = [0] * len(fleet_size)
        for i in range(len(individual[1])):
            subroute = individual[1][i]
            if not subroute:
                continue
            v_type = individual[2][i][0]
            depot_idx = individual[0][i][0]
            sub_arr = np.asarray(subroute, dtype=np.int64)
            if use_tw:
                total += float(route_cost_time_windows(
                    distance_matrix, demand, tw_early, tw_late, tw_st, tw_wc,
                    depot_idx, sub_arr, velocity[v_type], fixed_cost[v_type],
                    variable_cost[v_type], capacity[v_type], penalty_value, closed,
                ))
            else:
                total += float(route_cost_basic(
                    distance_matrix, demand, depot_idx, sub_arr,
                    fixed_cost[v_type], variable_cost[v_type],
                    capacity[v_type], penalty_value, closed,
                ))
            if fleet_size:
                flt_cnt[v_type] += 1

        pnlt = 0
        if fleet_size:
            for v in range(len(fleet_size)):
                over = flt_cnt[v] - fleet_size[v]
                if over > 0:
                    pnlt += over
        cost[k][0] = total + pnlt * penalty_value

    return cost, population


def _initial_population(distance_matrix, population_size, vehicle_types, n_depots, model):
    if model == "tsp":
        n_depots = 1
    depots = [[i] for i in range(n_depots)]
    vehicles = [[i] for i in range(vehicle_types)]
    clients = list(range(n_depots, distance_matrix.shape[0]))
    population = []
    for _ in range(population_size):
        remaining = clients[:]
        routes, routes_depot, routes_vehicles = [], [], []
        while remaining:
            e = random.choice(vehicles)
            d = random.choice(depots)
            if model == "tsp":
                c = random.sample(remaining, len(remaining))
            else:
                c = random.sample(remaining, random.randint(1, len(remaining)))
            routes_vehicles.append(e[:])
            routes_depot.append(d[:])
            routes.append(c)
            rem_set = set(c)
            remaining = [x for x in remaining if x not in rem_set]
        population.append([routes_depot, routes, routes_vehicles])
    return population


def _fitness_function(cost):
    c = np.asarray([row[0] for row in cost], dtype=np.float64)
    f = 1.0 / (1.0 + c + abs(c.min()))
    cdf = np.cumsum(f) / f.sum()
    return np.column_stack([f, cdf])


def _roulette_wheel(fitness):
    return int(np.searchsorted(fitness[:, 1], random.random(), side="left"))


# ---------------------------------------------------------------------------
# Crossovers + mutation
# ---------------------------------------------------------------------------

def _crossover_tsp_brbax(p1, p2):
    offspring = _clone_individual(p2)
    L = len(p1[1][0])
    cut = sorted(random.sample(range(L), 2))
    A = p1[1][0][cut[0]:cut[1]]
    A_set = set(A)
    B = [x for x in p2[1][0] if x not in A_set]
    if random.random() > 0.5:
        A = A[::-1]
    offspring[1][0] = A + B
    return offspring


def _best_insertion(distance_matrix, depot_idx, subroute, A):
    n = len(subroute)
    if n == 0:
        return 0, float(distance_matrix[depot_idx, A] + distance_matrix[A, depot_idx])
    prev = np.empty(n + 1, dtype=np.int64); nxt = np.empty(n + 1, dtype=np.int64)
    prev[0] = depot_idx; prev[1:] = subroute
    nxt[:-1] = subroute; nxt[-1] = depot_idx
    delta = (distance_matrix[prev, A] + distance_matrix[A, nxt]
             - distance_matrix[prev, nxt])
    pos = int(np.argmin(delta))
    return pos, float(delta[pos])


def _crossover_tsp_bcr(p1, p2, distance_matrix, velocity, capacity,
                       fixed_cost, variable_cost, penalty_value, time_window,
                       parameters, route):
    offspring = _clone_individual(p2)
    L = len(p1[1][0])
    cut = random.sample(range(L), 2)
    for idx in range(2):
        A = p1[1][0][cut[idx]]
        if A in offspring[1][0]:
            offspring[1][0].remove(A)
        depot_idx = offspring[0][0][0]
        if time_window == "with":
            sub = offspring[1][0]
            v_type = offspring[2][0][0]
            best_pos, best_cost = 0, float("inf")
            for n in range(len(sub) + 1):
                trial = sub[:n] + [A] + sub[n:]
                c = _single_route_penalised_cost(
                    distance_matrix, parameters, velocity, fixed_cost,
                    variable_cost, capacity, penalty_value, time_window,
                    route, offspring[0][0], trial, v_type,
                )
                if c < best_cost:
                    best_cost, best_pos = c, n
            offspring[1][0] = sub[:best_pos] + [A] + sub[best_pos:]
        else:
            pos, _ = _best_insertion(distance_matrix, depot_idx, offspring[1][0], A)
            offspring[1][0].insert(pos, A)
    return offspring


def _crossover_vrp_brbax(p1, p2):
    s = random.randrange(len(p1[0]))
    sd, sr, sv = p1[0][s][:], p1[1][s][:], p1[2][s][:]
    transferred = set(sr)
    offspring = _clone_individual(p2)
    for k in range(len(offspring[1]) - 1, -1, -1):
        offspring[1][k] = [x for x in offspring[1][k] if x not in transferred]
        if not offspring[1][k]:
            del offspring[0][k]; del offspring[1][k]; del offspring[2][k]
    offspring[0].append(sd); offspring[1].append(sr); offspring[2].append(sv)
    return offspring


def _crossover_vrp_bcr(p1, p2, distance_matrix, velocity, capacity,
                       fixed_cost, variable_cost, penalty_value, time_window,
                       parameters, route):
    s = random.randrange(len(p1[0]))
    offspring = _clone_individual(p2)
    demand_col = parameters[:, 0]
    route_demands = [_route_demand_sum(rt, demand_col) for rt in offspring[1]]

    if len(p1[1][s]) > 1:
        cut = random.sample(range(len(p1[1][s])), 2); gene = 2
    else:
        cut = [0, 0]; gene = 1

    for idx in range(gene):
        A = p1[1][s][cut[idx]]
        demand_A = demand_col[A]
        for m in range(len(offspring[1])):
            if A in offspring[1][m]:
                offspring[1][m].remove(A)
                route_demands[m] -= demand_A
                break

        best_m, best_pos, best_delta = 0, 0, float("inf")
        if time_window == "with":
            for m in range(len(offspring[1])):
                depot = offspring[0][m]
                sub = offspring[1][m]
                v_type = offspring[2][m][0]
                for n in range(len(sub) + 1):
                    trial = sub[:n] + [A] + sub[n:]
                    c = _single_route_penalised_cost(
                        distance_matrix, parameters, velocity, fixed_cost,
                        variable_cost, capacity, penalty_value, time_window,
                        route, depot, trial, v_type,
                    )
                    if c < best_delta:
                        best_delta, best_m, best_pos = c, m, n
        else:
            for m in range(len(offspring[1])):
                depot_idx = offspring[0][m][0]
                pos, delta = _best_insertion(distance_matrix, depot_idx, offspring[1][m], A)
                v_type = offspring[2][m][0]
                score = delta + (penalty_value if route_demands[m] + demand_A > capacity[v_type] else 0.0)
                if score < best_delta:
                    best_delta, best_m, best_pos = score, m, pos

        offspring[1][best_m].insert(best_pos, A)
        route_demands[best_m] += demand_A

    for i in range(len(offspring[1]) - 1, -1, -1):
        if not offspring[1][i]:
            del offspring[0][i]; del offspring[1][i]; del offspring[2][i]
            del route_demands[i]
    return offspring


def _breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity,
              capacity, fixed_cost, variable_cost, penalty_value, time_window,
              parameters, route, vehicle_types):
    if elite > 0:
        order = sorted(range(len(population)), key=lambda i: cost[i][0])
        population = [population[i] for i in order]
        cost = [cost[i] for i in order]
        offspring = [_clone_individual(population[i]) for i in range(elite)]
        offspring.extend([None] * (len(population) - elite))
    else:
        offspring = [None] * len(population)

    pop_len = len(population)
    for i in range(elite, pop_len):
        p1 = _roulette_wheel(fitness)
        p2 = _roulette_wheel(fitness)
        while p1 == p2:
            p2 = random.randrange(pop_len)
        parent_1, parent_2 = population[p1], population[p2]
        r = random.random()

        if len(parent_1[1]) == 1 and len(parent_2[1]) == 1:
            if r > 0.5:
                child = _crossover_tsp_brbax(parent_1, parent_2)
                child = _crossover_tsp_bcr(child, parent_2, distance_matrix, velocity,
                                           capacity, fixed_cost, variable_cost,
                                           penalty_value, time_window, parameters, route)
            else:
                child = _crossover_tsp_brbax(parent_2, parent_1)
                child = _crossover_tsp_bcr(child, parent_1, distance_matrix, velocity,
                                           capacity, fixed_cost, variable_cost,
                                           penalty_value, time_window, parameters, route)
        elif len(parent_1[1]) > 1 and len(parent_2[1]) > 1:
            if r > 0.5:
                child = _crossover_vrp_brbax(parent_1, parent_2)
                child = _crossover_vrp_bcr(child, parent_2, distance_matrix, velocity,
                                           capacity, fixed_cost, variable_cost,
                                           penalty_value, time_window, parameters, route)
            else:
                child = _crossover_vrp_brbax(parent_2, parent_1)
                child = _crossover_vrp_bcr(child, parent_1, distance_matrix, velocity,
                                           capacity, fixed_cost, variable_cost,
                                           penalty_value, time_window, parameters, route)
        else:
            child = _clone_individual(parent_1 if len(parent_1[1]) > len(parent_2[1]) else parent_2)

        if n_depots > 1:
            child = _reassign_depots(n_depots, child, distance_matrix)
        if vehicle_types > 1:
            child = _reassign_vehicles(vehicle_types, child, distance_matrix, parameters,
                                       velocity, fixed_cost, variable_cost, capacity,
                                       penalty_value, time_window, route)
        child = _split_overcapacity_routes(child, parameters, capacity)
        offspring[i] = child
    return offspring


def _mutation_swap(individual):
    if len(individual[1]) == 1:
        k1 = k2 = 0
    else:
        k1, k2 = random.sample(range(len(individual[1])), 2)
    c1 = random.randrange(len(individual[1][k1]))
    c2 = random.randrange(len(individual[1][k2]))
    individual[1][k1][c1], individual[1][k2][c2] = individual[1][k2][c2], individual[1][k1][c1]
    return individual


def _mutation_insertion(individual):
    if len(individual[1]) == 1:
        k1 = k2 = 0
    else:
        k1, k2 = random.sample(range(len(individual[1])), 2)
    c1 = random.randrange(len(individual[1][k1]))
    c2 = random.randrange(len(individual[1][k2]) + 1)
    A = individual[1][k1].pop(c1)
    individual[1][k2].insert(c2, A)
    if not individual[1][k1]:
        del individual[0][k1]; del individual[1][k1]; del individual[2][k1]
    return individual


def _mutate_population(offspring, mutation_rate, elite):
    for i in range(elite, len(offspring)):
        if random.random() <= mutation_rate:
            if random.random() <= 0.5:
                offspring[i] = _mutation_insertion(offspring[i])
            else:
                offspring[i] = _mutation_swap(offspring[i])
        for k in range(len(offspring[i][1])):
            if len(offspring[i][1][k]) >= 2 and random.random() <= mutation_rate:
                cut = sorted(random.sample(range(len(offspring[i][1][k])), 2))
                segment = offspring[i][1][k][cut[0]:cut[1] + 1]
                if random.random() <= 0.5:
                    random.shuffle(segment)
                else:
                    segment.reverse()
                offspring[i][1][k][cut[0]:cut[1] + 1] = segment
    return offspring


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _elite_distance(individual, distance_matrix, route):
    end = 2 if route == "open" else 1
    td = 0.0
    for n in range(len(individual[1])):
        td += _evaluate_distance(distance_matrix, individual[0][n], individual[1][n])[-end]
    return round(td, 2)


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
# Public engine driver
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
    """Run the GA and return ``(report_df, best_individual, history)``.

    ``history`` is a list of best-distance values per generation, useful
    for plotting convergence.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    fleet_size = list(fleet_size)
    parameters = parameters.copy()  # we zero out depot demands below

    start = tm.time()
    max_capacity = list(capacity)
    if model == "tsp":
        n_depots = 1
        max_capacity = [float("inf")] * len(max_capacity)
    elif model == "mtsp":
        max_capacity = [float("inf")] * len(max_capacity)

    for i in range(n_depots):
        parameters[i, 0] = 0  # depots cannot have demand

    population = _initial_population(distance_matrix, population_size,
                                     vehicle_types, n_depots, model)
    cost, population = _target_function(population, distance_matrix, parameters,
                                        velocity, fixed_cost, variable_cost,
                                        max_capacity, penalty_value,
                                        time_window=time_window, route=route,
                                        fleet_size=fleet_size)
    order = sorted(range(len(cost)), key=lambda i: cost[i][0])
    population = [population[i] for i in order]
    cost = [cost[i] for i in order]

    fitness = (_fitness_function(cost) if selection == "rw"
               else _fitness_function([[i] for i in range(1, len(cost) + 1)]))

    elite_dist = _elite_distance(population[0], distance_matrix, route=route)
    elite_cost = cost[0][0]
    solution = _clone_individual(population[0])

    history = [elite_dist]
    if verbose:
        print(f"Generation 0  Distance = {elite_dist}  f(x) = {round(elite_cost, 2)}")
    if on_generation:
        on_generation(0, elite_dist, elite_cost)

    for gen in range(1, generations + 1):
        offspring = _breeding(cost, population, fitness, distance_matrix, n_depots,
                              elite, velocity, max_capacity, fixed_cost, variable_cost,
                              penalty_value, time_window, parameters, route, vehicle_types)
        offspring = _mutate_population(offspring, mutation_rate=mutation_rate, elite=elite)
        cost, population = _target_function(offspring, distance_matrix, parameters,
                                            velocity, fixed_cost, variable_cost,
                                            max_capacity, penalty_value,
                                            time_window=time_window, route=route,
                                            fleet_size=fleet_size)
        order = sorted(range(len(cost)), key=lambda i: cost[i][0])
        population = [population[i] for i in order]
        cost = [cost[i] for i in order]

        elite_child = _elite_distance(population[0], distance_matrix, route=route)
        fitness = (_fitness_function(cost) if selection == "rw"
                   else _fitness_function([[i] for i in range(1, len(cost) + 1)]))

        if elite_dist > elite_child:
            elite_dist = elite_child
            solution = _clone_individual(population[0])
            elite_cost = cost[0][0]

        history.append(elite_dist)
        if verbose:
            print(f"Generation {gen}  Distance = {elite_dist}  f(x) = {round(elite_cost, 2)}")
        if on_generation:
            on_generation(gen, elite_dist, elite_cost)

    report = _build_report(solution, distance_matrix, parameters, velocity,
                           fixed_cost, variable_cost, route=route,
                           time_window=time_window)
    if verbose:
        print(f"Algorithm time: {round(tm.time() - start, 2)} s")
    return report, solution, history
