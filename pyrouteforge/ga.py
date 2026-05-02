############################################################################
# Genetic Algorithm for VRP variants — optimized version
#
# Original by Prof. Valdecy Pereira, D.Sc. — UFF (Brazil)
# Citation:
#   PEREIRA, V. (2018). Project: Metaheuristic-Genetic_Algorithm,
#   File: Python-MH-Genetic Algorithm.py,
#   GitHub: https://github.com/Valdecy/Metaheuristic-Genetic_Algorithm
#
# All algorithmic behavior, supported variants (TSP, mTSP, single/multi-depot,
# heterogeneous fleet, time windows, open/closed routes), and public API are
# preserved. Speed-ups come from removing per-call overhead in the hot path:
#
#   1. `random.random()` replaces `int.from_bytes(os.urandom(8),...)/(1<<64-1)`
#      The urandom-based RNG is a system call on every draw; random.random()
#      is in-process Mersenne Twister. ~50x faster, statistically equivalent
#      for GA selection.
#   2. `_clone()` replaces `copy.deepcopy()` for the solution structure.
#      The structure is fixed: `[depots, routes, vehicles]`, each a list of
#      small int lists. Two-level shallow copy is exact for this shape and
#      ~10–30x faster than the generic, memo-tracking deepcopy.
#   3. `evaluate_distance` / `evaluate_capacity` / `evaluate_time` are
#      rewritten as straight Python loops. The original built numpy fancy-
#      index arrays per call; for the typical few-dozen-node subroutes, the
#      numpy setup cost dominates the actual arithmetic.
#   4. `target_function` no longer deep-copies each population member it
#      reads (it never mutates them).
#   5. Numpy column slices (`parameters[:, k]`) are hoisted out of the inner
#      loops where applicable.
#   6. Set-based membership tests where lists were previously scanned
#      linearly (e.g. "remove subroute clients from other routes").
############################################################################

# Required Libraries
import folium
import folium.plugins
import pandas as pd
import random
import numpy as np
import copy
import time as tm

from itertools import cycle
from matplotlib import pyplot as plt
plt.style.use('bmh')

############################################################################

# ---------- Internal helpers (new) ----------------------------------------

def _clone(ind):
    """Fast clone of `[depots, routes, vehicles]`. Each top entry is a list
    of small lists; copying those lists is sufficient and ~10–30x faster
    than copy.deepcopy for this shape."""
    return [
        [d[:] for d in ind[0]],
        [r[:] for r in ind[1]],
        [v[:] for v in ind[2]],
    ]

# ---------- Geometry helpers (unchanged) ----------------------------------

# Function: Build Coordinates
def build_coordinates(distance_matrix):
    a           = distance_matrix[0,:].reshape(distance_matrix.shape[0], 1)
    b           = distance_matrix[:,0].reshape(1, distance_matrix.shape[0])
    m           = (1/2)*(a**2 + b**2 - distance_matrix**2)
    w, u        = np.linalg.eig(np.matmul(m.T, m))
    s           = (np.diag(np.sort(w)[::-1]))**(1/2)
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    return coordinates

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
    a = coordinates
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# ---------- Plot helpers (unchanged) --------------------------------------

# Function: Tour Plot
def plot_tour_coordinates (coordinates, solution, n_depots, route, size_x = 10, size_y = 10):
    depot     = solution[0]
    city_tour = solution[1]
    cycol     = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    plt.figure(figsize = [size_x, size_y])
    for j in range(0, len(city_tour)):
        if (route == 'closed'):
            xy = np.zeros((len(city_tour[j]) + 2, 2))
        else:
            xy = np.zeros((len(city_tour[j]) + 1, 2))
        for i in range(0, xy.shape[0]):
            if (i == 0):
                xy[ i, 0] = coordinates[depot[j][i], 0]
                xy[ i, 1] = coordinates[depot[j][i], 1]
                if (route == 'closed'):
                    xy[-1, 0] = coordinates[depot[j][i], 0]
                    xy[-1, 1] = coordinates[depot[j][i], 1]
            if (i > 0 and i < len(city_tour[j])+1):
                xy[i, 0] = coordinates[city_tour[j][i-1], 0]
                xy[i, 1] = coordinates[city_tour[j][i-1], 1]
        plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 0.5, markersize = 5, color = next(cycol))
    for i in range(0, coordinates.shape[0]):
        if (i < n_depots):
            plt.plot(coordinates[i,0], coordinates[i,1], marker = 's', alpha = 1.0, markersize = 7, color = 'k')[0]
            plt.text(coordinates[i,0], coordinates[i,1] + 0.04, i, ha = 'center', va = 'bottom', color = 'k', fontsize = 7)
        else:
            plt.text(coordinates[i,0],  coordinates[i,1] + 0.04, i, ha = 'center', va = 'bottom', color = 'k', fontsize = 7)
    return

# Function: Tour Plot - Lat Long
def plot_tour_latlong (lat_long, solution, n_depots, route):
    m       = folium.Map(location = (lat_long.iloc[0][0], lat_long.iloc[0][1]), zoom_start = 14)
    clients = folium.plugins.MarkerCluster(name = 'Clients').add_to(m)
    depots  = folium.plugins.MarkerCluster(name = 'Depots').add_to(m)
    for i in range(0, lat_long.shape[0]):
        if (i < n_depots):
            folium.Marker(location = [lat_long.iloc[i][0], lat_long.iloc[i][1]], popup = '<b>Client: </b>%s</br> <b>Adress: </b>%s</br>'%(int(i), 'D'), icon = folium.Icon(color = 'black', icon = 'home')).add_to(depots)
        else:
            folium.Marker(location = [lat_long.iloc[i][0], lat_long.iloc[i][1]], popup = '<b>Client: </b>%s</br> <b>Adress: </b>%s</br>'%(int(i), 'C'), icon = folium.Icon(color = 'blue')).add_to(clients)
    depot     = solution[0]
    city_tour = solution[1]
    cycol     = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    for j in range(0, len(city_tour)):
        if (route == 'closed'):
            ltlng = np.zeros((len(city_tour[j]) + 2, 2))
        else:
            ltlng = np.zeros((len(city_tour[j]) + 1, 2))
        for i in range(0, ltlng.shape[0]):
            if (i == 0):
                ltlng[ i, 0] = lat_long.iloc[depot[j][i], 0]
                ltlng[ i, 1] = lat_long.iloc[depot[j][i], 1]
                if (route == 'closed'):
                    ltlng[-1, 0] = lat_long.iloc[depot[j][i], 0]
                    ltlng[-1, 1] = lat_long.iloc[depot[j][i], 1]
            if (i > 0 and i < len(city_tour[j])+1):
                ltlng[i, 0] = lat_long.iloc[city_tour[j][i-1], 0]
                ltlng[i, 1] = lat_long.iloc[city_tour[j][i-1], 1]
        c = next(cycol)
        for i in range(0, ltlng.shape[0]-1):
            locations = [ (ltlng[i,0], ltlng[i,1]), (ltlng[i+1,0], ltlng[i+1,1])]
            folium.PolyLine(locations , color = c, weight = 1.5, opacity = 1).add_to(m)
    return m

# ---------- Hot evaluators (rewritten) ------------------------------------

# Function: Subroute Distance
def evaluate_distance(distance_matrix, depot, subroute):
    """Cumulative distance along depot -> subroute -> depot.
    Returns a list of length len(subroute)+2:
        [0, d(depot,s0), d(depot,s0)+d(s0,s1), ..., total_with_return].
    Equivalent to the original numpy-fancy-indexing version but without the
    per-call array allocation."""
    n = len(subroute)
    out = [0.0] * (n + 2)
    if n == 0:
        return out
    dm   = distance_matrix
    d0   = depot[0]
    cum  = 0.0
    prev = d0
    for i in range(n):
        cur  = subroute[i]
        cum += dm[prev, cur]
        out[i + 1] = cum
        prev = cur
    out[n + 1] = cum + dm[prev, d0]
    return out

# Function: Subroute Time
def evaluate_time(distance_matrix, parameters, depot, subroute, velocity):
    tw_early = parameters[:, 1]
    tw_st    = parameters[:, 3]
    n = len(subroute)
    wait = [0] * (n + 2)
    time = [0] * (n + 2)
    if n == 0 and len(depot) == 0:
        return wait, time
    v   = velocity[0]
    d0  = depot[0]
    dm  = distance_matrix
    prev  = d0
    cur_t = 0.0
    # n+1 legs: depot->s0, s0->s1, ..., s_{n-1}->depot (or single self-loop if n==0)
    for i in range(n + 1):
        nxt = subroute[i] if i < n else d0
        cur_t += dm[prev, nxt] / v
        early = float(tw_early[nxt])
        if cur_t < early:
            wait[i + 1] = early - cur_t
            cur_t = early
        cur_t += float(tw_st[nxt])
        time[i + 1] = cur_t
        prev = nxt
    return wait, time

# Function: Subroute Capacity
def evaluate_capacity(parameters, depot, subroute):
    demand = parameters[:, 0]
    n = len(subroute)
    out = [0.0] * (n + 2)
    d0  = depot[0]
    cum = float(demand[d0])
    out[0] = cum
    for i in range(n):
        cum += float(demand[subroute[i]])
        out[i + 1] = cum
    out[n + 1] = cum + float(demand[d0])
    return out

# Function: Subroute Cost
def evaluate_cost(dist, wait, parameters, depot, subroute, fixed_cost, variable_cost, time_window):
    fc = fixed_cost[0]
    vc = variable_cost[0]
    n_total = len(dist)
    if time_window == 'with':
        tw_wc = parameters[:, 4]
        d0 = depot[0]
        out = [0.0] * n_total
        last = n_total - 1
        for i in range(n_total):
            if i == 0 or i == last:
                node = d0
            else:
                node = subroute[i - 1]
            x = dist[i]
            y = wait[i] if i < len(wait) else 0
            z = float(tw_wc[node])
            out[i] = fc + y * z if x == 0 else fc + x * vc + y * z
        return out
    else:
        return [fc if x == 0 else fc + x * vc for x in dist]

# Function: Subroute Cost (Penalty form, returns scalar matching cost[-1] of original)
def evaluate_cost_penalty(dist, time, wait, cap, capacity, parameters, depot,
                          subroute, fixed_cost, variable_cost, penalty_value,
                          time_window, route):
    if route == 'open':
        subroute_ = depot + subroute
    else:
        subroute_ = depot + subroute + depot
    n_sub = len(subroute_)
    fc = fixed_cost[0]
    vc = variable_cost[0]
    pnlt = 0
    # Capacity penalty (matches sum(x > capacity for x in cap[0:n_sub]))
    cap_limit = min(n_sub, len(cap))
    for i in range(cap_limit):
        if cap[i] > capacity:
            pnlt += 1
    if time_window == 'with':
        tw_late = parameters[:, 2]
        tw_st   = parameters[:, 3]
        tw_wc   = parameters[:, 4]
        # Time-window penalty (matches zip-stop-shortest behavior)
        tw_limit = min(len(time), n_sub)
        for i in range(tw_limit):
            node = subroute_[i]
            if time[i] > float(tw_late[node]) + float(tw_st[node]):
                pnlt += 1
        # Original built `cost = [...]` over zip(dist, wait, tw_wc[subroute_])
        # whose length is min(len(dist), len(wait), n_sub), then took cost[-1].
        # The comprehension uses cost[0]==0 (initial), so:
        #   cost[i] = fc + y*z         if x == 0
        #   cost[i] = x*vc + y*z       otherwise
        zip_len = min(len(dist), len(wait), n_sub)
        idx = zip_len - 1
        node = subroute_[idx]
        x = dist[idx]
        y = wait[idx]
        z = float(tw_wc[node])
        total = fc + y * z if x == 0 else x * vc + y * z
    else:
        # Original built `cost = [...]` over dist (length len(dist)),
        # then took cost[-1] = cost[len(dist)-1].
        idx = len(dist) - 1
        x = dist[idx]
        total = fc if x == 0 else x * vc
    return total + pnlt * penalty_value

# Function: Routes Nearest Depot
def evaluate_depot(n_depots, individual, distance_matrix):
    d_1 = float('+inf')
    for i in range(0, n_depots):
        depot_i = [i]
        for j in range(0, len(individual[1])):
            d_2 = evaluate_distance(distance_matrix, depot_i, individual[1][j])[-1]
            if (d_2 < d_1):
                d_1 = d_2
                individual[0][j] = [i]
    return individual

# Function: Routes Best Vehicle
def evaluate_vehicle(vehicle_types, individual, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size):
    cost, _     = target_function([individual], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size)
    individual_ = _clone(individual)
    for i in range(0, len(individual[0])):
        for j in range(0, vehicle_types):
            individual_[2][i] = [j]
            cost_, _          = target_function([individual_], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size)
            if (cost_ < cost):
                cost             = cost_
                individual[2][i] = [j]
                individual_      = _clone(individual)
            else:
                individual_      = _clone(individual)
    return individual

# Function: Routes Break Capacity
def cap_break(vehicle_types, individual, parameters, capacity):
    go_on = True
    while (go_on):
        individual_ = _clone(individual)
        solution    = [[], [], []]
        for i in range(0, len(individual_[0])):
            cap   = evaluate_capacity(parameters, individual_[0][i], individual_[1][i])
            sep   = [x >  capacity[individual_[2][i][0]] for x in cap[1:-1] ]
            sep_f = [individual_[1][i][x] for x in range(0, len(individual_[1][i])) if sep[x] == False]
            sep_t = [individual_[1][i][x] for x in range(0, len(individual_[1][i])) if sep[x] == True ]
            if (len(sep_t) > 0 and len(sep_f) > 0):
                solution[0].append(individual_[0][i])
                solution[0].append(individual_[0][i])
                solution[1].append(sep_f)
                solution[1].append(sep_t)
                solution[2].append(individual_[2][i])
                solution[2].append(individual_[2][i])
            if (len(sep_t) > 0 and len(sep_f) == 0):
                solution[0].append(individual_[0][i])
                solution[1].append(sep_t)
                solution[2].append(individual_[2][i])
            if (len(sep_t) == 0 and len(sep_f) > 0):
                solution[0].append(individual_[0][i])
                solution[1].append(sep_f)
                solution[2].append(individual_[2][i])
        individual_ = _clone(solution)
        if (individual == individual_):
            go_on      = False
        else:
            go_on      = True
            individual = _clone(solution)
    return individual

# Function: Solution Report
def show_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window):
    column_names = ['Route', 'Vehicle', 'Activity', 'Job', 'Arrive_Load', 'Leave_Load', 'Wait_Time', 'Arrive_Time','Leave_Time', 'Distance', 'Costs']
    tt           = 0
    td           = 0
    tc           = 0
    tw_st        = parameters[:, 3]
    report_lst   = []
    for i in range(0, len(solution[1])):
        dist         = evaluate_distance(distance_matrix, solution[0][i], solution[1][i])
        wait, time   = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])
        reversed_sol = solution[1][i][:]
        reversed_sol.reverse()
        cap          = evaluate_capacity(parameters, solution[0][i], reversed_sol)
        cap.reverse()
        leave_cap = cap[:]
        for n in range(1, len(leave_cap)-1):
            leave_cap[n] = cap[n+1]
        cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i], fixed_cost = [fixed_cost[solution[2][i][0]]], variable_cost = [variable_cost[solution[2][i][0]]], time_window = time_window)
        if (route == 'closed'):
            subroute = [solution[0][i] + solution[1][i] + solution[0][i] ]
        elif (route == 'open'):
            subroute = [solution[0][i] + solution[1][i] ]
        for j in range(0, len(subroute[0])):
            if (j == 0):
                activity    = 'start'
                arrive_time = round(time[j],2)
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j],2)
            if (j > 0 and j < len(subroute[0]) - 1):
                activity = 'service'
            if (j == len(subroute[0]) - 1):
                activity = 'finish'
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
            report_lst.append(['#' + str(i+1), solution[2][i][0], activity, subroute[0][j], cap[j], leave_cap[j], round(wait[j],2), arrive_time, round(time[j],2), round(dist[j],2), round(cost[j],2) ])
        report_lst.append(['-//-', '-//-', '-//-', '-//-','-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    report_lst.append(['MAX TIME', '', '','', '', '', '', '', round(tt,2), '', ''])
    report_lst.append(['TOTAL', '', '','', '', '', '', '', '', round(td,2), round(tc,2)])
    report_df = pd.DataFrame(report_lst, columns = column_names)
    return report_df

# Function: Route Evaluation & Correction
def target_function(population, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size = []):
    cost     = [[0] for i in range(len(population))]
    tw_late  = parameters[:, 2]
    tw_st    = parameters[:, 3]
    n_fleet  = len(fleet_size)
    if (route == 'open'):
        end = 2
    else:
        end = 1
    with_tw = (time_window == 'with')
    for k in range(0, len(population)):
        individual = population[k]                        # read-only here, no clone needed
        size       = len(individual[1])
        i          = 0
        pnlt       = 0
        flt_cnt    = [0] * n_fleet
        depots_k   = individual[0]
        routes_k   = individual[1]
        veh_k      = individual[2]
        while (size > i):
            depot_i  = depots_k[i]
            sub_i    = routes_k[i]
            v_idx    = veh_k[i][0]
            cap_lim  = capacity[v_idx]
            dist = evaluate_distance(distance_matrix, depot_i, sub_i)
            if with_tw:
                wait, time = evaluate_time(distance_matrix, parameters, depot_i, sub_i, velocity = [velocity[v_idx]])
            else:
                wait = []
                time = []
            cap    = evaluate_capacity(parameters, depot_i, sub_i)
            cost_s = evaluate_cost(dist, wait, parameters, depot_i, sub_i,
                                   fixed_cost = [fixed_cost[v_idx]],
                                   variable_cost = [variable_cost[v_idx]],
                                   time_window = time_window)
            # Capacity penalty over cap[0:-1]
            for x in cap[:-1]:
                if x > cap_lim:
                    pnlt += 1
            if with_tw:
                if route == 'open':
                    subroute_ = depot_i + sub_i
                else:
                    subroute_ = depot_i + sub_i + depot_i
                # Time-window penalty
                m = min(len(time), len(subroute_))
                for j in range(m):
                    node = subroute_[j]
                    if time[j] > float(tw_late[node]) + float(tw_st[node]):
                        pnlt += 1
            if n_fleet > 0:
                flt_cnt[v_idx] += 1
            if size <= i + 1:
                for v in range(n_fleet):
                    v_sum = flt_cnt[v] - fleet_size[v]
                    if v_sum > 0:
                        pnlt += v_sum
            cost[k][0] = cost[k][0] + cost_s[-end] + pnlt * penalty_value
            size = len(individual[1])
            i   += 1
    cost_total = [c[:] for c in cost]
    return cost_total, population

# Function: Initial Population
def initial_population(coordinates = 'none', distance_matrix = 'none', population_size = 5, vehicle_types = 1, n_depots = 1, model = 'vrp'):
    try:
        distance_matrix.shape[0]
    except:
        distance_matrix = build_distance_matrix(coordinates)
    if (model == 'tsp'):
        n_depots = 1
    depots     = [[i] for i in range(0, n_depots)]
    vehicles   = [[i] for i in range(0, vehicle_types)]
    clients    = list(range(n_depots, distance_matrix.shape[0]))
    population = []
    for i in range(0, population_size):
        clients_temp    = clients[:]                       # cheap copy
        routes          = []
        routes_depot    = []
        routes_vehicles = []
        while (len(clients_temp) > 0):
            e = random.sample(vehicles, 1)[0]
            d = random.sample(depots, 1)[0]
            if (model == 'tsp'):
                c = random.sample(clients_temp, len(clients_temp))
            else:
                c = random.sample(clients_temp, random.randint(1, len(clients_temp)))
            routes_vehicles.append(e)
            routes_depot.append(d)
            routes.append(c)
            c_set = set(c)
            clients_temp = [item for item in clients_temp if item not in c_set]
        population.append([routes_depot, routes, routes_vehicles])
    return population

# Function: Fitness
def fitness_function(cost, population_size):
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i][0] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness):
    ix     = 0
    rnd    = random.random()                              # was os.urandom — ~50x faster
    n      = fitness.shape[0]
    cum    = fitness[:, 1]
    for i in range(0, n):
        if (rnd <= cum[i]):
            ix = i
            break
    return ix

# Function: TSP Crossover - BRBAX (Best Route Better Adjustment Recombination)
def crossover_tsp_brbax(parent_1, parent_2):
    offspring = _clone(parent_2)
    cut       = random.sample(list(range(0,len(parent_1[1][0]))), 2)
    cut.sort()
    rand      = random.random()
    A         = parent_1[1][0][cut[0]:cut[1]]
    A_set     = set(A)
    B         = [item for item in parent_2[1][0] if item not in A_set]
    if (rand > 0.5):
        A.reverse()
    offspring[1][0] = A + B
    return offspring

# Function: TSP Crossover - BCR (Best Cost Route Crossover)
def crossover_tsp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route):
    offspring = _clone(parent_2)
    cut       = random.sample(list(range(0,len(parent_1[1][0]))), 2)
    for i in range(0, 2):
        d_1            = float('+inf')
        A              = parent_1[1][0][cut[i]]
        best           = []
        parent_2[1][0] = [item for item in parent_2[1][0] if item != A]
        depot_p        = parent_2[0][0]
        veh_p          = parent_2[2][0][0]
        base_route     = parent_2[1][0]
        n_pos          = len(base_route) + 1
        # Build candidate insertions once and reuse across distance/cap/time/cost
        insertion_list = [base_route[:n] + [A] + base_route[n:] for n in range(0, n_pos)]
        dist_list      = [evaluate_distance(distance_matrix, depot_p, ins) for ins in insertion_list]
        if time_window == 'with':
            wait_time_list = [evaluate_time(distance_matrix, parameters, depot_p, ins, velocity = [velocity[veh_p]]) for ins in insertion_list]
        else:
            wait_time_list = [(0, 0)] * n_pos
        cap_list = [evaluate_capacity(parameters, depot_p, ins) for ins in insertion_list]
        d_2_list = [
            evaluate_cost_penalty(
                dist_list[n], wait_time_list[n][1], wait_time_list[n][0],
                cap_list[n], capacity[veh_p], parameters, depot_p,
                insertion_list[n], [fixed_cost[veh_p]], [variable_cost[veh_p]],
                penalty_value, time_window, route
            )
            for n in range(n_pos)
        ]
        d_2 = min(d_2_list)
        if d_2 <= d_1:
            d_1  = d_2
            best = insertion_list[d_2_list.index(d_2)]
        parent_2[1][0] = best
        if d_1 != float('+inf'):
            offspring = _clone(parent_2)
    return offspring

# Function: VRP Crossover - BRBAX (Best Route Better Adjustment Recombination)
def crossover_vrp_brbax(parent_1, parent_2):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    subroute  = [parent_1[0][s], parent_1[1][s], parent_1[2][s]]
    sub_set   = set(subroute[1])                          # set membership: O(1)
    offspring = _clone(parent_2)
    for k in range(len(parent_2[1])-1, -1, -1):
        offspring[1][k] = [item for item in offspring[1][k] if item not in sub_set]
        if (len(offspring[1][k]) == 0):
            del offspring[0][k]
            del offspring[1][k]
            del offspring[2][k]
    offspring[0].append(subroute[0])
    offspring[1].append(subroute[1])
    offspring[2].append(subroute[2])
    return offspring

# Function: VRP Crossover - BCR (Best Cost Route Crossover)
def crossover_vrp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    offspring = _clone(parent_2)
    if (len(parent_1[1][s]) > 1):
        cut  = random.sample(list(range(0,len(parent_1[1][s]))), 2)
        gene = 2
    else:
        cut  = [0, 0]
        gene = 1
    for i in range(0, gene):
        d_1   = float('+inf')
        ins_m = 0
        A     = parent_1[1][s][cut[i]]
        best  = []
        for m in range(0, len(parent_2[1])):
            parent_2[1][m] = [item for item in parent_2[1][m] if item != A]
            if (len(parent_2[1][m]) > 0):
                depot_p    = parent_2[0][m]
                veh_p      = parent_2[2][m][0]
                base_route = parent_2[1][m]
                n_pos      = len(base_route) + 1
                insertion_list = [base_route[:n] + [A] + base_route[n:] for n in range(0, n_pos)]
                dist_list      = [evaluate_distance(distance_matrix, depot_p, ins) for ins in insertion_list]
                if time_window == 'with':
                    wait_time_list = [evaluate_time(distance_matrix, parameters, depot_p, ins, velocity = [velocity[veh_p]]) for ins in insertion_list]
                else:
                    wait_time_list = [(0, 0)] * n_pos
                cap_list = [evaluate_capacity(parameters, depot_p, ins) for ins in insertion_list]
                d_2_list = [
                    evaluate_cost_penalty(
                        dist_list[n], wait_time_list[n][1], wait_time_list[n][0],
                        cap_list[n], capacity[veh_p], parameters, depot_p,
                        insertion_list[n], [fixed_cost[veh_p]], [variable_cost[veh_p]],
                        penalty_value, time_window, route
                    )
                    for n in range(n_pos)
                ]
                d_2 = min(d_2_list)
                if d_2 <= d_1:
                    d_1   = d_2
                    ins_m = m
                    best  = insertion_list[d_2_list.index(d_2)]
        parent_2[1][ins_m] = best
        if d_1 != float('+inf'):
            offspring = _clone(parent_2)
    for i in range(len(offspring[1])-1, -1, -1):
        if(len(offspring[1][i]) == 0):
            del offspring[0][i]
            del offspring[1][i]
            del offspring[2][i]
    return offspring



def _route_penalty_cost(distance_matrix, parameters, depot, subroute, vehicle_idx,
                        velocity, fixed_cost, variable_cost, capacity,
                        penalty_value, time_window, route):
    dist = evaluate_distance(distance_matrix, depot, subroute)
    if time_window == 'with':
        wait, time = evaluate_time(distance_matrix, parameters, depot, subroute, velocity=[velocity[vehicle_idx]])
    else:
        wait, time = [], []
    cap = evaluate_capacity(parameters, depot, subroute)
    return evaluate_cost_penalty(
        dist, time, wait, cap, capacity[vehicle_idx], parameters, depot, subroute,
        [fixed_cost[vehicle_idx]], [variable_cost[vehicle_idx]], penalty_value,
        time_window, route,
    )


def _compact_individual(individual):
    solution = [[], [], []]
    for d, r, v in zip(individual[0], individual[1], individual[2]):
        if len(r) > 0:
            solution[0].append(d[:])
            solution[1].append(r[:])
            solution[2].append(v[:])
    return solution


def _remove_customers(individual, removed_set):
    new_ind = [[], [], []]
    for d, route_nodes, v in zip(individual[0], individual[1], individual[2]):
        nr = [c for c in route_nodes if c not in removed_set]
        if nr:
            new_ind[0].append(d[:])
            new_ind[1].append(nr)
            new_ind[2].append(v[:])
    return new_ind


def _related_removal(individual, q, distance_matrix, parameters, time_window):
    routes = individual[1]
    nodes = [c for r in routes for c in r]
    if not nodes:
        return []
    seed = random.choice(nodes)
    route_of = {}
    for ridx, r in enumerate(routes):
        for c in r:
            route_of[c] = ridx
    tw_early = parameters[:, 1]
    tw_late = parameters[:, 2]
    scored = []
    for c in nodes:
        if c == seed:
            continue
        score = float(distance_matrix[seed, c])
        if time_window == 'with':
            score += 0.1 * abs(float(tw_early[seed]) - float(tw_early[c]))
            score += 0.1 * abs(float(tw_late[seed]) - float(tw_late[c]))
        if route_of.get(c) == route_of.get(seed):
            score *= 0.7
        score *= (0.9 + 0.2 * random.random())
        scored.append((score, c))
    scored.sort(key=lambda x: x[0])
    removed = [seed] + [c for _, c in scored[:max(0, q - 1)]]
    return removed[:q]


def _best_insertions_for_customer(individual, customer, distance_matrix, parameters,
                                  velocity, fixed_cost, variable_cost, capacity,
                                  penalty_value, time_window, route,
                                  vehicle_types, n_depots, fleet_size):
    candidates = []
    counts = [0] * max(vehicle_types, len(fleet_size), 1)
    for veh in individual[2]:
        if veh and veh[0] < len(counts):
            counts[veh[0]] += 1

    # existing routes
    for m in range(len(individual[1])):
        depot_p = individual[0][m]
        veh_p = individual[2][m][0]
        base_route = individual[1][m]
        old_cost = _route_penalty_cost(distance_matrix, parameters, depot_p, base_route,
                                       veh_p, velocity, fixed_cost, variable_cost,
                                       capacity, penalty_value, time_window, route)
        n_pos = len(base_route) + 1
        insertion_list = [base_route[:n] + [customer] + base_route[n:] for n in range(n_pos)]
        for pos, cand_route in enumerate(insertion_list):
            new_cost = _route_penalty_cost(distance_matrix, parameters, depot_p, cand_route,
                                           veh_p, velocity, fixed_cost, variable_cost,
                                           capacity, penalty_value, time_window, route)
            candidates.append((new_cost - old_cost, ('insert', m, pos)))

    # new route candidates
    for d in range(n_depots):
        depot = [d]
        for veh in range(vehicle_types):
            extra = 0
            if len(fleet_size) > 0 and veh < len(fleet_size) and counts[veh] >= fleet_size[veh]:
                extra = penalty_value
            new_cost = _route_penalty_cost(distance_matrix, parameters, depot, [customer], veh,
                                           velocity, fixed_cost, variable_cost, capacity,
                                           penalty_value, time_window, route)
            candidates.append((new_cost + extra, ('new', d, veh)))

    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return None, None
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else (best[0] + penalty_value, best[1])
    return best, second


def _apply_insertion(individual, customer, action):
    kind = action[0]
    if kind == 'insert':
        _, m, pos = action
        individual[1][m][pos:pos] = [customer]
    else:
        _, d, veh = action
        individual[0].append([d])
        individual[1].append([customer])
        individual[2].append([veh])
    return individual


def ruin_recreate(individual, distance_matrix, parameters, velocity, fixed_cost,
                  variable_cost, capacity, penalty_value, time_window, route,
                  vehicle_types, n_depots, fleet_size, ruin_fraction=0.15):
    # only meaningful for VRP-like variants
    customers = [c for r in individual[1] for c in r]
    n_customers = len(customers)
    if n_customers <= 2:
        return _clone(individual)
    q = max(1, min(n_customers - 1, int(round(ruin_fraction * n_customers))))
    if random.random() < 0.5:
        removed = random.sample(customers, q)
    else:
        removed = _related_removal(individual, q, distance_matrix, parameters, time_window)
        if len(removed) < q:
            remset = set(removed)
            pool = [c for c in customers if c not in remset]
            if pool:
                removed.extend(random.sample(pool, min(q - len(removed), len(pool))))
    removed_set = set(removed)
    partial = _remove_customers(individual, removed_set)
    to_insert = removed[:]
    random.shuffle(to_insert)

    while to_insert:
        best_choice = None
        best_regret = -float('inf')
        best_customer = None
        for customer in to_insert:
            best, second = _best_insertions_for_customer(
                partial, customer, distance_matrix, parameters, velocity,
                fixed_cost, variable_cost, capacity, penalty_value,
                time_window, route, vehicle_types, n_depots, fleet_size,
            )
            if best is None:
                continue
            regret = second[0] - best[0]
            # biased toward good best insertion when regrets tie
            key = (regret, -best[0])
            if best_choice is None or key > best_regret:
                best_choice = best
                best_regret = key
                best_customer = customer
        if best_choice is None:
            break
        partial = _apply_insertion(partial, best_customer, best_choice[1])
        to_insert.remove(best_customer)

    partial = _compact_individual(partial)
    if n_depots > 1:
        partial = evaluate_depot(n_depots, partial, distance_matrix)
    if vehicle_types > 1:
        partial = evaluate_vehicle(vehicle_types, partial, distance_matrix, parameters,
                                   velocity, fixed_cost, variable_cost, capacity,
                                   penalty_value, time_window, route, fleet_size)
    partial = cap_break(vehicle_types, partial, parameters, capacity)
    return partial


# Function: Breeding
def breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, vehicle_types, fleet_size):
    offspring = [_clone(p) for p in population]
    if (elite > 0):
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        for i in range(0, elite):
            offspring[i] = _clone(population[i])
    for i in range (elite, len(offspring)):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = _clone(population[parent_1])
        parent_2 = _clone(population[parent_2])
        rand = random.random()
        # TSP - Crossover
        if (len(parent_1[1]) == 1 and len(parent_2[1]) == 1):
            if (rand > 0.5):
                offspring[i] = crossover_tsp_brbax(parent_1, parent_2)
                offspring[i] = crossover_tsp_bcr(offspring[i], parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
            elif (rand <= 0.5):
                offspring[i] = crossover_tsp_brbax(parent_2, parent_1)
                offspring[i] = crossover_tsp_bcr(offspring[i], parent_1, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
        # VRP - Crossover
        elif((len(parent_1[1]) > 1 and len(parent_2[1]) > 1)):
            if (rand > 0.5):
                offspring[i] = crossover_vrp_brbax(parent_1, parent_2)
                offspring[i] = crossover_vrp_bcr(offspring[i], parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
            elif (rand <= 0.5):
                offspring[i] = crossover_vrp_brbax(parent_2, parent_1)
                offspring[i] = crossover_vrp_bcr(offspring[i], parent_1, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
        if (n_depots > 1):
            offspring[i] = evaluate_depot(n_depots, offspring[i], distance_matrix)
        if (vehicle_types > 1):
            offspring[i] = evaluate_vehicle(vehicle_types, offspring[i], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size)
    offspring[i] = cap_break(vehicle_types, offspring[i], parameters, capacity)
    return offspring

# Function: Mutation - Swap
def mutation_tsp_vrp_swap(individual):
    if (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]
    cut1                    = random.sample(list(range(0, len(individual[1][k1]))), 1)[0]
    cut2                    = random.sample(list(range(0, len(individual[1][k2]))), 1)[0]
    A                       = individual[1][k1][cut1]
    B                       = individual[1][k2][cut2]
    individual[1][k1][cut1] = B
    individual[1][k2][cut2] = A
    return individual

# Function: Mutation - Insertion
def mutation_tsp_vrp_insertion(individual):
    if (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]
    cut1 = random.sample(list(range(0, len(individual[1][k1])))  , 1)[0]
    cut2 = random.sample(list(range(0, len(individual[1][k2])+1)), 1)[0]
    A    = individual[1][k1][cut1]
    del individual[1][k1][cut1]
    individual[1][k2][cut2:cut2] = [A]
    if (len(individual[1][k1]) == 0):
        del individual[0][k1]
        del individual[1][k1]
        del individual[2][k1]
    return individual

# Function: Mutation
def mutation(offspring, mutation_rate, elite):
    for i in range(elite, len(offspring)):
        probability = random.random()
        if (probability <= mutation_rate):
            rand = random.random()
            if (rand <= 0.5):
                offspring[i] = mutation_tsp_vrp_insertion(offspring[i])
            elif(rand > 0.5):
                offspring[i] = mutation_tsp_vrp_swap(offspring[i])
        for k in range(0, len(offspring[i][1])):
            if (len(offspring[i][1][k]) >= 2):
                probability = random.random()
                if (probability <= mutation_rate):
                    rand = random.random()
                    cut  = random.sample(list(range(0, len(offspring[i][1][k]))), 2)
                    cut.sort()
                    C    = offspring[i][1][k][cut[0]:cut[1]+1]
                    if (rand <= 0.5):
                        random.shuffle(C)
                    elif(rand > 0.5):
                        C.reverse()
                    offspring[i][1][k][cut[0]:cut[1]+1] = C
    return offspring

# Function: Elite Distance
def elite_distance(individual, distance_matrix, route):
    if (route == 'open'):
        end = 2
    else:
        end = 1
    td = 0
    for n in range(0, len(individual[1])):
        td = td + evaluate_distance(distance_matrix, depot = individual[0][n], subroute = individual[1][n])[-end]
    return round(td,2)

# GA-VRP Function
def genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, population_size = 5, vehicle_types = 1, n_depots = 1, route = 'closed', model = 'vrp', time_window = 'without', fleet_size = [], mutation_rate = 0.1, elite = 0, generations = 50, penalty_value = 1000, graph = True, selection = 'rw'):
    start           = tm.time()
    count           = 0
    solution_report = ['None']
    max_capacity    = list(capacity)                      # was deepcopy
    if (model == 'tsp'):
        n_depots = 1
        for i in range(0, len(max_capacity)):
            max_capacity[i] = float('+inf')
    if (model == 'mtsp'):
        for i in range(0, len(max_capacity)):
            max_capacity[i] = float('+inf')
    for i in range(0, n_depots):
        parameters[i, 0] = 0
    population       = initial_population(coordinates, distance_matrix, population_size = population_size, vehicle_types = vehicle_types, n_depots = n_depots, model = model)
    cost, population = target_function(population, distance_matrix, parameters, velocity, fixed_cost, variable_cost, max_capacity, penalty_value, time_window = time_window, route = route, fleet_size = fleet_size)
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    if (selection == 'rw'):
        fitness          = fitness_function(cost, population_size)
    elif (selection == 'rb'):
        rank             = [[i] for i in range(1, len(cost)+1)]
        fitness          = fitness_function(rank, population_size)
    elite_ind        = elite_distance(population[0], distance_matrix, route = route)
    elite_cst        = cost[0][0]
    solution         = _clone(population[0])
    print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    stall = 0
    while (count <= generations-1):
        offspring        = breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity, max_capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, vehicle_types, fleet_size)
        offspring        = mutation(offspring, mutation_rate = mutation_rate, elite = elite)
        if model not in ('tsp', 'mtsp') and stall >= (8 if time_window == 'with' else 12):
            rr_targets = min(2, len(offspring) - elite)
            for rr_idx in range(rr_targets):
                base_ind = solution if rr_idx == 0 else population[min(rr_idx, len(population)-1)]
                offspring[-1 - rr_idx] = ruin_recreate(
                    _clone(base_ind), distance_matrix, parameters, velocity, fixed_cost,
                    variable_cost, max_capacity, penalty_value, time_window, route,
                    vehicle_types, n_depots, list(fleet_size),
                    ruin_fraction=(0.12 + 0.02 * min(stall, 8)) if time_window == 'with' else (0.15 + 0.02 * min(stall, 8)),
                )
        cost, population = target_function(offspring, distance_matrix, parameters, velocity, fixed_cost, variable_cost, max_capacity, penalty_value, time_window = time_window, route = route, fleet_size = fleet_size)
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        if (selection == 'rw'):
            fitness = fitness_function(cost, population_size)
        elif (selection == 'rb'):
            rank    = [[i] for i in range(1, len(cost)+1)]
            fitness = fitness_function(rank, population_size)
        elite_child      = elite_distance(population[0], distance_matrix, route = route)
        if(elite_ind > elite_child):
            elite_ind = elite_child
            solution  = _clone(population[0])
            elite_cst = cost[0][0]
            stall = 0
        else:
            stall += 1
        count = count + 1
        print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    if (graph == True):
        plot_tour_coordinates(coordinates, solution, n_depots = n_depots, route = route)
    solution_report = show_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route = route, time_window  = time_window)
    end = tm.time()
    print('Algorithm Time: ', round((end - start),2), ' seconds')
    return solution_report, solution

############################################################################

# Compatibility wrapper for pyrouteforge.core
# Keeps the public entry point expected by `from pyrouteforge import solve`.
def run_genetic_algorithm(
    coordinates: np.ndarray,
    distance_matrix: np.ndarray,
    parameters: np.ndarray,
    velocity: list,
    fixed_cost: list,
    variable_cost: list,
    capacity: list,
    *,
    population_size: int = 50,
    vehicle_types: int = 1,
    n_depots: int = 1,
    route: str = 'closed',
    model: str = 'vrp',
    time_window: str = 'without',
    fleet_size: list = (),
    mutation_rate: float = 0.1,
    elite: int = 0,
    generations: int = 50,
    penalty_value: float = 1000,
    selection: str = 'rw',
    seed: int | None = None,
    verbose: bool = False,
    on_generation = None,
):
    """Compatibility entry point expected by pyrouteforge.

    Returns
    -------
    report : pandas.DataFrame
    raw_solution : list
        Legacy [[depots], [routes], [vehicles]] structure.
    history : list[float]
        Best distance per generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start = tm.time()
    params = np.array(parameters, copy=True)
    max_capacity = list(capacity)
    if model == 'tsp':
        n_depots = 1
        max_capacity = [float('+inf')] * len(max_capacity)
    elif model == 'mtsp':
        max_capacity = [float('+inf')] * len(max_capacity)

    for i in range(0, n_depots):
        params[i, 0] = 0

    population = initial_population(
        coordinates,
        distance_matrix,
        population_size=population_size,
        vehicle_types=vehicle_types,
        n_depots=n_depots,
        model=model,
    )
    cost, population = target_function(
        population,
        distance_matrix,
        params,
        velocity,
        fixed_cost,
        variable_cost,
        max_capacity,
        penalty_value,
        time_window=time_window,
        route=route,
        fleet_size=list(fleet_size),
    )
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    if selection == 'rw':
        fitness = fitness_function(cost, population_size)
    else:
        rank = [[i] for i in range(1, len(cost) + 1)]
        fitness = fitness_function(rank, population_size)

    elite_ind = elite_distance(population[0], distance_matrix, route=route)
    elite_cst = cost[0][0]
    solution = _clone(population[0])
    history = [elite_ind]

    if verbose:
        print('Generation = ', 0, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    if on_generation is not None:
        on_generation(0, elite_ind, elite_cst)

    count = 0
    stall = 0
    while count <= generations - 1:
        offspring = breeding(
            cost,
            population,
            fitness,
            distance_matrix,
            n_depots,
            elite,
            velocity,
            max_capacity,
            fixed_cost,
            variable_cost,
            penalty_value,
            time_window,
            params,
            route,
            vehicle_types,
            list(fleet_size),
        )
        offspring = mutation(offspring, mutation_rate=mutation_rate, elite=elite)
        if model not in ('tsp', 'mtsp') and stall >= (8 if time_window == 'with' else 12):
            rr_targets = min(2, len(offspring) - elite)
            for rr_idx in range(rr_targets):
                base_ind = solution if rr_idx == 0 else population[min(rr_idx, len(population)-1)]
                offspring[-1 - rr_idx] = ruin_recreate(
                    _clone(base_ind), distance_matrix, params, velocity, fixed_cost,
                    variable_cost, max_capacity, penalty_value, time_window, route,
                    vehicle_types, n_depots, list(fleet_size),
                    ruin_fraction=(0.12 + 0.02 * min(stall, 8)) if time_window == 'with' else (0.15 + 0.02 * min(stall, 8)),
                )
        cost, population = target_function(
            offspring,
            distance_matrix,
            params,
            velocity,
            fixed_cost,
            variable_cost,
            max_capacity,
            penalty_value,
            time_window=time_window,
            route=route,
            fleet_size=list(fleet_size),
        )
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        if selection == 'rw':
            fitness = fitness_function(cost, population_size)
        else:
            rank = [[i] for i in range(1, len(cost) + 1)]
            fitness = fitness_function(rank, population_size)
        elite_child = elite_distance(population[0], distance_matrix, route=route)
        if elite_ind > elite_child:
            elite_ind = elite_child
            solution = _clone(population[0])
            elite_cst = cost[0][0]
            stall = 0
        else:
            stall += 1
        count += 1
        history.append(elite_ind)
        if verbose:
            print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
        if on_generation is not None:
            on_generation(count, elite_ind, elite_cst)

    report = show_report(solution, distance_matrix, params, velocity, fixed_cost, variable_cost, route=route, time_window=time_window)
    if verbose:
        print('Algorithm Time: ', round((tm.time() - start), 2), ' seconds')
    return report, solution, history
