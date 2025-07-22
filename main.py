import pandas as pd
import numpy as np
from mip import Model, xsum, BINARY, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import traceback
import math 

v_xy = 1.5      
v_z_up = 1.0    
v_z_down = 2.0  

def euristica_basi(points, possible_bases, attack_indices):
    if len(possible_bases) <= 10:
        return possible_bases
    
    baricentro = np.mean(np.array(points), axis=0)
    
    base_scores = []
    
    for base in possible_bases:
        dist_bari = np.sqrt((base[0] - baricentro[0])**2 + (base[1] - baricentro[1])**2)
        
        if attack_indices:
            tempo_medio_attacco = sum(calcola_tempo(base, points[i]) for i in attack_indices) / len(attack_indices)
        else:
            tempo_medio_attacco = sum(calcola_tempo(base, points[i]) for i in range(len(points))) / len(points)
        
        score = dist_bari * 0.5 + tempo_medio_attacco * 0.5
        base_scores.append((score, base))
    
    base_scores.sort()
    return [base for score, base in base_scores[:8]]

def calcola_tempo(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    dz = p2[2] - p1[2] 
    
    a = np.sqrt(dx*dx + dy*dy)
    b = abs(dz)
    
    ta = a / v_xy
    
    if dz > 0:  
        tb = b / v_z_up
    else:  
        tb = b / v_z_down
    
    return max(ta, tb)

def calcola_energia(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    dz = p2[2] - p1[2] 
    
    a = np.sqrt(dx*dx + dy*dy)
    b = abs(dz)
    
    consumo = 0
    if a > 0:
        consumo += a * 10  

    if dz > 0:  
        consumo += b * 50    
    else:  
        consumo += b * 5     
    
    return consumo

def arco(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    dz = abs(p1[2] - p2[2])
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if dist <= 4.0:
        return True
    
    if dist <= 11.0:
        return sum(1 for i in range(3) if abs(p1[i] - p2[i]) <= 0.5) >= 2
    
    return False

def genera_basi(x_range, y_range, z):
    bases = []
    for x in range(x_range[0], x_range[1] + 1):
        for y in range(y_range[0], y_range[1] + 1):
            bases.append((x, y, z))
    return bases

def extract_tours_from_solution(x_vars, K, n_total, base_idx, all_points, time_matrix, energy_matrix):
    tours = []
    
    for k in range(K):
        tour = [base_idx]
        current = base_idx
        visited_in_tour = {base_idx}
        
        while True:
            next_node = None
            for j in range(n_total):
                if j != current and (current, j, k) in x_vars:
                    if x_vars[current, j, k].x > 0.5: 
                        next_node = j
                        break
            
            if next_node is None:
                break
            if next_node == base_idx:  
                tour.append(next_node)
                break
            if next_node in visited_in_tour:
                break
                
            tour.append(next_node)
            visited_in_tour.add(next_node)
            current = next_node
        
        if len(tour) > 2:  
            tour_info = calculate_tour_stats(tour, all_points, time_matrix, energy_matrix)
            tours.append(tour_info)
    
    return tours

def calculate_tour_stats(tour, all_points, time_matrix, energy_matrix):
    total_time = 0
    total_energy = 0
    segments = []
    
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        
        segment_time = time_matrix[from_node, to_node]
        segment_energy = energy_matrix[from_node, to_node]
        
        total_time += segment_time
        total_energy += segment_energy
        
        segments.append({
            'from': from_node,
            'to': to_node,
            'from_coords': all_points[from_node],
            'to_coords': all_points[to_node],
            'time': segment_time,
            'energy': segment_energy,
            'cumulative_energy': total_energy
        })
    
    return {
        'tour': tour,
        'total_time': total_time,
        'total_energy': total_energy,
        'segments': segments
    }

def visualize_solution(tours, all_points, base_idx):
    if not tours:
        return
        
    try:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(tours), 1)))
        
        points_array = np.array(all_points[:-1])  
        ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                   c='blue', s=50, alpha=0.6, label='Punti')
        
        base_coords = all_points[base_idx]
        ax.scatter([base_coords[0]], [base_coords[1]], [base_coords[2]], c='red', s=200, marker='^', label=f'Base scelta: {base_coords}')
        
        for i, tour_info in enumerate(tours):
            tour = tour_info['tour']
            color = colors[i % len(colors)]
            
            tour_coords = np.array([all_points[node] for node in tour])
            
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], tour_coords[:, 2], 
                    color=color, linewidth=3, label=f'Viaggio {i+1}', alpha=0.9)
            
            tour_points = [node for node in tour if node != base_idx]
            if tour_points:
                tour_points_coords = np.array([all_points[node] for node in tour_points])
                ax.scatter(tour_points_coords[:, 0], tour_points_coords[:, 1], tour_points_coords[:, 2], color = color, s = 80, alpha = 0.8, edgecolors = 'black', linewidth=1)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("Viaggi del Drone")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Errore nella visualizzazione: {e}")

def add_cuts(model, x, z, n, K, connections, battery_capacity):
    for k in range(K):
        high_energy_arcs = []
        for i in range(n+1):
            for j, _, energy in connections.get(i, []):
                if energy > battery_capacity * 0.25 and (i, j, k) in x:
                    high_energy_arcs.append((i, j))
        
        if high_energy_arcs:
            model.add_constr(xsum(x[i, j, k] for (i, j) in high_energy_arcs if (i, j, k) in x) <= 3)
    
    for k in range(K-1):
        model.add_constr(z[k] >= z[k+1])
    
    total_min_energy = 0
    for i in range(n):
        if i in connections and connections[i]:
            min_energy = min(energy for _, _, energy in connections[i])
            total_min_energy += min_energy
    
    if total_min_energy > 0:
        min_trips = max(1, math.ceil(total_min_energy / (battery_capacity * 0.8)))
        if min_trips <= K:
            model.add_constr(xsum(z[k] for k in range(min_trips)) >= min_trips)


def def_modello(best_time, connections, base_coords, battery_capacity, n):
    base_idx = n
    n_total = n + 1

    if connections:
        total_energy = 0
        total_connections = 0
        for i in connections:
            for _, _, energy in connections[i]:
                total_energy += energy
                total_connections += 1
    
        if total_connections > 0:
            avg_energy_per_move = total_energy / total_connections
        else:
            avg_energy_per_move = 200
    else:
        avg_energy_per_move = 200

    estimated_moves_per_trip = max(1, battery_capacity // avg_energy_per_move)
    K = min(max(5, math.ceil(n / estimated_moves_per_trip)), 25)

    model = Model(sense=minimize)

    # Variabili decisionali:
    x = {}
    for k in range(K):
        for i in range(n_total):
            for j, _, _ in connections.get(i, []):
                if i != j:
                    x[i, j, k] = model.add_var(var_type = BINARY)

    z = {k: model.add_var(var_type = BINARY) for k in range(K)}

    u = {}
    for i in range(n):
        for k in range(K):
            u[i, k] = model.add_var(lb=1, ub=n)
    
    travel_times = {}
    travel_energies = {}
    for i in range(n_total):
        for j, time_val, energy_val in connections.get(i, []):
            travel_times[i, j] = time_val
            travel_energies[i, j] = energy_val

    # F.O:
    model.objective = xsum(travel_times[i, j] * x[i, j, k] for (i, j, k) in x)

    # b&b
    if best_time < float('inf'):
        model.add_constr(
            xsum(travel_times[i, j] * x[i, j, k] for (i, j, k) in x) <= best_time * 0.98
        )

    for i in range(n):
        model.add_constr(xsum(x[j, i, k] for j in range(n_total) for k in range(K) if (j, i, k) in x) == 1)
        model.add_constr(xsum(x[i, j, k] for j in range(n_total) for k in range(K) if (i, j, k) in x) == 1)

    for k in range(K):
        for i in range(n_total):
            model.add_constr(
                xsum(x[i, j, k] for j in range(n_total) if (i, j, k) in x) ==
                xsum(x[j, i, k] for j in range(n_total) if (j, i, k) in x)
            )

    for k in range(K):
        model.add_constr(xsum(x[base_idx, j, k] for j in range(n_total) if (base_idx, j, k) in x) == z[k])
        model.add_constr(xsum(x[j, base_idx, k] for j in range(n_total) if (j, base_idx, k) in x) == z[k])

    for k in range(K):
        model.add_constr(xsum(travel_energies[i, j] * x[i, j, k] for (i, j, kk) in x if kk == k) <= battery_capacity)

    #implicazione logica
    for k in range(K - 1):
        model.add_constr(z[k] >= z[k + 1])

    # eliminazione sottocicli
    M = n + 1
    for k in range(K):
        for i in range(n):
            for j in range(n):
                if i != j and (i, j, k) in x: 
                    model.add_constr(u[i, k] - u[j, k] + 1 <= M * (1 - x[i, j, k]))

    #tagli avanzati
    add_cuts(model, x, z, n, K, connections, battery_capacity)

    model.verbose = 0
    model.max_gap = 0.02
    model.cuts = -1
    model.preprocess = 1
    model.max_seconds = 900
    model.cut_passes = 8
    model.max_nodes = 15000

    status = model.optimize()
    
    if status.name in {"FEASIBLE", "OPTIMAL"}:
        #print(f"Soluzione trovata per la base {base_coords}: {model.objective_value:.2f} secondi")
        return model.objective_value, {
            'x': x,
            'z': z,
            'base': base_coords,
            'status': status.name
        }
    else:
        #print(f"Nessuna soluzione trovata per la base {base_coords}")
        return float('inf'), None

def solve(input_file):
    df = pd.read_csv(input_file)
    points = df[['x', 'y', 'z']].values
    n = len(points)
    
    # Determina configurazione
    filename = input_file.lower()
    if "edificio1" in filename:
        attack_threshold = -12.5
        base_x_range = (-8, 5)
        base_y_range = (-17, -15)
        battery_capacity = 1 * 3600  
    elif "edificio2" in filename:
        attack_threshold = -20
        base_x_range = (-10, 10)
        base_y_range = (-31, -30)
        battery_capacity = 6 * 3600 
    else:
        attack_threshold = -12.5
        base_x_range = (-8, 5)
        base_y_range = (-17, -15)
        battery_capacity = 1 * 3600  
    
    attack_indices = [i for i in range(n) if points[i][1] <= attack_threshold]
    possible_bases = genera_basi(base_x_range, base_y_range, 0)
    candidate_bases = euristica_basi(points, possible_bases, attack_indices)
    n_total = n + 1

    connections = {}
    for i in range(n): 
        connections[i] = []
        for j in range(n):
            if i == j:
                continue
            if arco(points[i], points[j]):
                connections[i].append((j, calcola_tempo(points[i], points[j]), calcola_energia(points[i], points[j])))

    best_base = {
        'base_coords': None,
        'objective': float('inf'),
        'connections': None,
        'solution': None
    }

    for base in candidate_bases:
        temp_connections = {i: list(connections[i]) for i in range(n)}
        temp_connections[n] = []
        if attack_indices:
            for nodo_attacco in attack_indices:
                temp_connections[n].append((nodo_attacco, calcola_tempo(base, points[nodo_attacco]), calcola_energia(base, points[nodo_attacco])))
                temp_connections[nodo_attacco].append((n, calcola_tempo(points[nodo_attacco], base), calcola_energia(points[nodo_attacco], base)))
        else:
            for i in range(n):
                temp_connections[n].append((i, calcola_tempo(base, points[i]), calcola_energia(base, points[i])))
                temp_connections[i].append((n, calcola_tempo(points[i], base), calcola_energia(points[i], base)))

        current_time, current_solution = def_modello(best_base['objective'], temp_connections, base, battery_capacity, n)

        if current_time < best_base['objective']:
            best_base = {
                'base_coords': base,
                'objective': current_time,
                'connections': temp_connections,
                'solution': current_solution
            }
            if current_solution and current_solution.get('gap', 1.0) < 0.01:
                break

    if best_base['solution'] and best_base['solution']['status'] in ["OPTIMAL", "FEASIBLE"]:
        base_coords = best_base['base_coords']
        base_idx = n 
        all_points = list(points) + [base_coords]

        time_matrix = {}
        energy_matrix = {}
        for i in range(n_total):
            for j, time_val, energy_val in best_base['connections'].get(i, []):
                time_matrix[i, j] = time_val
                energy_matrix[i, j] = energy_val
        
        x_vars = best_base['solution']['x']
        tours = extract_tours_from_solution(x_vars, K=n, n_total=n + 1, base_idx=base_idx, all_points=all_points, time_matrix=time_matrix, energy_matrix=energy_matrix)
        
        for i, tour_info in enumerate(tours, 1):
            tour = tour_info['tour']
    
            converted_tour = []
            for node in tour:
                if node == base_idx: 
                    converted_tour.append(0)
                else:  
                    converted_tour.append(node + 1)
        
            print(f"Viaggio {i}: {'-'.join(map(str, converted_tour))}")

        visualize_solution(tours, all_points, base_idx)

    return best_base['solution']

def main():
    sol = solve(sys.argv[1])
    if not sol:
        print("\nNessuna soluzione trovata")

if __name__ == '__main__':
    main()