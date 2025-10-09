#vehicle routing problem

import numpy as np
import random

def distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return dist

def roulette_wheel_selection(probabilities):
    cumulative_sum = np.cumsum(probabilities)
    r = random.random()
    for idx, val in enumerate(cumulative_sum):
        if r <= val:
            return idx
    return len(probabilities) - 1

def construct_solution(cities, demands, vehicle_capacity, tau, eta, alpha, beta):
    n = len(cities)
    depot = 0
    unvisited = set(range(1, n))  # all customers except depot
    solution = []
    
    while unvisited:
        route = [depot]
        load = 0
        current = depot
        
        while True:
            allowed = [cust for cust in unvisited if demands[cust] + load <= vehicle_capacity]
            if not allowed:
                break
            
            probabilities = []
            for cust in allowed:
                pheromone = tau[current][cust] ** alpha
                heuristic = eta[current][cust] ** beta
                probabilities.append(pheromone * heuristic)
            
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            
            selected_idx = roulette_wheel_selection(probabilities)
            next_customer = allowed[selected_idx]
            
            route.append(next_customer)
            load += demands[next_customer]
            unvisited.remove(next_customer)
            current = next_customer
        
        route.append(depot)  # return to depot
        solution.append(route)
    
    return solution

def route_length(route, dist):
    length = 0
    for i in range(len(route) - 1):
        length += dist[route[i]][route[i+1]]
    return length

def total_length(solution, dist):
    return sum(route_length(route, dist) for route in solution)

def update_pheromone(tau, solutions, lengths, rho):
    tau *= (1 - rho)  # evaporation
    for solution, length in zip(solutions, lengths):
        deposit = 1 / length
        for route in solution:
            for i in range(len(route) - 1):
                a, b = route[i], route[i+1]
                tau[a][b] += deposit
                tau[b][a] += deposit  # symmetric graph

def ant_colony_vrp(cities, demands, vehicle_capacity, m=10, T=100, alpha=1.0, beta=5.0, rho=0.5, tau0=1.0):
    n = len(cities)
    dist = distance_matrix(cities)
    tau = np.full((n, n), tau0)
    np.fill_diagonal(dist, np.inf)  # avoid division by zero
    eta = 1 / dist
    
    best_solution = None
    best_length = float('inf')
    
    for iteration in range(1, T+1):
        all_solutions = []
        all_lengths = []
        
        for ant in range(m):
            solution = construct_solution(cities, demands, vehicle_capacity, tau, eta, alpha, beta)
            length = total_length(solution, dist)
            all_solutions.append(solution)
            all_lengths.append(length)
            
            if length < best_length:
                best_solution = solution
                best_length = length
        
        update_pheromone(tau, all_solutions, all_lengths, rho)
        
        print(f"Iteration {iteration}: Best length so far = {best_length}")
    
    print("\nFinal best solution (routes):")
    for idx, route in enumerate(best_solution):
        print(f" Vehicle {idx+1}: {route}")
    print(f"Final best length: {best_length}")
    
    return best_solution, best_length

# Example usage:
if __name__ == "__main__":
    cities = [
        (0, 0),   # depot
        (1, 5),
        (5, 2),
        (6, 6),
        (8, 3),
        (7, 9)
    ]
    
    demands = [0, 2, 3, 1, 4, 2]  # depot demand is 0
    vehicle_capacity = 5
    
    best_solution, best_length = ant_colony_vrp(cities, demands, vehicle_capacity)
