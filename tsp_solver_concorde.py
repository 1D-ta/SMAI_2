import random
import time
import numpy as np
from typing import List, Tuple, Set
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree

City = complex
Cities = frozenset
Tour = List[City]

random.seed(0)

city_to_index = {}
DISTANCE_MATRIX = None

def distance(A: City, B: City) -> float:
    global DISTANCE_MATRIX
    return DISTANCE_MATRIX[city_to_index[A], city_to_index[B]]

def tour_length(tour: Tour) -> float:
    return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))

def read_input():
    tsp_type = input().strip()
    n = int(input().strip())
    cities = [complex(*map(float, input().split())) for _ in range(n)]
    global city_to_index
    city_to_index = {city: idx for idx, city in enumerate(cities)}
    distances = [list(map(float, input().split())) for _ in range(n)]
    return tsp_type, cities, distances

def plot_tour(tour: Tour):
    x = [city.real for city in tour]
    y = [city.imag for city in tour]
    plt.figure(figsize=(10, 6))
    plt.plot(x + [tour[0].real], y + [tour[0].imag], marker='o', linestyle='-', color='b')
    for i, city in enumerate(tour):
        plt.annotate(f'{city_to_index[city]}', (city.real, city.imag), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('TSP Tour Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.axis('equal')
    plt.show()

def calculate_1tree_lower_bound(cities: Cities) -> float:
    n = len(cities)
    city_list = list(cities)
    dist_matrix = np.array([[distance(city1, city2) for city2 in city_list] for city1 in city_list])
    
    # Remove first city for MST calculation
    mst = minimum_spanning_tree(dist_matrix[1:, 1:])
    mst_weight = mst.sum()
    
    # Find two cheapest edges connecting to the first city
    first_city_edges = dist_matrix[0, 1:]
    cheapest_edges = np.partition(first_city_edges, 1)[:2]
    
    return mst_weight + sum(cheapest_edges)

def nearest_neighbor_tour(cities: Cities) -> Tour:
    unvisited = set(cities)
    tour = [unvisited.pop()]
    while unvisited:
        nearest = min(unvisited, key=lambda city: distance(tour[-1], city))
        tour.append(nearest)
        unvisited.remove(nearest)
    return tour

def two_opt_swap(tour: Tour, i: int, j: int) -> Tour:
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def two_opt(tour: Tour) -> Tour:
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue
                new_tour = two_opt_swap(tour, i, j)
                if tour_length(new_tour) < tour_length(tour):
                    tour = new_tour
                    improved = True
    return tour

def get_candidate_set(city: City, cities: Cities, size: int = 5) -> Set[City]:
    return set(sorted(cities - {city}, key=lambda c: distance(city, c))[:size])

def branch_and_bound_tsp(cities: Cities) -> Tour:
    best_tour = two_opt(nearest_neighbor_tour(cities))
    best_length = tour_length(best_tour)
    
    def branch(partial_tour: Tour, unvisited: Cities, lower_bound: float):
        nonlocal best_tour, best_length
        
        if not unvisited:
            current_length = tour_length(partial_tour)
            if current_length < best_length:
                best_length = current_length
                best_tour = partial_tour.copy()
            return
        
        if lower_bound >= best_length:
            return
        
        last_city = partial_tour[-1]
        candidates = get_candidate_set(last_city, unvisited)
        
        for next_city in candidates:
            new_partial_tour = partial_tour + [next_city]
            new_unvisited = unvisited - {next_city}
            new_lower_bound = max(lower_bound, calculate_1tree_lower_bound(set(new_partial_tour) | new_unvisited))
            branch(new_partial_tour, new_unvisited, new_lower_bound)
    
    start = next(iter(cities))
    initial_lower_bound = calculate_1tree_lower_bound(cities)
    branch([start], cities - {start}, initial_lower_bound)
    return best_tour

def main():
    tsp_type, cities, distances = read_input()
    global DISTANCE_MATRIX 
    DISTANCE_MATRIX = np.array(distances)

    start_time = time.perf_counter()
    print("started")
    best_tour = branch_and_bound_tsp(frozenset(cities))
    best_length = tour_length(best_tour)
    end_time = time.perf_counter()
    
    print(f"Cost of Route: {best_length:.2f}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")

    plot_tour(best_tour)

if __name__ == "__main__":
    main()