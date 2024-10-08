# ew, but gives us some tour better than prev for 500 noneuc

import random
import time
import numpy as np
from typing import List, Tuple, Callable

City = complex  # Represent city coordinates as complex numbers
Cities = frozenset  # A set of cities
Tour = List[int]  # A list of city indices visited in order

random.seed(0)  # Ensure consistency across runs

DISTANCE_MATRIX = None  # Global distance matrix for non-Euclidean TSP

def distance(A: int, B: int, use_matrix: bool = False) -> float:
    """Compute distance between two cities."""
    return DISTANCE_MATRIX[A][B] if use_matrix else abs(cities[A] - cities[B])

def tour_length(tour: Tour, use_matrix: bool = False) -> float:
    """Calculate total length of the tour."""
    return sum(distance(tour[i], tour[(i+1) % len(tour)], use_matrix) for i in range(len(tour)))

def nearest_neighbor(A: int, unvisited: set, use_matrix: bool = False) -> int:
    """Find the nearest unvisited city to city A."""
    return min(unvisited, key=lambda C: distance(A, C, use_matrix))

def nearest_tsp(cities_set: Cities, use_matrix: bool = False) -> Tour:
    """Generate a tour using the nearest neighbor heuristic."""
    start = random.choice(list(cities_set))
    tour = [start]
    unvisited = set(cities_set) - {start}
    while unvisited:
        nearest = nearest_neighbor(tour[-1], unvisited, use_matrix)
        tour.append(nearest)
        unvisited.remove(nearest)
    return tour

def opt2(tour: Tour, use_matrix: bool = False) -> Tour:
    """Perform 2-opt optimization to improve the tour length."""
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue
                if tour_length(tour[i-1:j+1], use_matrix) > tour_length(tour[i-1:j+1][::-1], use_matrix):
                    tour[i:j] = tour[i:j][::-1]
                    improved = True
        if improved: break  # Stop after first improvement for faster runtime
    return tour

def rep_opt2_nearest_tsp(cities_set: Cities, k: int = 10, use_matrix: bool = False) -> Tour:
    """Apply nearest neighbor with 2-opt optimization over multiple initializations."""
    return min((opt2(nearest_tsp(cities_set, use_matrix), use_matrix) 
                for _ in range(k)), 
               key=lambda t: tour_length(t, use_matrix))

def simulated_annealing(initial_tour: Tour, initial_temp: float, cooling_rate: float, use_matrix: bool = False) -> Tour:
    """Simulated Annealing for solving TSP."""
    current_tour = initial_tour
    current_length = tour_length(current_tour, use_matrix)
    temp = initial_temp
    
    while temp > 1:
        i, j = sorted(random.sample(range(len(current_tour)), 2))
        new_tour = current_tour[:]
        new_tour[i:j] = reversed(new_tour[i:j])
        new_length = tour_length(new_tour, use_matrix)
        
        if new_length < current_length or random.random() < np.exp((current_length - new_length) / temp):
            current_tour, current_length = new_tour, new_length
        
        temp *= cooling_rate
    
    return current_tour

def solve_tsp(use_matrix: bool = False) -> Tuple[Tour, float]:
    """Solve TSP using a combination of heuristics and optimization techniques."""
    initial_tour = rep_opt2_nearest_tsp(set(range(len(cities))), k=5, use_matrix=use_matrix)
    initial_temp = 100.0
    cooling_rate = 0.995
    
    best_tour = simulated_annealing(initial_tour, initial_temp, cooling_rate, use_matrix)
    best_length = tour_length(best_tour, use_matrix)
    
    return best_tour, best_length

def read_input() -> Tuple[str, List[City], np.ndarray]:
    """Read input for TSP problem."""
    tsp_type = input().strip()
    n = int(input())
    global cities
    cities = [complex(*map(float, input().split())) for _ in range(n)]
    distances = np.array([list(map(float, input().split())) for _ in range(n)]) if tsp_type == "NON-EUCLIDEAN" else None
    return tsp_type, cities, distances

def main():
    tsp_type, cities_list, distances = read_input()
    global DISTANCE_MATRIX
    DISTANCE_MATRIX = distances

    use_matrix = tsp_type == "NON-EUCLIDEAN"
    
    start_time = time.perf_counter()
    best_tour, best_length = solve_tsp(use_matrix)
    
    while time.perf_counter() - start_time < 290:  # Run for up to 290 seconds (leaving 10s buffer)
        new_tour, new_length = solve_tsp(use_matrix)
        if new_length < best_length:
            best_tour, best_length = new_tour, new_length
            print(' '.join(map(str, best_tour)))  # Output current best tour

    print(f"Final tour: {' '.join(map(str, best_tour))}")
    print(f"Final tour length: {best_length:.2f}")

if __name__ == "__main__":
    main()