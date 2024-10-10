# non euc better than previous ones, still does not work for 500
# 100 under 10s
# struggles with it beyond that

# Imports
import random
import time
import numpy as np
from typing import List, Tuple

City = complex  # Represent city coordinates as complex numbers
Cities = frozenset  # A set of cities
Tour = List[City]  # A list of cities visited in order

random.seed(0)  # Ensure consistency across runs

# Global distance matrix for non-Euclidean TSP
DISTANCE_MATRIX = None

def distance(A: City, B: City, use_matrix=False, index_a=None, index_b=None) -> float:
    """Compute distance between two cities. Use distance matrix for non-Euclidean, Euclidean otherwise."""
    if use_matrix:
        return DISTANCE_MATRIX[index_a][index_b]
    else:
        return abs(A - B)  # Euclidean distance

def tour_length(tour: Tour, use_matrix=False, city_indices=None) -> float:
    """Calculate total length of the tour, including the link from last city back to the first."""
    if use_matrix:
        return sum(distance(tour[i], tour[i-1], use_matrix, city_indices[tour[i]], city_indices[tour[i-1]]) 
                   for i in range(len(tour)))
    else:
        return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))

def nearest_neighbor(A: City, cities, use_matrix=False, index_A=None, city_indices=None) -> City:
    """Find the nearest city to city A using either the Euclidean distance or the distance matrix."""
    if use_matrix:
        return min(cities, key=lambda C: DISTANCE_MATRIX[index_A][city_indices[C]])
    else:
        return min(cities, key=lambda C: distance(A, C))

def nearest_tsp(cities, start=None, use_matrix=False) -> Tour:
    """Generate a tour using the nearest neighbor heuristic, handling both Euclidean and non-Euclidean cases."""
    start = start or next(iter(cities))
    tour = [start]
    unvisited = set(cities) - {start}
    
    if use_matrix:
        city_indices = {city: idx for idx, city in enumerate(cities)}  # Map cities to matrix indices
        index_A = city_indices[start]
        
        while unvisited:
            nearest = nearest_neighbor(tour[-1], unvisited, use_matrix=True, index_A=index_A, city_indices=city_indices)
            tour.append(nearest)
            unvisited.remove(nearest)
            index_A = city_indices[nearest]  # Update index for next iteration
    else:
        while unvisited:
            nearest = nearest_neighbor(tour[-1], unvisited)
            tour.append(nearest)
            unvisited.remove(nearest)
    
    return tour

def opt2(tour, use_matrix=False, city_indices=None) -> Tour:
    """Perform 2-opt optimization to improve the tour length."""
    changed = True
    while changed:
        changed = False
        for (i, j) in subsegments(len(tour)):
            if reversal_is_improvement(tour, i, j, use_matrix, city_indices):
                tour[i:j] = reversed(tour[i:j])
                changed = True
    return tour

import random

def _perturbation(x: List[int], perturbation_scheme: str) -> List[int]:
    """Generate a single random neighbor of the current solution using 2-opt."""
    if perturbation_scheme == "two_opt":
        i, j = sorted(random.sample(range(len(x)), 2))  # Choose two random indices
        new_solution = x[:]  # Make a copy of the current solution
        new_solution[i:j] = reversed(new_solution[i:j])  # Reverse the segment between i and j
        return new_solution
    else:
        raise ValueError(f"Perturbation scheme '{perturbation_scheme}' not supported.")


def two_opt_neighborhood(solution: List[int]):
    """Generate 2-opt neighbors of the given solution."""
    n = len(solution)
    for i in range(n - 1):
        for j in range(i + 2, n):
            neighbor = solution[:]
            # Reverse the segment between i and j
            neighbor[i:j] = reversed(solution[i:j])
            yield neighbor

# Define neighborhood_gen to support 'two_opt' scheme
neighborhood_gen = {
    'two_opt': two_opt_neighborhood
}


def subsegments(N):
    """Generate subsegments for 2-opt optimization."""
    return tuple((i, i + length) for length in reversed(range(2, N - 1)) for i in range(N - length))

def reversal_is_improvement(tour, i, j, use_matrix=False, city_indices=None) -> bool:
    """Check if reversing a segment improves the tour length."""
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
    if use_matrix:
        idx_A, idx_B, idx_C, idx_D = city_indices[A], city_indices[B], city_indices[C], city_indices[D]
        return (DISTANCE_MATRIX[idx_A][idx_B] + DISTANCE_MATRIX[idx_C][idx_D] >
                DISTANCE_MATRIX[idx_A][idx_C] + DISTANCE_MATRIX[idx_B][idx_D])
    else:
        return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)

def rep_opt2_nearest_tsp(cities, k=10, use_matrix=False) -> Tour:
    """Apply nearest neighbor with 2-opt optimization over multiple initializations."""
    city_indices = {city: idx for idx, city in enumerate(cities)} if use_matrix else None
    return min((opt2(nearest_tsp(cities, start, use_matrix), use_matrix, city_indices) 
                for start in random.sample(list(cities), min(k, len(cities)))), 
               key=lambda t: tour_length(t, use_matrix, city_indices))

def solve_tsp_local_search(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """Local search algorithm for improving non-Euclidean TSP solutions."""
    x, fx = setup_initial_solution(distance_matrix)
    while True:
        improvement = False
        for xn in neighborhood_gen['two_opt'](x):  # Use 2-opt neighbors
            fn = compute_permutation_distance(distance_matrix, xn)
            if fn < fx:
                x, fx = xn, fn
                improvement = True
                break  # Stop after first improvement
        if not improvement:
            break
    return x, fx

def solve_tsp_simulated_annealing(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """Simulated Annealing for solving non-Euclidean TSP."""
    x, fx = setup_initial_solution(distance_matrix)
    temp = _initial_temperature(distance_matrix, x, fx, 'two_opt')
    while temp > 1:
        for xn in neighborhood_gen['two_opt'](x):
            fn = compute_permutation_distance(distance_matrix, xn)
            if fn < fx or np.random.rand() < np.exp((fx - fn) / temp):
                x, fx = xn, fn
        temp *= 0.9  # Cooling
    return x, fx

def compute_permutation_distance(distance_matrix: np.ndarray, permutation: List[int]) -> float:
    """
    Calculate the total distance of a given tour (permutation) based on the distance matrix.
    
    Parameters:
    - distance_matrix: A 2D numpy array where entry [i, j] is the distance from city i to city j.
    - permutation: A list of city indices representing the tour.

    Returns:
    - The total distance of the tour.
    """
    total_distance = 0
    n = len(permutation)

    # Sum the distance between consecutive cities in the tour
    for i in range(n):
        total_distance += distance_matrix[permutation[i]][permutation[(i + 1) % n]]  # Wrap around to the start

    return total_distance

def setup_initial_solution(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """Initialize a random tour and calculate its length."""
    n = distance_matrix.shape[0]
    initial_tour = list(range(n))  # Create a list [0, 1, ..., n-1]
    random.shuffle(initial_tour)  # Randomize the initial tour
    initial_length = compute_permutation_distance(distance_matrix, initial_tour)
    return initial_tour, initial_length

import numpy as np

def _initial_temperature(distance_matrix: np.ndarray, x: List[int], fx: float, perturbation_scheme: str) -> float:
    """Estimate the initial temperature for Simulated Annealing."""
    # Step 1: Generate perturbations and calculate the mean change in distance
    dfx_list = []
    for _ in range(100):  # Evaluate 100 random perturbations
        xn = _perturbation(x, perturbation_scheme)  # Generate a new tour via perturbation
        fn = compute_permutation_distance(distance_matrix, xn)  # Calculate distance for new tour
        dfx_list.append(fn - fx)  # Store the change in distance

    # Step 2: Compute the average change in distance (absolute value)
    dfx_mean = np.mean(np.abs(dfx_list))

    # Step 3: Define tau0 (a parameter to control the initial temperature)
    tau0 = 0.5

    # Step 4: Calculate the initial temperature T0
    return -dfx_mean / np.log(tau0)

def read_input():
    """Read input for TSP problem and handle both Euclidean and non-Euclidean cases."""
    tsp_type = input().strip()  # Either "EUCLIDEAN" or "NON-EUCLIDEAN"
    n = int(input())  # Number of cities
    cities = [complex(*map(float, input().split())) for _ in range(n)]  # Parse city coordinates
    distances = [list(map(float, input().split())) for _ in range(n)] if tsp_type == "NON-EUCLIDEAN" else None
    return tsp_type, cities, distances

def main():
    tsp_type, cities, distances = read_input()
    
    if tsp_type == "NON-EUCLIDEAN":
        global DISTANCE_MATRIX
        DISTANCE_MATRIX = np.array(distances)  # Set global distance matrix for non-Euclidean TSP
        start_time = time.perf_counter()  # Start timing
        permutation, distance = solve_tsp_simulated_annealing(DISTANCE_MATRIX)
        permutation2, distance2 = solve_tsp_local_search(DISTANCE_MATRIX)
        end_time = time.perf_counter()
        print(permutation2)
        print(f"Cost of Shortest Route (Non-Euclidean): {distance2:.2f}")
        print(f"Time Taken (Non-Euclidean): {end_time - start_time:.4f} seconds")
    else:
        start_time = time.perf_counter()
        best_tour_heuristic = rep_opt2_nearest_tsp(frozenset(cities), use_matrix=False)
        best_length = tour_length(best_tour_heuristic, use_matrix=False)
        end_time = time.perf_counter()
        print(f"Best Tour (Euclidean): {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")
        print(f"Cost of Shortest Route (Euclidean): {best_length:.2f}")
        print(f"Time Taken (Euclidean): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()