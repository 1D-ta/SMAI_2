# Vandita Lodha and Suhani Jain
# SMAI Assignment 2

# imports
import functools
import itertools
import random
import time
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple

City = complex  # e.g. City(300, 100)
Cities = frozenset  # A set of cities
Tour = List[City]  # A list of cities visited, in order

random.seed(0) # to ensure consistency

# Global distance matrix for non-Euclidean TSP
DISTANCE_MATRIX = None

def distance(A: City, B: City, use_matrix=False, index_a=None, index_b=None) -> float:
    """
    Compute the distance between two cities.
    If use_matrix is True, use the precomputed distance matrix.
    """
    if use_matrix:
        return DISTANCE_MATRIX[index_a][index_b]  # Use distance matrix for Non-Euclidean TSP
    else:
        return abs(A - B)  # Euclidean distance for standard TSP

def tour_length(tour: Tour, use_matrix=False, city_indices=None) -> float:
    """
    The total distance of each link in the tour, including the link from last back to first.
    """
    if use_matrix:
        # Use matrix, so we need to pass the city indices for lookups
        return sum(distance(tour[i], tour[i - 1], use_matrix, city_indices[tour[i]], city_indices[tour[i - 1]]) 
                   for i in range(len(tour)))
    else:
        # Euclidean case, no need for city_indices
        return sum(distance(tour[i], tour[i - 1]) for i in range(len(tour)))

def valid_tour(tour: Tour, cities: Cities) -> bool:
    # Does `tour` visit every city in `cities` exactly once?
    return Counter(tour) == Counter(cities)

def nearest_neighbor(A: City, cities, use_matrix=False, index_A=None, city_indices=None) -> City:
    # Find the city nearest to city A using Euclidean or distance matrix.
    if use_matrix:
        # Use the distance matrix to find the nearest city
        return min(cities, key=lambda C: DISTANCE_MATRIX[index_A][city_indices[C]])
    else:
        # Euclidean case
        return min(cities, key=lambda C: distance(A, C))

def nearest_tsp(cities, start=None, use_matrix=False) -> Tour:
    """
    Generate a tour using the nearest neighbour heuristic for Non-Euclidean or Euclidean cases.
    """
    start = start or next(iter(cities))
    tour = [start]
    unvisited = set(cities) - {start}
    
    if use_matrix:
        # Map each city to its index in the distance matrix
        city_indices = {city: idx for idx, city in enumerate(cities)}
        index_A = city_indices[start]  # Get the index of the starting city
        
        while unvisited:
            nearest = nearest_neighbor(tour[-1], unvisited, use_matrix=True, index_A=index_A, city_indices=city_indices)
            tour.append(nearest)
            unvisited.remove(nearest)
            index_A = city_indices[nearest]  # Update the index for the next iteration
    else:
        # Euclidean case
        while unvisited:
            nearest = nearest_neighbor(tour[-1], unvisited)
            tour.append(nearest)
            unvisited.remove(nearest)
    
    return tour

@functools.lru_cache(None)
def subsegments(N):
    return tuple((i, i + length)
                 for length in reversed(range(2, N - 1))
                 for i in range(N - length))

def opt2(tour, use_matrix=False, city_indices=None) -> Tour:
    """Perform 2-opt segment reversals to optimize tour.
    Handles both Euclidean and Non-Euclidean cases."""
    changed = True
    while changed:
        changed = False
        for (i, j) in subsegments(len(tour)):
            if reversal_is_improvement(tour, i, j, use_matrix, city_indices):
                tour[i:j] = reversed(tour[i:j])
                changed = True
    return tour

def reversal_is_improvement(tour, i, j, use_matrix=False, city_indices=None) -> bool:
    """
    Check if reversing the segment between i and j improves the tour length.
    """
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
    if use_matrix:
        idx_A, idx_B = city_indices[A], city_indices[B]
        idx_C, idx_D = city_indices[C], city_indices[D]
        return (DISTANCE_MATRIX[idx_A][idx_B] + DISTANCE_MATRIX[idx_C][idx_D] >
                DISTANCE_MATRIX[idx_A][idx_C] + DISTANCE_MATRIX[idx_B][idx_D])
    else:
        # Euclidean case
        return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)

def rep_opt2_nearest_tsp(cities, k=10, use_matrix=False) -> Tour:
    """
    Apply 2-opt to *each* of the repeated nearest neighbors tours.
    """
    city_indices = {city: idx for idx, city in enumerate(cities)} if use_matrix else None
    return min((opt2(nearest_tsp(cities, start, use_matrix), use_matrix, city_indices) 
               for start in random.sample(list(cities), min(k, len(cities)))),
               key=lambda t: tour_length(t, use_matrix, city_indices))

def brute_force_tsp(cities: List[City], use_matrix=False) -> Tour:
    # checking all permutations.
    city_indices = {city: idx for idx, city in enumerate(cities)} if use_matrix else None
    best_tour = None
    best_length = float('inf')
    
    for perm in itertools.permutations(cities):
        current_length = tour_length(perm, use_matrix, city_indices)
        if current_length < best_length:
            best_length = current_length
            best_tour = perm
            
    return best_tour

def plot_tour(tour: Tour):
    """
    Plot the tour.
    The starting city is marked in red, and the remaining cities are marked in blue.
    """
    plt.figure(figsize=(8, 6))
    X = [city.real for city in tour + [tour[0]]]
    Y = [city.imag for city in tour + [tour[0]]]
    
    plt.plot(X, Y, 'bo-', label="Tour Path")
    
    plt.plot(tour[0].real, tour[0].imag, 'ro', markersize=10, label="Starting Point")  # Red dot for the start.
    
    plt.title('Optimal Tour Path')
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.show()

def read_input():
    #Reads the input for TSP
    tsp_type = input().strip()  # Read TSP type (EUCLIDEAN or NON-EUCLIDEAN)
    n = int(input())  # Read number of cities
    cities = []
    
    for _ in range(n):
        x, y = map(float, input().split())
        cities.append(complex(x, y))  # Store coordinates as complex numbers
    
    distances = []
    
    if tsp_type == "NON-EUCLIDEAN":
        for _ in range(n):
            distances.append(list(map(float, input().split())))  # Read distance matrix
    
    return tsp_type, cities, distances

def main():
    tsp_type, cities, distances = read_input()
    
    if tsp_type == "NON-EUCLIDEAN":
        global DISTANCE_MATRIX  # Set the global distance matrix
        DISTANCE_MATRIX = distances  # Store the distance matrix
        use_matrix = True
    else:
        use_matrix = False  # For Euclidean case, no matrix needed
    
    start_time = time.perf_counter()  # Start timing
    # Solve using nearest neighbor + 2-opt heuristic
    best_tour_heuristic = rep_opt2_nearest_tsp(frozenset(cities), use_matrix=use_matrix)  # Pass the use_matrix flag
    end_time_heuristic = time.perf_counter()
    
    best_length_heuristic = tour_length(best_tour_heuristic, use_matrix, {city: idx for idx, city in enumerate(cities)})
    
    # Print results
    print(f"Best Tour (Heuristic): {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")
    print(f"Cost of Shortest Route (Heuristic): {best_length_heuristic:.2f}")
    print(f"Time Taken (Heuristic): {end_time_heuristic - start_time:.4f} seconds")
    
    # Plot the heuristic tour
    #plot_tour(best_tour_heuristic)

    # Brute force solution (commented out for large cases)
    '''
    start_time_brute = time.perf_counter()  # Start timing for brute force
    best_tour_brute = brute_force_tsp(cities, use_matrix=use_matrix)  # Solve using brute force
    end_time_brute = time.perf_counter()

    best_length_brute = tour_length(best_tour_brute, use_matrix, {city: idx for idx, city in enumerate(cities)})  # Calculate the tour length

    print(f"Best Tour (Brute Force): {' '.join(str(cities.index(city)) for city in best_tour_brute)}")
    print(f"Cost of Shortest Route (Brute Force): {best_length_brute:.2f}")
    print(f"Time Taken (Brute Force): {end_time_brute - start_time_brute:.4f} seconds")

    plot_tour(best_tour_brute)
    '''

if __name__ == "__main__":
    main()