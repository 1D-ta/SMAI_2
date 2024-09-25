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

def distance(A: City, B: City) -> float:
    "Distance between two cities"
    return abs(A - B)

def tour_length(tour: Tour) -> float:
    "The total distances of each link in the tour, including the link from last back to first."
    return sum(distance(tour[i], tour[i - 1]) for i in range(len(tour)))

def valid_tour(tour: Tour, cities: Cities) -> bool:
    "Does `tour` visit every city in `cities` exactly once?"
    return Counter(tour) == Counter(cities)

def nearest_neighbor(A: City, cities) -> City:
    """Find the city C in cities that is nearest to city A."""
    return min(cities, key=lambda C: distance(C, A))

def nearest_tsp(cities, start=None) -> Tour:
    """Create a partial tour that initially is just the start city."""
    start = start or next(iter(cities))
    tour = [start]
    unvisited = set(cities) - {start}
    
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

def opt2(tour) -> Tour:
    "Perform 2-opt segment reversals to optimize tour."
    changed = True
    while changed:
        changed = False
        for (i, j) in subsegments(len(tour)):
            if reversal_is_improvement(tour, i, j):
                tour[i:j] = reversed(tour[i:j])
                changed = True
    return tour

def reversal_is_improvement(tour, i, j) -> bool:
    "Would reversing the segment `tour[i:j]` make the tour shorter?"
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
    return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)

def rep_opt2_nearest_tsp(cities, k=10) -> Tour:
    """Apply 2-opt to *each* of the repeated nearest neighbors tours."""
    return min((opt2(nearest_tsp(cities, start)) for start in random.sample(list(cities), min(k, len(cities)))),
               key=tour_length)

def brute_force_tsp(cities: List[City]) -> Tour:
    """Find the optimal tour using brute force by checking all permutations."""
    best_tour = None
    best_length = float('inf')
    
    for perm in itertools.permutations(cities):
        current_length = tour_length(perm)
        if current_length < best_length:
            best_length = current_length
            best_tour = perm
            
    return best_tour

def plot_tour(tour: Tour):
    plt.figure(figsize=(8, 6))
    X = [city.real for city in tour + [tour[0]]]
    Y = [city.imag for city in tour + [tour[0]]]
    plt.plot(X, Y, 'bo-')
    plt.title('Optimal Tour Path')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def read_input():
    tsp_type = input().strip()  # Read TSP type (EUCLIDEAN or NON-EUCLIDEAN)
    n = int(input())  # Read number of cities
    cities = []
    
    for _ in range(n):
        x, y = map(float, input().split())
        cities.append(complex(x, y))  # Store coordinates as complex numbers
    
    distances = []
    
    for _ in range(n):
        distances.append(list(map(float, input().split())))  # Read distance matrix
    
    return tsp_type, cities, distances

def main():
    tsp_type, cities, distances = read_input()
    
    start_time = time.perf_counter()  # Start timing for nearest neighbor + 2-opt
    best_tour_heuristic = rep_opt2_nearest_tsp(frozenset(cities))  # Find best tour using heuristic
    end_time_heuristic = time.perf_counter()  # End timing
    
    best_length_heuristic = tour_length(best_tour_heuristic)  # Calculate length of best heuristic tour
    
    print(f"Best Tour (Heuristic): {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")  # Output path representation
    print(f"Cost of Shortest Route (Heuristic): {best_length_heuristic:.2f}")  # Output cost of shortest route (heuristic)

    print(f"Time Taken (Heuristic): {end_time_heuristic - start_time:.4f} seconds")  # Output time taken for heuristic
    
    plot_tour(best_tour_heuristic)  # Plot the path from heuristic solution

    # Brute force solution
    start_time_brute_force = time.perf_counter()  # Start timing for brute force
    best_tour_brute_force = brute_force_tsp(cities)  # Find best tour using brute force
    end_time_brute_force = time.perf_counter()  # End timing
    
    best_length_brute_force = tour_length(best_tour_brute_force)  # Calculate length of best brute force tour
    
    print(f"Best Tour (Brute Force): {' '.join(str(cities.index(city)) for city in best_tour_brute_force)}")  # Output path representation
    print(f"Cost of Shortest Route (Brute Force): {best_length_brute_force:.2f}")  # Output cost of shortest route (brute force)
    
    print(f"Time Taken (Brute Force): {end_time_brute_force - start_time_brute_force:.4f} seconds")  # Output time taken for brute force


if __name__ == "__main__":
    main()
