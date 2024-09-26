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

random.seed(0) # to ensure randomisation is consistent between runs

City = complex  # A city is represented as a complex number (x, y coordinates).
Cities = frozenset  # A set of cities.
Tour = List[City]  # A list of cities visited in a specific order.

def distance(A: City, B: City) -> float:
    "Compute the Euclidean distance between two cities."
    return abs(A - B)

def tour_length(tour: Tour) -> float:
    """Calculate the total distance of a tour.
IMP, GET THIS CHECKED: Includes the distance from the last city back to the first one."""
    return sum(distance(tour[i], tour[i - 1]) for i in range(len(tour)))

def valid_tour(tour: Tour, cities: Cities) -> bool:
    """Check if a given tour visits every city exactly once."""
    return Counter(tour) == Counter(cities)

def nearest_neighbor(A: City, cities) -> City:
    """Find the nearest city to a given city A from a set of unvisited cities."""
    return min(cities, key=lambda C: distance(C, A))

def nearest_tsp(cities, start=None) -> Tour:
    """Generate a tour using the nearest neighbour heuristic.
    Start at a specified city (or randomly if none is provided), 
    and repeatedly visit the nearest unvisited city."""
# ASK: Is start city defined?
    start = start or next(iter(cities))  # If no start city provided, pick one randomly.
    tour = [start]
    unvisited = set(cities) - {start}  # Set of cities yet to be visited.
    
    while unvisited:
        nearest = nearest_neighbor(tour[-1], unvisited)
        tour.append(nearest) 
        unvisited.remove(nearest)
    
    return tour

@functools.lru_cache(None)
def subsegments(N):
    """Generate all possible subsegments of a tour.
    Used for optimizing the tour by swapping segments."""
    return tuple((i, i + length)
                 for length in reversed(range(2, N - 1))
                 for i in range(N - length))

def opt2(tour) -> Tour:
    """Apply 2-opt optimization: Try reversing any two edges in the tour to find a shorter tour.
    Continues until no more improvements are possible."""
    changed = True
    while changed:
        changed = False
        for (i, j) in subsegments(len(tour)):  # Check all subsegments of the tour.
            if reversal_is_improvement(tour, i, j):  # If reversing the segment improves the tour.
                tour[i:j] = reversed(tour[i:j])  # Reverse the segment.
                changed = True  # Mark that a change has been made.
    return tour

def reversal_is_improvement(tour, i, j) -> bool:
    """Determine if reversing the segment between i and j will shorten the tour."""
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
    # Check if swapping would reduce the distance.
    return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)

def rep_opt2_nearest_tsp(cities, k=10) -> Tour:
    """Repeatedly apply 2-opt to the tours generated from the nearest neighbor heuristic.
    Tries k different starting cities."""
    return min((opt2(nearest_tsp(cities, start)) for start in random.sample(list(cities), min(k, len(cities)))),
               key=tour_length)

def brute_force_tsp(cities: List[City]) -> Tour:
    """Find the optimal tour by brute force.
    Check all possible permutations of the cities and return the one with the shortest distance."""
    best_tour = None
    best_length = float('inf')  # Initialize best length to a very large number.
    
    # Check every possible permutation of cities.
    for perm in itertools.permutations(cities):
        current_length = tour_length(perm)  # Calculate the length of the current permutation.
        if current_length < best_length:  # If the current tour is shorter than the best found so far.
            best_length = current_length
            best_tour = perm  # Update the best tour.
            
    return best_tour

def plot_tour(tour: Tour):
    """Plot the given tour using matplotlib.
    The starting city is marked in red, and the remaining cities are marked in blue."""
    plt.figure(figsize=(8, 6))
    X = [city.real for city in tour + [tour[0]]]  # Extract the real (x) coordinates.
    Y = [city.imag for city in tour + [tour[0]]]  # Extract the imaginary (y) coordinates.
    
    plt.plot(X, Y, 'bo-', label="Tour Path")  # Plot the tour with blue dots connected by lines.
    
    # Highlight the starting city (first city in the tour) in red
    plt.plot(tour[0].real, tour[0].imag, 'ro', markersize=10, label="Starting Point")  # Red dot for the start.
    
    plt.title('Optimal Tour Path')
    plt.axis('equal')  # Ensure equal scaling for both axes.
    plt.axis('off')  # Turn off the axes.
    plt.legend()  # Add a legend to differentiate the start point.
    plt.show()

def read_input():
    """Read input for the problem: TSP type, number of cities, and city coordinates.
    Also read the distance matrix if needed (for non-Euclidean TSP)."""
    tsp_type = input().strip()  # Read TSP type (EUCLIDEAN or NON-EUCLIDEAN).
    n = int(input())  # Read number of cities.
    cities = []
    
    # Read city coordinates.
    for _ in range(n):
        x, y = map(float, input().split())
        cities.append(complex(x, y))  # Store coordinates as complex numbers.
    
    distances = []
    
    # Read the distance matrix for non-Euclidean TSP.
    for _ in range(n):
        distances.append(list(map(float, input().split())))
    
    return tsp_type, cities, distances

def main():
    tsp_type, cities, distances = read_input()  # Read the input from the user.
    
    start_time = time.perf_counter()  # Start timing.
    best_tour_heuristic = rep_opt2_nearest_tsp(frozenset(cities))
    end_time_heuristic = time.perf_counter()  # End timing.
    
    best_length_heuristic = tour_length(best_tour_heuristic)  # Calculate the total length of the heuristic tour.
    
    # Print the results for the heuristic approach.
    print(f"Best Tour (Heuristic): {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")  # Print the path.
    print(f"Cost of Shortest Route (Heuristic): {best_length_heuristic:.2f}")  # Print the cost (length of the tour).

    print(f"Time Taken (Heuristic): {end_time_heuristic - start_time:.4f} seconds")  # Print the time taken.

    plot_tour(best_tour_heuristic)  # Plot the tour generated by the heuristic.
    
    # Uncomment the following section to use brute force (only feasible for small n).
    '''
    start_time_brute_force = time.perf_counter()  # Start timing.
    best_tour_brute_force = brute_force_tsp(cities)
    end_time_brute_force = time.perf_counter()  # End timing.
    
    best_length_brute_force = tour_length(best_tour_brute_force)
    
    print(f"Best Tour (Brute Force): {' '.join(str(cities.index(city)) for city in best_tour_brute_force)}")
    print(f"Cost of Shortest Route (Brute Force): {best_length_brute_force:.2f}")
    
    print(f"Time Taken (Brute Force): {end_time_brute_force - start_time_brute_force:.4f} seconds")
    '''


if __name__ == "__main__":
    main()