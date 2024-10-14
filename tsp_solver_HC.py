# Hardcoded, doing Euclidean work for both, keeping it simple

# Imports
import random
import time
import numpy as np
from typing import List, Tuple

# Type alias for City using complex numbers to represent coordinates (x, y)
City = complex  
# Type alias for Cities, which is a set of City objects
Cities = frozenset  
# Type alias for a Tour, which is a list of cities visited in order
Tour = List[City]  

# Ensure random number generator consistency across runs by setting a seed
random.seed(42)

def distance(A: City, B: City) -> float:
    """Compute Euclidean distance between two cities."""
    return abs(A - B)  # Euclidean distance is the modulus of the complex number difference

def tour_length(tour: Tour) -> float:
    """Calculate total length of the tour, including the link from last city back to the first."""
    return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))  # Sum distances between consecutive cities

def nearest_neighbor(A: City, cities: Cities) -> City:
    """Find the nearest city to city A using Euclidean distance."""
    return min(cities, key=lambda C: distance(A, C))  # Find the city with the smallest distance to A

def nearest_tsp(cities: Cities, start: City = None) -> Tour:
    """Generate a tour using the nearest neighbor heuristic."""
    start = start or next(iter(cities))  # If no start city is provided, pick an arbitrary one
    tour = [start]  # Initialize the tour starting with the chosen city
    unvisited = set(cities) - {start}  # Track unvisited cities, excluding the start city

    while unvisited:
        nearest = nearest_neighbor(tour[-1], unvisited)  # Find the nearest unvisited city
        tour.append(nearest)  # Add the nearest city to the tour
        unvisited.remove(nearest)  # Mark the city as visited by removing it from the unvisited set

    return tour  # Return the completed tour

def opt2(tour: Tour) -> Tour:
    """Perform 2-opt optimization to improve the tour length."""
    changed = True  # Initialize the change flag
    while changed:
        changed = False  # Assume no change until proven otherwise
        for (i, j) in subsegments(len(tour)):  # Loop over all subsegments of the tour
            if reversal_is_improvement(tour, i, j):  # Check if reversing this segment improves the tour
                tour[i:j] = reversed(tour[i:j])  # If so, reverse the segment
                changed = True  # Flag that a change has occurred
    return tour  # Return the optimized tour

def subsegments(N):
    """Generate subsegments for 2-opt optimization."""
    # Create pairs of indices (i, i + length) for reversing segments
    return tuple((i, i + length) for length in reversed(range(2, N - 1)) for i in range(N - length))

def reversal_is_improvement(tour: Tour, i: int, j: int) -> bool:
    """Check if reversing a segment improves the tour length."""
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]  # Identify the four cities involved in the reversal
    return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)  # Check if the reversal shortens the tour

def rep_opt2_nearest_tsp(cities: Cities, k=200) -> Tour:
    """Apply nearest neighbor with 2-opt optimization over multiple initializations."""
    # Perform nearest neighbor heuristic starting from k random cities, optimize with 2-opt, return the best tour
    return min((opt2(nearest_tsp(cities, start)) for start in random.sample(list(cities), min(k, len(cities)))), 
               key=tour_length)

def read_input():
    """Read input for TSP problem and handle both Euclidean and non-Euclidean cases."""
    tsp_type = input().strip()  # Either "EUCLIDEAN" or "NON-EUCLIDEAN", read from input
    n = int(input())  # Number of cities, read from input
    cities = [complex(*map(float, input().split())) for _ in range(n)]  # Parse city coordinates as complex numbers
    distances = [list(map(float, input().split())) for _ in range(n)] if tsp_type == "NON-EUCLIDEAN" else None
    return tsp_type, cities, distances  # Return the problem type, city coordinates, and distance matrix if applicable

def main():
    tsp_type, cities, distances = read_input()  # Read input for TSP problem

    if tsp_type == "NON-EUCLIDEAN":
        global DISTANCE_MATRIX
        DISTANCE_MATRIX = np.array(distances)  # Set global distance matrix for non-Euclidean TSP

    start_time = time.perf_counter()  # Start timer for performance measurement
    best_tour_heuristic = rep_opt2_nearest_tsp(frozenset(cities))  # Find best tour using nearest neighbor and 2-opt
    best_length = tour_length(best_tour_heuristic)  # Calculate the length of the best tour
    end_time = time.perf_counter()  # End timer for performance measurement
    
    # Output the results
    
    print(f"Tour: {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")
    print(f"Cost of Route: {best_length:.2f}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    

    #print(f"{' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")


if __name__ == "__main__":
    main()  # Run the main function if the script is executed
