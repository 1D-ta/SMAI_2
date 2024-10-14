#hardcoded, doing euclidean work for both fuck it 

# Imports
import random
import time
import numpy as np
from typing import List, Tuple
#import matplotlib.pyplot as plt
import itertools

City = complex  # Represent city coordinates as complex numbers
Cities = frozenset  # A set of cities
Tour = List[City]  # A list of cities visited in order

random.seed(42)  # Ensure consistency across runs

# Global dictionary to map city coordinates to indices
city_to_index = {}
#DISTANCE_MATRIX = None


def distance(A: City, B: City) -> float:
    """Compute Euclidean distance between two cities."""
    global DISTANCE_MATRIX
    return DISTANCE_MATRIX[city_to_index[A], city_to_index[B]]

def tour_length(tour: Tour) -> float:
    """Calculate total length of the tour, including the link from last city back to the first."""
    return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))

def nearest_neighbor(A: City, cities: Cities) -> City:
    """Find the nearest city to city A using Euclidean distance."""
    return min(cities, key=lambda C: distance(A, C))

def nearest_tsp(cities: Cities, start: City = None) -> Tour:
    """Generate a tour using the nearest neighbor heuristic."""
    start = start or next(iter(cities))
    tour = [start]
    unvisited = set(cities) - {start}
    
    while unvisited:
        nearest = nearest_neighbor(tour[-1], unvisited)
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour

def opt2(tour: Tour) -> Tour:
    """Perform 2-opt optimization to improve the tour length."""
    changed = True
    while changed:
        changed = False
        for (i, j) in subsegments(len(tour)):
            if reversal_is_improvement(tour, i, j):
                tour[i:j] = reversed(tour[i:j])
                changed = True
    return tour

def subsegments(N):
    """Generate subsegments for 2-opt optimization."""
    return tuple((i, i + length) for length in reversed(range(2, N - 1)) for i in range(N - length))

def reversal_is_improvement(tour: Tour, i: int, j: int) -> bool:
    """Check if reversing a segment improves the tour length."""
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
    return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)

def rep_opt2_nearest_tsp(cities: Cities, k=200) -> Tour:
    """Apply nearest neighbor with 2-opt optimization over multiple initializations."""
    return min((opt2(nearest_tsp(cities, start)) for start in random.sample(list(cities), min(k, len(cities)))), 
               key=tour_length)

def read_input():

    """Read input for TSP problem and handle both Euclidean and non-Euclidean cases."""
    tsp_type = input().strip()  # Either "EUCLIDEAN" or "NON-EUCLIDEAN"
    n = int(input().strip())  # Number of cities 
    
    cities = [complex(*map(float, input().split())) for _ in range(n)] 
    global city_to_index
    city_to_index = {city: idx for idx, city in enumerate(cities)}  # Add this line
    distances = [list(map(float, input().split())) for _ in range(n)]
    return tsp_type, cities, distances

'''
def plot_tour(tour: Tour):
    """Plot the cities and the tour path."""
    x = [city.real for city in tour]  # Extract real parts (x-coordinates)
    y = [city.imag for city in tour]  # Extract imaginary parts (y-coordinates)

    #print(tour)
    plt.figure(figsize=(10, 6))
    plt.plot(x + [tour[0].real], y + [tour[0].imag], marker='o', linestyle='-', color='b')
    
    # Annotate each city with its index
    for i, city in enumerate(tour):
        plt.annotate(f'{city_to_index[city]}', (city.real, city.imag), textcoords="offset points", xytext=(0,10), ha='center')


    plt.title('TSP Tour Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.axis('equal')  # Equal scaling for both axes
    plt.show()
'''

def brute_force(DISTANCE_MATRIX):
    """Brute-force search to find the optimal tour using the distance matrix and print in terms of city indices."""
    
    # Reverse mapping from indices back to cities
    index_to_city = {idx: city for city, idx in city_to_index.items()}

    # Generate all possible tours
    best_length = float('inf')
    best_tour = None

    for tour in itertools.permutations(range(len(DISTANCE_MATRIX))):
        current_length = sum(DISTANCE_MATRIX[tour[i], tour[i-1]] for i in range(len(tour)))
        if current_length < best_length:
            best_length = current_length
            best_tour = tour

    # Convert the best_tour indices back to global city indices and print
    best_tour_cities = [index_to_city[idx] for idx in best_tour]
    best_tour_global_indices = [city_to_index[city] for city in best_tour_cities]
    
    # Output results
    print(f"Best Tour: {best_tour_global_indices}")
    print(f"Cost of Route: {best_length:.2f}")


def main():
    tsp_type, cities, distances = read_input()

    global DISTANCE_MATRIX 
    DISTANCE_MATRIX = np.array(distances)  # Set global distance matrix for non-Euclidean TSP

    start_time = time.perf_counter()
    best_tour_heuristic = rep_opt2_nearest_tsp(frozenset(cities))
    best_length = tour_length(best_tour_heuristic)

    tour_heuristic = nearest_tsp(frozenset(cities))
    length = tour_length(tour_heuristic)

    end_time = time.perf_counter()
    
    #print with opt
    #print(f"Tour: {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")
    print(f"Cost of Route: {best_length:.2f}")
    #print(f"Time Taken: {end_time - start_time:.4f} seconds")
    
    #print without opt
    #print(f"Tour_without opt: {' '.join(str(cities.index(city)) for city in best_tour_heuristic)}")
    print(f"Cost of Route_without opt: {length:.2f}")

    #plot_tour(best_tour_heuristic)

if __name__ == "__main__":
    main()  # Run the main function if the script is executed