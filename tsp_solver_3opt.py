import random
import time
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

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

def nearest_neighbor_tour(cities: Cities) -> Tour:
    unvisited = set(cities)
    tour = [random.choice(list(unvisited))]
    unvisited.remove(tour[0])
    while unvisited:
        nearest = min(unvisited, key=lambda city: distance(tour[-1], city))
        tour.append(nearest)
        unvisited.remove(nearest)
    return tour

def two_opt_swap(tour: Tour, i: int, j: int) -> Tour:
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def two_opt(tour: Tour, max_iterations: int = 1000) -> Tour:
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue
                new_tour = two_opt_swap(tour, i, j)
                if tour_length(new_tour) < tour_length(tour):
                    tour = new_tour
                    improved = True
                    break
            if improved: break
        iterations += 1
    return tour

def three_opt_swap(tour: Tour, i: int, j: int, k: int) -> Tour:
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j]
    E, F = tour[k-1], tour[k % len(tour)]
    d0 = distance(A, B) + distance(C, D) + distance(E, F)
    d1 = distance(A, C) + distance(B, D) + distance(E, F)
    d2 = distance(A, B) + distance(C, E) + distance(D, F)
    d3 = distance(A, D) + distance(E, B) + distance(C, F)
    d4 = distance(F, B) + distance(C, D) + distance(E, A)

    if d0 > d1:
        tour[i:j] = reversed(tour[i:j])
    elif d0 > d2:
        tour[j:k] = reversed(tour[j:k])
    elif d0 > d4:
        tour[i:k] = reversed(tour[i:k])
    elif d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
    return tour

def three_opt(tour: Tour, max_iterations: int = 100) -> Tour:
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        for i in range(1, len(tour) - 3):
            for j in range(i + 2, len(tour) - 1):
                for k in range(j + 2, len(tour) + (i > 0)):
                    new_tour = three_opt_swap(tour, i, j, k)
                    if tour_length(new_tour) < tour_length(tour):
                        tour = new_tour
                        improved = True
                        break
                if improved: break
            if improved: break
        iterations += 1
    return tour

def iterated_local_search(cities: Cities, max_iterations: int = 10) -> Tour:
    best_tour = nearest_neighbor_tour(cities)
    best_tour = two_opt(best_tour)
    best_length = tour_length(best_tour)

    for _ in range(max_iterations):
        perturbed_tour = best_tour[:]
        # Double-bridge move
        n = len(perturbed_tour)
        i, j, k = sorted(random.sample(range(1, n), 3))
        perturbed_tour[i:k] = perturbed_tour[j:k] + perturbed_tour[i:j]
        
        improved_tour = two_opt(perturbed_tour)
        improved_tour = three_opt(improved_tour, max_iterations=10)
        
        current_length = tour_length(improved_tour)
        if current_length < best_length:
            best_tour = improved_tour
            best_length = current_length

    return best_tour

def main():
    tsp_type, cities, distances = read_input()
    global DISTANCE_MATRIX 
    DISTANCE_MATRIX = np.array(distances)

    start_time = time.perf_counter()
    best_tour = iterated_local_search(frozenset(cities))
    best_length = tour_length(best_tour)
    end_time = time.perf_counter()
    
    print(f"Cost of Route: {best_length:.2f}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")

    plot_tour(best_tour)

if __name__ == "__main__":
    main()