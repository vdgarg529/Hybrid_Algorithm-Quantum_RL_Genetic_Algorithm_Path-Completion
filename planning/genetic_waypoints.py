# === planning/genetic_waypoints.py ===
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chromosome:
    waypoints: List[Tuple[int, int]]
    fitness: float = -np.inf


class GeneticWaypointsPlanner:
    def __init__(self, config, grid_size):
        self.config = config
        self.grid_size = grid_size
        self.waypoints_per_uav = config.waypoints_per_uav
        self.pop_size = config.ga_pop
        self.max_iters = config.ga_iters
        self.elite_size = max(1, int(0.05 * config.ga_pop))
        self.known_obstacles = None
        self.visited_mask = None
    
    def run_ga(self, known_obstacles, visited_mask, start_positions):
        self.known_obstacles = known_obstacles
        self.visited_mask = visited_mask
        
        # Initialize population
        population = self._initialize_population()
        
        for gen in range(self.max_iters):
            # Evaluate fitness
            for chrom in population:
                chrom.fitness = self._fitness(chrom)
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Select parents (tournament selection)
            parents = []
            for _ in range(self.pop_size - self.elite_size):
                # FIXED: Removed extra parenthesis
                tournament = random.sample(population, min(3, len(population)))
                winner = max(tournament, key=lambda x: x.fitness)
                parents.append(winner)
            
            # Create next generation
            next_gen = population[:self.elite_size]  # Elitism
            
            # Crossover and mutation
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else parent1
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                next_gen.extend([child1, child2])
            
            population = next_gen[:self.pop_size]
        
        # Return best solution
        best = max(population, key=lambda x: x.fitness)
        waypoints0 = best.waypoints[:self.waypoints_per_uav]
        waypoints1 = best.waypoints[self.waypoints_per_uav:]
        return waypoints0, waypoints1
    
    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            waypoints = []
            # UAV0 waypoints
            for _ in range(self.waypoints_per_uav):
                waypoints.append(self._random_free_position())
            # UAV1 waypoints
            for _ in range(self.waypoints_per_uav):
                waypoints.append(self._random_free_position())
            population.append(Chromosome(waypoints))
        return population
    
    def _random_free_position(self):
        # Get all known free positions
        free_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.known_obstacles[x, y] == 0:
                    free_positions.append((x, y))
        
        if free_positions:
            return random.choice(free_positions)
        return (0, 0)  # Fallback
    
    def _fitness(self, chrom):
        # Coverage potential (unique cells in paths)
        coverage = 0
        for path in [chrom.waypoints[:self.waypoints_per_uav], 
                     chrom.waypoints[self.waypoints_per_uav:]]:
            for wp in path:
                if not self.visited_mask[wp]:
                    coverage += 1
        
        # Dispersion (distance between waypoints)
        dispersion = 0
        for path in [chrom.waypoints[:self.waypoints_per_uav], 
                     chrom.waypoints[self.waypoints_per_uav:]]:
            for i in range(1, len(path)):
                # FIXED: Added missing parenthesis
                dispersion += np.linalg.norm(
                    np.array(path[i]) - np.array(path[i-1])
                )
        
        # Victim likelihood (prioritize unexplored areas)
        victim_likelihood = 0
        for wp in chrom.waypoints:
            if not self.visited_mask[wp]:
                victim_likelihood += 1
        
        # Weighted fitness
        return 0.7 * coverage + 0.2 * dispersion + 0.1 * victim_likelihood
    
    def _crossover(self, parent1, parent2):
        # One-point crossover
        pt = np.random.randint(1, len(parent1.waypoints)-1)
        child1 = Chromosome(parent1.waypoints[:pt] + parent2.waypoints[pt:])
        child2 = Chromosome(parent2.waypoints[:pt] + parent1.waypoints[pt:])
        return child1, child2
    
    def _mutate(self, chrom):
        # Random waypoint jitter
        mutated = Chromosome(chrom.waypoints.copy())
        idx = np.random.randint(len(mutated.waypoints))
        mutated.waypoints[idx] = self._random_free_position()
        return mutated