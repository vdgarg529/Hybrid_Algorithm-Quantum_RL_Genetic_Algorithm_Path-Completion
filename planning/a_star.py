# === planning/a_star.py ===
import heapq
import numpy as np
from typing import List, Tuple, Optional


class AStarPlanner:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.paths = {}
    
    def get_action(self, current_pos, visited_mask, known_obstacles):
        # If no path or path is invalid, replan
        if current_pos not in self.paths or not self.paths.get(current_pos):
            target = self._find_nearest_unvisited(current_pos, visited_mask, known_obstacles)
            if target:
                self.paths[current_pos] = self._a_star(current_pos, target, known_obstacles)
            else:
                return 0  # No target found
        
        # Follow path if available
        if self.paths.get(current_pos):
            return self.paths[current_pos].pop(0)
        
        return 0  # Default to no-op
    
    def _find_nearest_unvisited(self, start, visited_mask, known_obstacles):
        # BFS for nearest unvisited cell
        queue = [start]
        visited = set([start])
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while queue:
            pos = queue.pop(0)
            if not visited_mask[pos] and known_obstacles[pos] == 0:
                return pos
            
            for dx, dy in moves:
                nx, ny = pos[0] + dx, pos[1] + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    (nx, ny) not in visited and known_obstacles[nx, ny] == 0):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return None
    
    def _a_star(self, start, target, obstacles):
        # Standard A* with Manhattan distance
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, target)}
        
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Correspond to actions 0-3
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == target:
                return self._reconstruct_path(came_from, current, moves)
            
            for action, (dx, dy) in enumerate(moves):
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.grid_size and 
                        0 <= neighbor[1] < self.grid_size):
                    continue
                if obstacles[neighbor]:
                    continue
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(self, came_from, current, moves):
        path = []
        while current in came_from:
            current, action = came_from[current]
            path.append(action)
        return list(reversed(path))