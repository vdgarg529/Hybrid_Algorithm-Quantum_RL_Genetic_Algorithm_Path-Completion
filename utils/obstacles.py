# === utils/obstacles.py ===
import numpy as np
from typing import Tuple


def generate_grid(grid_size: int, p_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate grid with obstacles ensuring connectivity"""
    while True:
        obstacles = np.random.choice(
            [0, 1], size=(grid_size, grid_size), p=[1-p_obs, p_obs]
        )
        
        # Ensure start positions are free
        obstacles[0, 0] = 0
        obstacles[grid_size-1, grid_size-1] = 0
        
        # Create grid with obstacles
        grid = np.zeros((grid_size, grid_size))
        grid[obstacles == 1] = 1
        
        # Check connectivity
        if is_fully_connected(obstacles, grid_size):
            return grid, obstacles


def is_fully_connected(obstacles: np.ndarray, grid_size: int) -> bool:
    """Check if at least 90% of free cells are reachable"""
    free_cells = grid_size**2 - np.sum(obstacles)
    if free_cells == 0:
        return False
    
    start = (0, 0)
    if obstacles[start]:
        return False
    
    # BFS to count reachable cells
    queue = [start]
    visited = set([start])
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        x, y = queue.pop(0)
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                    obstacles[nx, ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return len(visited) >= 0.9 * free_cells