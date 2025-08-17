# # === envs/grid_uav_env.py ===
# import gymnasium as gym
# import numpy as np
# import utils.obstacles as obstacle_utils
# import utils.wind as wind_utils
# import cv2
# from gymnasium import spaces
# from utils.visualize import colorize_grid, draw_uav, draw_wind


# class GridUAVEnv(gym.Env):
#     """15x15 grid environment for UAV exploration"""
    
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.grid_size = config.grid_size
#         self.max_steps = config.max_steps
#         self.victim_count = config.victims
#         self.p_obs = config.p_obs
        
#         # Action space: two UAVs, each with 4 actions
#         self.action_space = spaces.MultiDiscrete([4, 4])
        
#         # Observation space
#         if config.obs_mode == "tabular":
#             self.observation_space = spaces.Box(
#                 low=0, high=1, 
#                 shape=(5*self.grid_size**2 + 8,), dtype=np.float32
#             )
#         else:  # CNN
#             self.observation_space = spaces.Dict({
#                 "image": spaces.Box(
#                     low=0, high=1, 
#                     shape=(5, self.grid_size, self.grid_size), 
#                     dtype=np.float32
#                 ),
#                 "vector": spaces.Box(
#                     low=-self.grid_size, high=self.grid_size,
#                     shape=(8,), dtype=np.float32
#                 )
#             })
        
#         # State variables
#         self.grid = None
#         self.known_obstacles = None
#         self.known_victims = None
#         self.visited_mask = None
#         self.victim_discovery_count = None
#         self.uav_positions = None
#         self.start_positions = None
#         self.wind_field = None
#         self.step_count = 0
#         self.episode_count = 0
#         self.total_free_cells = 0
        
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.episode_count += 1
        
#         # Generate grid
#         self.grid, self.known_obstacles = obstacle_utils.generate_grid(
#             self.grid_size, self.p_obs
#         )
        
#         # Place victims
#         self._place_victims()
#         self.known_victims = np.zeros_like(self.grid)
#         self.victim_discovery_count = np.zeros_like(self.grid)
        
#         # Initialize UAVs
#         self.start_positions = [(0, 0), (self.grid_size-1, self.grid_size-1)]
#         self.uav_positions = self.start_positions.copy()
        
#         # Initialize visited mask
#         self.visited_mask = np.zeros_like(self.grid)
#         for pos in self.uav_positions:
#             if self.grid[pos] == 0 or self.grid[pos] == 2:
#                 self.visited_mask[pos] = 1
        
#         # Generate wind field
#         self.wind_field = wind_utils.WindField(self.grid_size)
        
#         # Calculate total free cells
#         self.total_free_cells = np.sum(self.grid == 0) + np.sum(self.grid == 2)
        
#         # Update known area around UAVs
#         self._update_known_area()
        
#         return self._get_obs(), self._get_info()
    
#     def step(self, actions):
#         self.step_count += 1
#         rewards = 0
#         new_victims = 0
#         obstacle_hits = 0
#         uav_collision = 0
        
#         # Move each UAV
#         new_positions = []
#         for i, action in enumerate(actions):
#             # Get wind effect
#             wind_vec = self.wind_field.get_wind(self.uav_positions[i], self.step_count)
            
#             # Calculate new position
#             new_pos = self._calculate_new_position(self.uav_positions[i], action, wind_vec)
            
#             # Check collisions and update position
#             if self._is_valid_position(new_pos):
#                 # Check if UAVs collide
#                 if new_pos == self.uav_positions[1-i]:
#                     uav_collision = 1
#                     rewards -= 0.2
#                     # Stay in place
#                     new_pos = self.uav_positions[i]
#                 else:
#                     # Mark visited if unvisited
#                     if not self.visited_mask[new_pos] and (self.grid[new_pos] == 0 or self.grid[new_pos] == 2):
#                         rewards += 0.2
#                         self.visited_mask[new_pos] = 1
                    
#                     # Check for victim
#                     if self.grid[new_pos] == 2 and not self.known_victims[new_pos]:
#                         new_victims += 1
#                         rewards += 5
#                         self.known_victims[new_pos] = 1
#                         self.victim_discovery_count[new_pos] = 1
#             else:
#                 # Hit obstacle or boundary
#                 obstacle_hits += 1
#                 rewards -= 0.5
#                 new_pos = self.uav_positions[i]
            
#             new_positions.append(new_pos)
        
#         self.uav_positions = new_positions
        
#         # Step cost
#         rewards -= 0.01
        
#         # Update known obstacles around new positions
#         self._update_known_area()
        
#         # Check termination
#         terminated = self._is_fully_explored() or (self.step_count >= self.max_steps)
#         truncated = False
        
#         return self._get_obs(), rewards, terminated, truncated, self._get_info(
#             new_victims, obstacle_hits, uav_collision
#         )
    
#     def render(self, mode="human"):
#         img = colorize_grid(
#             self.grid, 
#             self.known_obstacles,
#             self.visited_mask,
#             self.known_victims
#         )
        
#         # Draw UAVs
#         for i, pos in enumerate(self.uav_positions):
#             color = (0, 255, 0) if i == 0 else (128, 0, 128)  # Green and purple
#             draw_uav(img, pos, color)
            
#             # Draw wind vector
#             wind_vec = self.wind_field.get_wind(pos, self.step_count)
#             draw_wind(img, pos, wind_vec)
        
#         return img
    
#     def _get_obs(self):
#         # Create observation channels
#         channels = [
#             self.visited_mask,
#             self.known_obstacles,
#             self.known_victims,
#             self._onehot_position(0),
#             self._onehot_position(1)
#         ]
        
#         # Get relative vectors to next waypoint (dummy in env)
#         rel_vec = np.zeros(8)  # Will be set by controller
        
#         if self.config.obs_mode == "tabular":
#             flat_obs = np.concatenate([c.flatten() for c in channels] + [rel_vec])
#             return flat_obs.astype(np.float32)
#         else:
#             return {
#                 "image": np.stack(channels).astype(np.float32),
#                 "vector": rel_vec.astype(np.float32)
#             }
    
#     def _get_info(self, new_victims=0, obstacle_hits=0, uav_collision=0):
#         return {
#             "new_victims": new_victims,
#             "obstacle_hits": obstacle_hits,
#             "uav_collision": uav_collision,
#             "coverage": np.sum(self.visited_mask) / self.total_free_cells
#         }
    
#     def _onehot_position(self, uav_idx):
#         onehot = np.zeros((self.grid_size, self.grid_size))
#         if 0 <= self.uav_positions[uav_idx][0] < self.grid_size and 0 <= self.uav_positions[uav_idx][1] < self.grid_size:
#             onehot[self.uav_positions[uav_idx]] = 1
#         return onehot
    
#     def _place_victims(self):
#         free_cells = np.argwhere((self.grid == 0))
#         if len(free_cells) < self.victim_count:
#             self.victim_count = len(free_cells)
        
#         victim_cells = free_cells[np.random.choice(
#             len(free_cells), self.victim_count, replace=False
#         )]
#         for cell in victim_cells:
#             self.grid[tuple(cell)] = 2
    
#     def _calculate_new_position(self, position, action, wind_vec):
#         moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
#         dx, dy = moves[action]
#         wx, wy = wind_vec
        
#         # Apply wind effect (stochastic)
#         if np.random.random() < 0.15:  # Gust probability
#             wx += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             wy += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
        
#         # Clip wind to [-1,1]
#         wx = max(-1, min(1, wx))
#         wy = max(-1, min(1, wy))
        
#         # Calculate new position with wind
#         new_x = max(0, min(self.grid_size-1, position[0] + dx + wx))
#         new_y = max(0, min(self.grid_size-1, position[1] + dy + wy))
#         return (int(new_x), int(new_y))
    
#     def _is_valid_position(self, pos):
#         return (
#             0 <= pos[0] < self.grid_size and 
#             0 <= pos[1] < self.grid_size and
#             self.grid[pos] != 1  # Not an obstacle
#         )
    
#     def _update_known_area(self):
#         # Reveal area around UAVs (Chebyshev distance 2)
#         for pos in self.uav_positions:
#             for dx in range(-2, 3):
#                 for dy in range(-2, 3):
#                     x, y = pos[0] + dx, pos[1] + dy
#                     if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
#                         if self.grid[x, y] == 1:
#                             self.known_obstacles[x, y] = 1
#                         elif self.grid[x, y] == 2 and self.known_victims[x, y]:
#                             self.known_victims[x, y] = 1
    
#     def _is_fully_explored(self):
#         return np.sum(self.visited_mask) >= self.total_free_cells






# === envs/grid_uav_env.py ===
import gymnasium as gym
import numpy as np
import utils.obstacles as obstacle_utils
import utils.wind as wind_utils
import cv2
from gymnasium import spaces
from utils.visualize import colorize_grid, draw_uav, draw_wind, draw_wind_field, draw_info_panel


class GridUAVEnv(gym.Env):
    """15x15 grid environment for UAV exploration"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_size = config.grid_size
        self.max_steps = config.max_steps
        self.victim_count = config.victims
        self.p_obs = config.p_obs
        
        # Action space: two UAVs, each with 4 actions
        self.action_space = spaces.MultiDiscrete([4, 4])
        
        # Observation space
        if config.obs_mode == "tabular":
            self.observation_space = spaces.Box(
                low=0, high=1, 
                shape=(5*self.grid_size**2 + 8,), dtype=np.float32
            )
        else:  # CNN
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=0, high=1, 
                    shape=(5, self.grid_size, self.grid_size), 
                    dtype=np.float32
                ),
                "vector": spaces.Box(
                    low=-self.grid_size, high=self.grid_size,
                    shape=(8,), dtype=np.float32
                )
            })
        
        # State variables
        self.grid = None
        self.known_obstacles = None
        self.known_victims = None
        self.visited_mask = None
        self.victim_discovery_count = None
        self.uav_positions = None
        self.start_positions = None
        self.wind_field = None
        self.step_count = 0
        self.episode_count = 0
        self.total_free_cells = 0
        self.wind_grid = None  # Store wind vector for each cell
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        
        # Generate grid
        self.grid, self.known_obstacles = obstacle_utils.generate_grid(
            self.grid_size, self.p_obs
        )
        
        # Place victims
        self._place_victims()
        self.known_victims = np.zeros_like(self.grid)
        self.victim_discovery_count = np.zeros_like(self.grid)
        
        # Initialize UAVs
        self.start_positions = [(0, 0), (self.grid_size-1, self.grid_size-1)]
        self.uav_positions = self.start_positions.copy()
        
        # Initialize visited mask
        self.visited_mask = np.zeros_like(self.grid)
        for pos in self.uav_positions:
            if self.grid[pos] == 0 or self.grid[pos] == 2:
                self.visited_mask[pos] = 1
        
        # Generate wind field
        self.wind_field = wind_utils.WindField(self.grid_size)
        self.wind_grid = self.wind_field.get_full_wind_grid()
        
        # Calculate total free cells
        self.total_free_cells = np.sum(self.grid == 0) + np.sum(self.grid == 2)
        
        # Update known area around UAVs
        self._update_known_area()
        
        return self._get_obs(), self._get_info()
    
    def step(self, actions):
        self.step_count += 1
        rewards = 0
        new_victims = 0
        obstacle_hits = 0
        uav_collision = 0
        wind_effects = [0, 0]  # Track wind effect for each UAV
        
        # Move each UAV
        new_positions = []
        for i, action in enumerate(actions):
            # Get wind effect (base + gust)
            wind_vec = self.wind_field.get_wind(self.uav_positions[i], self.step_count)
            
            # Calculate new position
            new_pos = self._calculate_new_position(self.uav_positions[i], action, wind_vec)
            
            # Track wind effect magnitude
            wind_effects[i] = np.linalg.norm(wind_vec)
            
            # Check collisions and update position
            if self._is_valid_position(new_pos):
                # Check if UAVs collide
                if new_pos == self.uav_positions[1-i]:
                    uav_collision = 1
                    rewards -= 0.2
                    # Stay in place
                    new_pos = self.uav_positions[i]
                else:
                    # Mark visited if unvisited
                    if not self.visited_mask[new_pos] and (self.grid[new_pos] == 0 or self.grid[new_pos] == 2):
                        rewards += 0.2
                        self.visited_mask[new_pos] = 1
                    
                    # Check for victim
                    if self.grid[new_pos] == 2 and not self.known_victims[new_pos]:
                        new_victims += 1
                        rewards += 5
                        self.known_victims[new_pos] = 1
                        self.victim_discovery_count[new_pos] = 1
            else:
                # Hit obstacle or boundary
                obstacle_hits += 1
                rewards -= 0.5
                new_pos = self.uav_positions[i]
            
            new_positions.append(new_pos)
        
        self.uav_positions = new_positions
        
        # Step cost
        rewards -= 0.01
        
        # Update known obstacles around new positions
        self._update_known_area()
        
        # Check termination
        terminated = self._is_fully_explored() or (self.step_count >= self.max_steps)
        truncated = False
        
        return self._get_obs(), rewards, terminated, truncated, self._get_info(
            new_victims, obstacle_hits, uav_collision, wind_effects
        )
    
    def render(self, mode="human"):
        # Create base grid image
        img = colorize_grid(
            self.grid, 
            self.known_obstacles,
            self.visited_mask,
            self.known_victims
        )
        
        # Draw wind field
        img = draw_wind_field(img, self.wind_grid)
        
        # Draw UAVs
        for i, pos in enumerate(self.uav_positions):
            color = (0, 255, 0) if i == 0 else (128, 0, 128)  # Green and purple
            draw_uav(img, pos, color)
            
            # Draw wind vector at UAV position
            wind_vec = self.wind_field.get_wind(pos, self.step_count)
            draw_wind(img, pos, wind_vec)
        
        # Add info panel
        img = draw_info_panel(img, self.step_count, self.victim_count, 
                              np.sum(self.known_victims), 
                              np.sum(self.visited_mask), self.total_free_cells)
        
        return img
    
    def _get_obs(self):
        # Create observation channels
        channels = [
            self.visited_mask,
            self.known_obstacles,
            self.known_victims,
            self._onehot_position(0),
            self._onehot_position(1)
        ]
        
        # Get relative vectors to next waypoint (dummy in env)
        rel_vec = np.zeros(8)  # Will be set by controller
        
        if self.config.obs_mode == "tabular":
            flat_obs = np.concatenate([c.flatten() for c in channels] + [rel_vec])
            return flat_obs.astype(np.float32)
        else:
            return {
                "image": np.stack(channels).astype(np.float32),
                "vector": rel_vec.astype(np.float32)
            }
    
    def _get_info(self, new_victims=0, obstacle_hits=0, uav_collision=0, wind_effects=None):
        return {
            "new_victims": new_victims,
            "obstacle_hits": obstacle_hits,
            "uav_collision": uav_collision,
            "wind_effects": wind_effects or [0, 0],
            "coverage": np.sum(self.visited_mask) / self.total_free_cells
        }
    
    def _onehot_position(self, uav_idx):
        onehot = np.zeros((self.grid_size, self.grid_size))
        if 0 <= self.uav_positions[uav_idx][0] < self.grid_size and 0 <= self.uav_positions[uav_idx][1] < self.grid_size:
            onehot[self.uav_positions[uav_idx]] = 1
        return onehot
    
    def _place_victims(self):
        free_cells = np.argwhere((self.grid == 0))
        if len(free_cells) < self.victim_count:
            self.victim_count = len(free_cells)
        
        victim_cells = free_cells[np.random.choice(
            len(free_cells), self.victim_count, replace=False
        )]
        for cell in victim_cells:
            self.grid[tuple(cell)] = 2
    
    def _calculate_new_position(self, position, action, wind_vec):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        dx, dy = moves[action]
        wx, wy = wind_vec
        
        # Apply wind effect (stochastic)
        if np.random.random() < 0.15:  # Gust probability
            wx += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
            wy += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
        
        # Clip wind to [-1,1]
        wx = max(-1, min(1, wx))
        wy = max(-1, min(1, wy))
        
        # Calculate new position with wind
        new_x = max(0, min(self.grid_size-1, position[0] + dx + wx))
        new_y = max(0, min(self.grid_size-1, position[1] + dy + wy))
        return (int(new_x), int(new_y))
    
    def _is_valid_position(self, pos):
        return (
            0 <= pos[0] < self.grid_size and 
            0 <= pos[1] < self.grid_size and
            self.grid[pos] != 1  # Not an obstacle
        )
    
    def _update_known_area(self):
        # Reveal area around UAVs (Chebyshev distance 2)
        for pos in self.uav_positions:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = pos[0] + dx, pos[1] + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        if self.grid[x, y] == 1:
                            self.known_obstacles[x, y] = 1
                        elif self.grid[x, y] == 2 and self.known_victims[x, y]:
                            self.known_victims[x, y] = 1
    
    def _is_fully_explored(self):
        return np.sum(self.visited_mask) >= self.total_free_cells