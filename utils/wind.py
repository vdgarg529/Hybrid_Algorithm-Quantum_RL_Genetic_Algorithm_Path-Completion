# # === utils/wind.py ===
# import numpy as np


# class WindField:
#     def __init__(self, grid_size, macro_size=5):
#         self.grid_size = grid_size
#         self.macro_size = macro_size
#         self.base_field = self._generate_base_field()
    
#     def _generate_base_field(self):
#         # Create macro-cells
#         macro_x = (self.grid_size + self.macro_size - 1) // self.macro_size
#         macro_y = (self.grid_size + self.macro_size - 1) // self.macro_size
#         field = np.zeros((macro_x, macro_y, 2), dtype=np.int8)
        
#         for i in range(macro_x):
#             for j in range(macro_y):
#                 # Sample wind from {-1,0,1} with bias
#                 field[i, j, 0] = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#                 field[i, j, 1] = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#         return field
    
#     def get_wind(self, position, step_count):
#         # Get macro-cell index
#         i = position[0] // self.macro_size
#         j = position[1] // self.macro_size
        
#         # Base wind
#         wx, wy = self.base_field[i, j]
        
#         # Gust noise (15% chance)
#         if np.random.random() < 0.15:
#             wx += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             wy += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
        
#         # Clip to [-1,1]
#         wx = max(-1, min(1, wx))
#         wy = max(-1, min(1, wy))
#         return (wx, wy)




# === utils/wind.py ===
import numpy as np


class WindField:
    def __init__(self, grid_size, macro_size=5):
        self.grid_size = grid_size
        self.macro_size = macro_size
        self.base_field = self._generate_base_field()
    
    def _generate_base_field(self):
        # Create macro-cells
        macro_x = (self.grid_size + self.macro_size - 1) // self.macro_size
        macro_y = (self.grid_size + self.macro_size - 1) // self.macro_size
        field = np.zeros((macro_x, macro_y, 2), dtype=np.float32)
        
        for i in range(macro_x):
            for j in range(macro_y):
                # Sample wind from {-1,0,1} with bias
                field[i, j, 0] = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
                field[i, j, 1] = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
        return field
    
    def get_full_wind_grid(self):
        """Create a grid where each cell has its base wind vector"""
        wind_grid = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get macro-cell index
                macro_i = i // self.macro_size
                macro_j = j // self.macro_size
                if macro_i < self.base_field.shape[0] and macro_j < self.base_field.shape[1]:
                    wind_grid[i, j] = self.base_field[macro_i, macro_j]
        return wind_grid
    
    def get_wind(self, position, step_count):
        # Get macro-cell index
        macro_i = position[0] // self.macro_size
        macro_j = position[1] // self.macro_size
        
        if (macro_i >= self.base_field.shape[0] or 
            macro_j >= self.base_field.shape[1]):
            return (0, 0)
        
        # Base wind
        wx, wy = self.base_field[macro_i, macro_j]
        
        # Gust noise (15% chance)
        if np.random.random() < 0.15:
            wx += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
            wy += np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
        
        # Clip to [-1,1]
        wx = max(-1, min(1, wx))
        wy = max(-1, min(1, wy))
        return (wx, wy)