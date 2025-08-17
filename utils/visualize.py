# # === utils/visualize.py ===
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Tuple


# # Cell size for visualization (pixels per grid cell)
# CELL_SIZE = 40

# def colorize_grid(
#     grid: np.ndarray, 
#     known_obstacles: np.ndarray,
#     visited_mask: np.ndarray,
#     known_victims: np.ndarray
# ) -> np.ndarray:
#     """Create RGB visualization of grid state"""
#     h, w = grid.shape
#     # Create a larger image for better visibility
#     img = np.ones((h * CELL_SIZE, w * CELL_SIZE, 3), dtype=np.uint8) * 200
    
#     # Draw grid cells
#     for x in range(h):
#         for y in range(w):
#             # Calculate cell position in the scaled image
#             x1 = x * CELL_SIZE
#             y1 = y * CELL_SIZE
#             x2 = (x + 1) * CELL_SIZE
#             y2 = (y + 1) * CELL_SIZE
            
#             # Determine cell color
#             if known_obstacles[x, y] == 1:
#                 color = (50, 50, 50)  # Dark gray for obstacles
#             elif visited_mask[x, y] == 1:
#                 color = (200, 200, 255)  # Light blue for visited
#             else:
#                 color = (200, 200, 200)  # Light gray for unvisited
                
#             # Draw the cell
#             cv2.rectangle(img, (y1, x1), (y2, x2), color, -1)
            
#             # Draw grid lines
#             cv2.rectangle(img, (y1, x1), (y2, x2), (150, 150, 150), 1)
            
#             # Draw victims
#             if known_victims[x, y] == 1:
#                 center = (y1 + CELL_SIZE // 2, x1 + CELL_SIZE // 2)
#                 cv2.circle(img, center, CELL_SIZE // 4, (0, 0, 255), -1)
    
#     return img


# def draw_uav(img: np.ndarray, position: Tuple[int, int], color: Tuple[int, int, int]):
#     """Draw UAV as a colored triangle"""
#     x, y = position
#     # Calculate center position in scaled image
#     cx = int((x + 0.5) * CELL_SIZE)
#     cy = int((y + 0.5) * CELL_SIZE)
    
#     # Create triangle points
#     pts = np.array([[
#         (cy, cx - CELL_SIZE // 3),          # Top
#         (cy - CELL_SIZE // 3, cx + CELL_SIZE // 4),  # Bottom-left
#         (cy + CELL_SIZE // 3, cx + CELL_SIZE // 4)   # Bottom-right
#     ]])
#     cv2.fillPoly(img, pts, color)
    
#     # Draw outline for better visibility
#     cv2.polylines(img, pts, True, (0, 0, 0), 1)


# def draw_wind(img: np.ndarray, position: Tuple[int, int], wind_vec: Tuple[int, int]):
#     """Draw wind vector as arrow"""
#     x, y = position
#     wx, wy = wind_vec
#     if wx != 0 or wy != 0:
#         # Calculate center position in scaled image
#         cx = int((x + 0.5) * CELL_SIZE)
#         cy = int((y + 0.5) * CELL_SIZE)
        
#         # Calculate end point
#         end_x = cx + wx * CELL_SIZE // 2
#         end_y = cy + wy * CELL_SIZE // 2
        
#         # Draw arrow
#         cv2.arrowedLine(img, (cy, cx), (end_y, end_x), (0, 0, 0), 2, tipLength=0.3)


# def save_frame(img: np.ndarray, frame_num: int):
#     """Save frame to output directory"""
#     cv2.imwrite(f"outputs/frames/frame_{frame_num:04d}.png", img)


# def generate_heatmap(victim_counts: np.ndarray, path: str):
#     """Generate victim discovery heatmap"""
#     plt.figure(figsize=(10, 8))
#     plt.imshow(victim_counts, cmap="YlGn", vmin=0, vmax=10)
    
#     # Add grid lines
#     plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
#     plt.xticks(np.arange(-0.5, victim_counts.shape[1], 1), minor=True)
#     plt.yticks(np.arange(-0.5, victim_counts.shape[0], 1), minor=True)
    
#     plt.colorbar(label="Victim Discovery Count")
#     plt.title("Victim Discovery Heatmap")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.tight_layout()
#     plt.savefig(path, dpi=150)
#     plt.close()




# === utils/visualize.py ===
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# Cell size for visualization (pixels per grid cell)
CELL_SIZE = 50

def colorize_grid(
    grid: np.ndarray, 
    known_obstacles: np.ndarray,
    visited_mask: np.ndarray,
    known_victims: np.ndarray
) -> np.ndarray:
    """Create RGB visualization of grid state"""
    h, w = grid.shape
    # Create a larger image for better visibility
    img = np.ones((h * CELL_SIZE, w * CELL_SIZE, 3), dtype=np.uint8) * 220
    
    # Draw grid cells
    for x in range(h):
        for y in range(w):
            # Calculate cell position in the scaled image
            x1 = x * CELL_SIZE
            y1 = y * CELL_SIZE
            x2 = (x + 1) * CELL_SIZE
            y2 = (y + 1) * CELL_SIZE
            
            # Determine cell color
            if known_obstacles[x, y] == 1:
                color = (80, 80, 80)  # Dark gray for obstacles
            elif visited_mask[x, y] == 1:
                color = (180, 180, 255)  # Light blue for visited
            else:
                color = (220, 220, 220)  # Light gray for unvisited
                
            # Draw the cell
            cv2.rectangle(img, (y1, x1), (y2, x2), color, -1)
            
            # Draw grid lines
            cv2.rectangle(img, (y1, x1), (y2, x2), (150, 150, 150), 1)
            
            # Draw victims
            if known_victims[x, y] == 1:
                center = (y1 + CELL_SIZE // 2, x1 + CELL_SIZE // 2)
                cv2.circle(img, center, CELL_SIZE // 4, (0, 0, 255), -1)
    
    return img

def draw_wind_field(img: np.ndarray, wind_grid: np.ndarray):
    """Draw wind vectors for the entire grid"""
    h, w, _ = wind_grid.shape
    for x in range(h):
        for y in range(w):
            wx, wy = wind_grid[x, y]
            # Only draw if there's significant wind
            if abs(wx) > 0.1 or abs(wy) > 0.1:
                draw_wind_in_cell(img, (x, y), (wx, wy))
    return img

def draw_wind_in_cell(img: np.ndarray, position: Tuple[int, int], wind_vec: Tuple[float, float]):
    """Draw wind vector in a specific cell"""
    x, y = position
    # Calculate center position in scaled image
    cx = int((y + 0.5) * CELL_SIZE)
    cy = int((x + 0.5) * CELL_SIZE)
    
    wx, wy = wind_vec
    # Calculate wind strength (0-1 scale)
    strength = min(1.0, np.sqrt(wx**2 + wy**2))
    
    # Calculate arrow length based on strength
    length = int(strength * CELL_SIZE * 0.4)
    
    # Calculate end point
    end_x = cx + int(wy * length)
    end_y = cy + int(wx * length)
    
    # Choose color based on wind strength
    if strength > 0.7:
        color = (0, 0, 255)  # Strong wind (red)
    elif strength > 0.4:
        color = (0, 165, 255)  # Medium wind (orange)
    else:
        color = (0, 255, 0)  # Light wind (green)
    
    # Draw arrow
    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

def draw_uav(img: np.ndarray, position: Tuple[int, int], color: Tuple[int, int, int]):
    """Draw UAV as a colored triangle with proper coordinates"""
    x, y = position
    # Calculate center position in scaled image
    cx = int((y + 0.5) * CELL_SIZE)
    cy = int((x + 0.5) * CELL_SIZE)
    
    # Create triangle points (adjust for CELL_SIZE)
    size = CELL_SIZE // 3
    pts = np.array([[
        (cx, cy - size),          # Top
        (cx - size, cy + size),   # Bottom-left
        (cx + size, cy + size)    # Bottom-right
    ]])
    cv2.fillPoly(img, pts, color)
    cv2.polylines(img, pts, True, (0, 0, 0), 1)  # Black outline

def draw_wind(img: np.ndarray, position: Tuple[int, int], wind_vec: Tuple[float, float]):
    """Draw wind vector at UAV position"""
    x, y = position
    # Calculate center position in scaled image
    cx = int((y + 0.5) * CELL_SIZE)
    cy = int((x + 0.5) * CELL_SIZE)
    
    wx, wy = wind_vec
    # Calculate arrow length based on wind strength
    strength = min(1.0, np.sqrt(wx**2 + wy**2))
    length = int(strength * CELL_SIZE * 0.6)
    
    # Calculate end point
    end_x = cx + int(wy * length)
    end_y = cy + int(wx * length)
    
    # Draw thick black arrow
    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 0, 0), 3, tipLength=0.4)
    
    # Draw red indicator if wind is strong
    if strength > 0.5:
        cv2.circle(img, (cx, cy), CELL_SIZE//4, (0, 0, 255), 2)

def draw_info_panel(img: np.ndarray, step: int, total_victims: int, found_victims: int, 
                   visited: int, total_cells: int):
    """Add information panel to the visualization"""
    h, w, _ = img.shape
    panel_height = 100
    full_img = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    full_img[:h, :] = img
    
    # Draw panel background
    cv2.rectangle(full_img, (0, h), (w, h+panel_height), (50, 50, 50), -1)
    
    # Add text information
    texts = [
        f"Step: {step}",
        f"Victims: {found_victims}/{total_victims}",
        f"Coverage: {100*visited/total_cells:.1f}%",
        "Wind Key: [GREEN: Light, ORANGE: Medium, RED: Strong]"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(full_img, text, (20, h + 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw wind scale examples
    for i, strength in enumerate([0.3, 0.6, 0.9]):
        x_pos = w - 300 + i*80
        cv2.arrowedLine(full_img, (x_pos, h + 60), 
                        (x_pos + 30, h + 60), 
                        get_wind_color(strength), 2, tipLength=0.3)
        cv2.putText(full_img, f"{strength}", (x_pos, h + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return full_img

def get_wind_color(strength: float) -> Tuple[int, int, int]:
    """Get color based on wind strength"""
    if strength > 0.7:
        return (0, 0, 255)  # Red
    elif strength > 0.4:
        return (0, 165, 255)  # Orange
    return (0, 255, 0)  # Green

def save_frame(img: np.ndarray, frame_num: int):
    """Save frame to output directory"""
    cv2.imwrite(f"outputs/frames/frame_{frame_num:04d}.png", img)

def generate_heatmap(victim_counts: np.ndarray, path: str):
    """Generate victim discovery heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(victim_counts, cmap="YlGn", vmin=0, vmax=10)
    
    # Add grid lines
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, victim_counts.shape[1], 1), minor=True)
    plt.yticks(np.arange(-0.5, victim_counts.shape[0], 1), minor=True)
    
    plt.colorbar(label="Victim Discovery Count")
    plt.title("Victim Discovery Heatmap")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()