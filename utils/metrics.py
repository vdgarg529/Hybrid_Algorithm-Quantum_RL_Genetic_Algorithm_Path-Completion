# # === utils/metrics.py ===
# import json
# from dataclasses import dataclass, field
# from typing import List


# @dataclass
# class MetricsTracker:
#     coverage_history: List[float] = field(default_factory=list)
#     victims_history: List[int] = field(default_factory=list)
#     obstacle_history: List[int] = field(default_factory=list)
#     collision_history: List[int] = field(default_factory=list)
#     reward_history: List[float] = field(default_factory=list)
    
#     def update(
#         self,
#         coverage: float,
#         victims_found: int,
#         obstacle_hits: int,
#         uav_collisions: int,
#         avg_reward: float
#     ):
#         self.coverage_history.append(coverage)
#         self.victims_history.append(victims_found)
#         self.obstacle_history.append(obstacle_hits)
#         self.collision_history.append(uav_collisions)
#         self.reward_history.append(avg_reward)
    
#     def save(self, path: str):
#         data = {
#             "coverage": self.coverage_history,
#             "victims_found": self.victims_history,
#             "obstacle_hits": self.obstacle_history,
#             "uav_collisions": self.collision_history,
#             "avg_reward": self.reward_history
#         }
#         with open(path, "w") as f:
#             json.dump(data, f, indent=2)


# === utils/metrics.py ===
import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricsTracker:
    coverage_history: List[float] = field(default_factory=list)
    victims_history: List[int] = field(default_factory=list)
    obstacle_history: List[int] = field(default_factory=list)
    collision_history: List[int] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    steps_history: List[int] = field(default_factory=list)
    
    def update(
        self,
        coverage: float,
        victims_found: int,
        obstacle_hits: int,
        uav_collisions: int,
        avg_reward: float,
        steps: int
    ):
        self.coverage_history.append(coverage)
        self.victims_history.append(victims_found)
        self.obstacle_history.append(obstacle_hits)
        self.collision_history.append(uav_collisions)
        self.reward_history.append(avg_reward)
        self.steps_history.append(steps)
    
    def save(self, path: str):
        data = {
            "coverage": self.coverage_history,
            "victims_found": self.victims_history,
            "obstacle_hits": self.obstacle_history,
            "uav_collisions": self.collision_history,
            "avg_reward": self.reward_history,
            "steps": self.steps_history
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)