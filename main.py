# === main.py ===
"""
Grid Exploration Simulator with Training and Testing Modes

How to Run:
1. Training:
   python main.py --episodes 50
   
2. Testing:
   python eval.py

3. Smoke Test:
   python main.py --smoke-test
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils.obstacles as obstacle_utils
import utils.visualize as vis
from envs.grid_uav_env import GridUAVEnv
from planning.a_star import AStarPlanner
from planning.genetic_waypoints import GeneticWaypointsPlanner
from rl.dqn_agent import DQNAgent
from utils.metrics import MetricsTracker


@dataclass
class Config:
    grid_size: int = 15
    p_obs: float = 0.15
    victims: int = 20
    episodes: int = 50
    max_steps: int = 900
    ga_iters: int = 200
    ga_pop: int = 60
    waypoints_per_uav: int = 12
    dqn_lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    obs_mode: str = "cnn"
    render: bool = True
    save_video: bool = True
    seed: int = 42
    smoke_test: bool = False
    test_mode: bool = False
    model_path: str = "outputs/models/dqn_model.pth"

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--episodes", type=int, default=50)
        parser.add_argument("--render", type=int, default=1)
        parser.add_argument("--save-video", type=int, default=1)
        parser.add_argument("--obs-mode", choices=["tabular", "cnn"], default="cnn")
        parser.add_argument("--p-obs", type=float, default=0.15)
        parser.add_argument("--victims", type=int, default=20)
        parser.add_argument("--ga-iters", type=int, default=200)
        parser.add_argument("--grid", type=int, default=15)
        parser.add_argument("--smoke-test", action="store_true")
        parser.add_argument("--test-mode", action="store_true")
        parser.add_argument("--model-path", type=str, default="outputs/models/dqn_model.pth")
        args = parser.parse_args()
        return cls(
            grid_size=args.grid,
            p_obs=args.p_obs,
            victims=args.victims,
            episodes=args.episodes,
            obs_mode=args.obs_mode,
            render=bool(args.render),
            save_video=bool(args.save_video),
            seed=args.seed,
            smoke_test=args.smoke_test,
            test_mode=args.test_mode,
            model_path=args.model_path
        )


def setup_directories():
    os.makedirs("outputs/frames", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    if os.path.exists("outputs/frames"):
        for f in os.listdir("outputs/frames"):
            if f.startswith("frame_"):
                os.remove(os.path.join("outputs/frames", f))


def run_smoke_test():
    """Lightweight test with small grid"""
    test_config = Config(
        grid_size=7,
        p_obs=0.1,
        victims=5,
        episodes=1,
        max_steps=100,
        ga_iters=20,
        ga_pop=20,
        waypoints_per_uav=3,
        render=False,
        save_video=False,
        smoke_test=True,
    )
    main(test_config)
    assert os.path.exists("outputs/metrics.json")
    assert os.path.exists("outputs/heatmap.png")
    print("Smoke test passed!")


def run_episode(env, ga_planner, a_star_planner, dqn_agent, config, is_training=True):
    """Run a single episode and return metrics"""
    obs, info = env.reset()
    episode_metrics = {
        "coverage": [],
        "victims_found": 0,
        "obstacle_hits": 0,
        "uav_collisions": 0,
        "rewards": [],
        "steps": 0
    }
    
    # Run GA for waypoint planning
    waypoints0, waypoints1 = ga_planner.run_ga(
        env.known_obstacles, 
        env.visited_mask,
        env.start_positions
    )
    waypoint_idx0, waypoint_idx1 = 0, 0
    
    frame_count = 0
    done = False
    for step in range(config.max_steps):
        # Check if we should switch to A*
        if waypoint_idx0 >= len(waypoints0) or waypoint_idx1 >= len(waypoints1):
            # Get A* actions
            action0 = a_star_planner.get_action(
                env.uav_positions[0],
                env.visited_mask,
                env.known_obstacles
            )
            action1 = a_star_planner.get_action(
                env.uav_positions[1],
                env.visited_mask,
                env.known_obstacles
            )
            actions = [action0, action1]
        else:
            # DQN actions
            actions = dqn_agent.act(obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        
        # Update waypoints if reached
        if waypoint_idx0 < len(waypoints0):
            if env.uav_positions[0] == waypoints0[waypoint_idx0]:
                waypoint_idx0 += 1
                reward += 2  # Bonus for reaching waypoint
        
        if waypoint_idx1 < len(waypoints1):
            if env.uav_positions[1] == waypoints1[waypoint_idx1]:
                waypoint_idx1 += 1
                reward += 2
        
        # Store experience and train DQN during training
        if is_training:
            dqn_agent.store_experience(obs, actions, reward, next_obs, done)
            if len(dqn_agent.replay_buffer) > 1000:
                dqn_agent.train_step()
        
        # Update metrics
        episode_metrics["victims_found"] += info["new_victims"]
        episode_metrics["obstacle_hits"] += info["obstacle_hits"]
        episode_metrics["uav_collisions"] += info["uav_collision"]
        episode_metrics["rewards"].append(reward)
        coverage = np.sum(env.visited_mask) / (config.grid_size**2 - np.sum(env.known_obstacles))
        episode_metrics["coverage"].append(coverage)
        episode_metrics["steps"] = step
        
        # Render and save frame
        if config.render or config.save_video:
            frame = env.render()
            if config.save_video:
                vis.save_frame(frame, frame_count)
            if config.render:
                cv2.imshow("Grid Exploration", frame)
                if cv2.waitKey(30) == 27:  # ESC key
                    break
        frame_count += 1
        
        if done:
            break
    
    return episode_metrics


def main(config: Config):
    setup_directories()
    metrics = MetricsTracker()
    global_heatmap = np.zeros((config.grid_size, config.grid_size))
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Initialize environment
    env = GridUAVEnv(config)
    ga_planner = GeneticWaypointsPlanner(config, env.grid_size)
    a_star_planner = AStarPlanner(env.grid_size)
    dqn_agent = DQNAgent(
        obs_mode=config.obs_mode,
        grid_size=config.grid_size,
        action_dim=8,  # 4 actions × 2 UAVs
        lr=config.dqn_lr,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
    )
    
    # Load model if in test mode
    if config.test_mode:
        if os.path.exists(config.model_path):
            dqn_agent.load(config.model_path)
            print(f"Loaded trained model from {config.model_path}")
        else:
            print("No trained model found. Starting with random weights.")
        dqn_agent.epsilon = 0.05  # Minimal exploration during testing
    
    for episode in range(config.episodes):
        episode_metrics = run_episode(
            env, ga_planner, a_star_planner, dqn_agent, config, 
            is_training=(not config.test_mode)
        )
        
        # Update global heatmap
        global_heatmap += env.victim_discovery_count
        
        # Update episode metrics
        metrics.update(
            coverage=episode_metrics["coverage"][-1],
            victims_found=episode_metrics["victims_found"],
            obstacle_hits=episode_metrics["obstacle_hits"],
            uav_collisions=episode_metrics["uav_collisions"],
            avg_reward=np.mean(episode_metrics["rewards"]),
            steps=episode_metrics["steps"]
        )
        
        print(f"Episode {episode+1}/{config.episodes}: "
              f"Coverage={metrics.coverage_history[-1]*100:.1f}%, "
              f"Victims={metrics.victims_history[-1]}/{config.victims}, "
              f"Steps={metrics.steps_history[-1]}, "
              f"Reward={metrics.reward_history[-1]:.2f}")
    
    # Save results
    if config.test_mode:
        metrics.save("outputs/test_metrics.json")
    else:
        metrics.save("outputs/train_metrics.json")
        # Save trained model
        dqn_agent.save(config.model_path)
        print(f"Saved trained model to {config.model_path}")
    
    vis.generate_heatmap(global_heatmap, "outputs/heatmap.png")
    
    if config.render:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = Config.from_args()
    if config.smoke_test:
        run_smoke_test()
    else:
        main(config)