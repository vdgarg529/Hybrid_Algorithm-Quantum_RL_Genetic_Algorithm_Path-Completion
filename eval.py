# === eval.py ===
"""
Evaluation Script for Trained Grid Exploration Model

How to Run:
python eval.py --episodes 10 --model-path outputs/models/dqn_model.pth
"""

import argparse
import os
from main import Config, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--model-path", type=str, default="outputs/models/dqn_model.pth", 
                        help="Path to trained model")
    parser.add_argument("--render", type=int, default=0, help="Render evaluation episodes")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed")
    args = parser.parse_args()
    
    # Create evaluation config
    eval_config = Config(
        episodes=args.episodes,
        model_path=args.model_path,
        render=bool(args.render),
        test_mode=True,
        seed=args.seed,
        save_video=False
    )
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Starting evaluation with {args.episodes} episodes...")
    main(eval_config)
    print("Evaluation complete. Results saved to outputs/test_metrics.json")