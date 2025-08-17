# Open Markdown Preview:
```
Ctrl + Shift + V (Windows/Linux) or Cmd + Shift + V (macOS)
```


# Smoke Test
```python
python main.py --smoke-test
```
Behavior:
    Uses 7x7 grid with 5 victims
    Runs 1 episode with reduced parameters
Validates:
    Environment initialization
    Basic movement
    Victim detection
    Output file creation
Output:
    "Smoke test passed!" on success
    Automatic cleanup after test


# Training:
```python
python main.py --episodes 50
```

## Headless mode (for training)
```python
python main.py --episodes 50 --render 0
```
render = 1 for  visual rendering




# Testing:
```python
python eval.py --episodes 10
```




```python
python eval.py --episodes 10 --model-path outputs/models/dqn_model.pth
```




# Folder Structure
```text
grid_uav_simulator/
├── main.py
├── eval.py
├── requirements.txt
├── envs/
│   └── grid_uav_env.py
├── planning/
│   ├── genetic_waypoints.py
│   └── a_star.py
├── rl/
│   └── dqn_agent.py
├── utils/
│   ├── wind.py
│   ├── obstacles.py
│   ├── visualize.py
│   └── metrics.py
└── outputs/
    ├── frames/
    ├── models/
    ├── heatmap.png
    ├── test_metrics.json
    └── train_metrics.json
```





# Most Important requirements.txt

```text
# === requirements.txt ===
numpy==1.26.4
gymnasium==0.29.1
opencv-python==4.9.0.80
torch==2.2.1
matplotlib==3.8.3
```

```python
pip install -r requirements.txt
pip install torch==2.2.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```
