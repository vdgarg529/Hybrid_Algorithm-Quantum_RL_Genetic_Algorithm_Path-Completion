@echo off
REM === setup_project.cmd ===
REM Creates folder structure and files for UAV Grid Exploration Simulator

REM Create root directory
mkdir grid_uav_simulator
cd grid_uav_simulator

REM Create main project files
echo. > main.py

REM Create environment directory and files
mkdir envs
echo. > envs\grid_uav_env.py

REM Create planning directory and files
mkdir planning
echo. > planning\genetic_waypoints.py
echo. > planning\a_star.py

REM Create RL directory and files
mkdir rl
echo. > rl\dqn_agent.py

REM Create utils directory and files
mkdir utils
echo. > utils\wind.py
echo. > utils\obstacles.py
echo. > utils\visualize.py
echo. > utils\metrics.py

REM Create outputs directory
mkdir outputs

echo Project structure created successfully!
echo Please copy the code content into the respective files