src/
├── baseline_lib/                         # BASELINE
|   ├── cbs/
|   |   └── cbs_planner.py                # CBS implementation.
|   |
|   ├── results/
|   |   ├── plots                         # Saved plots when training.
|   |   └── intersection_schedules.json   # Saved results for demo (auto-generated when training).
|   |
|   ├── simulation/
|   |   ├── grid_env.py                   # Grid environment with disturbances.
|   |   ├── disturbance_utils.py          # Disturbances injection.
|   |   ├── plots_generator.py            # Generate the plots for the 4 metrics against collisions
|   |   ├── scenario_generator.py         # Vehicle start/goal generator.
|   |   └── schedule_genereator.py        # Main simulation runner.
|   |
|   └── README.md                         # This file.
|
├── project/                              # PROPOSED SOLUTION - no external ML libraries.
│   |                              
|   ├── cbs_planner.py                # Global (Priority Queue - Heap).
|   ├── grid.py                       # 10x10 grid of the intersection.
|   ├── demo.py                       # Demo of the q_learning full project only.
|   ├── schedule.py                   # Each vehicles schedule.
|   ├── vehicle.py                    # Single agent.                          
|   ├── q_learning_agent.py           # QLearning agent
|   ├── simulation_results.json       # Schedules
|   └── simulation.py                 # Intersection simulation.
|
└──user_interface/
      ├── static/
      |   ├── css/
      |   |   └── style.css               # css styles
      |   └── js/
      |       └── script.js               # Scripts for movement, animations and interaction with the server.
      ├── templates/
      |   └── index.html                  # Graphics for the UI
      └── ui_controller.py                # UI server (reads from JSON).


- How to train the baseline model  : cd to baseline_lib
                                   - python simulation/schedule_genereator - Results saved in json file in results folder.

                                   : cd to baseline_lib
						                       - python simulation/plots_genereator - This will generate the plots for all the parameters.

- How to train the project model : cd to project
                                 - python simulation.py - Results saved in simulation_results.json file.