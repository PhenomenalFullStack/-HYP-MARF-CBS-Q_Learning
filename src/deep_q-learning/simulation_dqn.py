# simulation_dqn.py

# imports
import os # tools for files and folders.
import sys
import json # json file read/write
import numpy as np

# file path setup
_HERE = os.path.dirname(os.path.abspath(__file__)) # script path
PROJECT_ROOT = os.path.join(_HERE, "..", "project") # go up one level and into the project folder
sys.path.insert(0, PROJECT_ROOT) # scrips in project folder.

# imports for project scripts
from grid import Grid
from vehicle import Vehicle
from schedule import Schedule
from cbs_planner import CBSPlanner
from deep_q_learning_agent import DQNAgent

# Constants
GRID_SIZE = 10 # 10x10 grid
MAX_STEPS = 50 # maximum steps per episode
TRAINING_EPISODES = 500 # number of training episodes
STATE_SIZE = (
    9 + 2 + 2 + 1
)  # observation (9) + schedule_pos (2) + actual_pos (2) + delay (1) = 14 numbers


# Scenario generation (1‑8 vehicles)
def generate_scenario(num_vehicles, seed=None):
    base_ids = ["A", "B", "C", "D"] # fist 4 vehicles are A, B, C, D. Additional vehicles will be E, F, G, H.
    sides = [
        {
            "lane_axis": "row",
            "lane_value": 5,
            "start_depth_base": 9,
            "start_depth_dir": -1, # move vehicle from top to bottom for row lanes.
            "goal_depth_base": 0,
        },
        {
            "lane_axis": "col",
            "lane_value": 4,
            "start_depth_base": 9,
            "start_depth_dir": -1, # move vehicle from right to left for column lanes.
            "goal_depth_base": 0,
        },
        {
            "lane_axis": "row",
            "lane_value": 4,
            "start_depth_base": 0,
            "start_depth_dir": 1, # move vehicle from bottom to top for row lanes.
            "goal_depth_base": 9,
        },
        {
            "lane_axis": "col",
            "lane_value": 5,
            "start_depth_base": 1,
            "start_depth_dir": 1, # move vehicle from left to right for column lanes.
            "goal_depth_base": 9,
        },
    ]
    occupants = [list() for _ in range(4)] # 4 empty lists for the 4 sides of the intersection.
    if num_vehicles <= 4:
        for i in range(num_vehicles):
            occupants[i].append(base_ids[i]) # put vehicle i alone on side i.
    else:
        for i in range(4):
            occupants[i].append(base_ids[i]) # first, A-D go on sides 0-3.
        extra = num_vehicles - 4 # how many more vehicles to add after A-D.
        next_id_ord = ord("E") # next vehicle ID starts at E (ASCII 69).
        side_idx = 0 # start adding extra vehicles from side 0.
        for _ in range(extra): 
            vid = chr(next_id_ord) # char(69) = E, char(70) = F
            next_id_ord += 1 # next use F, G, H
            occupants[side_idx].insert(0, vid) # insert at front of the list to keep A-D at the back of the list for each side.
            side_idx = (side_idx + 1) % 4 # cycle through sides 0-3 for extra vehicles.

    starts = {} # map vehicle ID to starting position (row, col)
    goals = {} # map vehicle ID to goal position (row, col)
    for side, queue in zip(sides, occupants): # zip pair each side with its queue of vehicles.
        if not queue: # check if the queue is empty
            continue
        
        # pull the info from the side dictionary for easier access
        lane_axis = side["lane_axis"]
        lane_value = side["lane_value"]
        start_base = side["start_depth_base"]
        start_dir = side["start_depth_dir"]
        goal_base = side["goal_depth_base"]
        goal_dir = -start_dir # goal queue counts the opposite way to the start queue
        for offset, vid in enumerate(queue): # offset = 0  for the front vehicle and 1 for the next
            start_depth = start_base + offset * start_dir # how far along the lane it starts
            goal_depth = goal_base + offset * goal_dir # how far along the lane it must reach
            if lane_axis == "row":
                starts[vid] = (lane_value, start_depth) # row is fixed, column varies
                goals[vid] = (lane_value, goal_depth)
            else:
                starts[vid] = (start_depth, lane_value) # column is fixed, row varies
                goals[vid] = (goal_depth, lane_value)
    return starts, goals # Start() and Goal() for each vehicle ID






# Helper functions

# Give each vehicle its schedule from the CBS planner.
def distribute_schedules(vehicles, schedules):
    for sch in schedules: # for each schedule.
        for v in vehicles: # for each vehicle.
            if v.vehicle_id == sch.vehicle_id: # check if the vehicle ID matches the schedule's vehicle ID.
                v.set_schedule(sch) # give the vehicle its schedule.
                break # run away.

# Reset the simulation to its initial state.
def reset_simulation(vehicles, grid):
    for r in range(grid.height): # for each row in the grid.
        for c in range(grid.width): # for each column in the grid.
            grid.cells[r][c] = "empty" # set the cell to empty.
    grid.vehicles.clear() # clear the vehicles.
    for v in vehicles: # for each vehicle.
        v.current_position = v.start # move it back to its starting position.
        grid.place_vehicle(v.vehicle_id, v.start) # place the vehicle on the grid at its starting position.

# Get the position of a vehicle on its path based on the index.
def get_position_on_path(vehicle, index):
    if index < 0: # if the index is negative, return the starting position.
        return vehicle.start
    if index >= len(vehicle.path): # if the index is beyond the path length, return the goal position.
        return vehicle.goal
    return vehicle.path[index] # return the position on the path at the given index.





# Episode runner, one attempt, resert the world and repeat.
def run_episode(
    vehicles, grid, delays, agents, vehicle_ids, time_limit=MAX_STEPS, train=True
):
    reset_simulation(vehicles, grid) # refresh world to initial state.
    paths = {v.vehicle_id: [v.start] for v in vehicles} # every position visisted.
    done = False # set true when all vehicles have reached their goals.
    step = 0 # time step counter
    total_reward = 0.0 # episodic reward.
    # a copy of starting delays.
    delay = {vid: delays.get(vid, 0) for vid in vehicle_ids}
    path_index = {vid: 0 for vid in vehicle_ids} # how along a vehicle is. 0 at start

    prev_states = {} # remember the past state
    prev_actions = {} # remember the past action.

    # keep looping until vehicles reached their goals and time limit (time steps) is out
    while not done and step < time_limit:
        scheduled_current = {} # schedule position the vehicle should be at
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0] # take the first vehicle from the list
            eff_idx = step - delay[vid] # timestep of the vehicle - delay of the vehicle
            scheduled_current[vid] = get_position_on_path(v, eff_idx) # delayed vehicle reads the past position of the schedule

        # Taking an action
        actions = {} # choose(0,1,2)
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            obs = v.get_observation(grid) # get the sensor noise from the vehicle.
            state = agents[vid].build_state_key(
                obs, scheduled_current[vid], v.current_position, delay[vid]  # save the 14-number state.
            )

            if delay[vid] > 0: # delayed: try moving, wait, or catch  up
                allowed = [0, 1, 2]
            else:
                allowed = [0, 1] # move or wait

            # training, select an action exploration.
            if train:
                action = agents[vid].select_action(
                    obs, scheduled_current[vid], v.current_position, delay[vid], allowed
                )
            else: # not training, exploitation.
                action = agents[vid].select_action(
                    obs, scheduled_current[vid], v.current_position, delay[vid], allowed
                )
            actions[vid] = action # save action
            if train:
                prev_states[vid] = state # remember what i saw
                prev_actions[vid] = action # remember action taken

        # apply actions to the delay counter and work out new positions
        for vid in vehicle_ids:
            if actions[vid] == 1: # wait: falls one step behind
                delay[vid] += 1 
            elif actions[vid] == 2: # catch up: recover one step of delay
                if delay[vid] > 0:
                    delay[vid] -= 1

        new_positions = {} # where to be after
        new_indices = {} # new index on the path
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            current_idx = path_index[vid] # where the vehicle is on its path now
            if actions[vid] == 0:
                new_idx = current_idx + 1 # proceed one cells
            elif actions[vid] == 2:
                new_idx = current_idx + 2 # catch up two cells
            else:
                new_idx = current_idx # wait
            new_idx = min(new_idx, len(v.path) - 1) if v.path else current_idx # do not run of the end of the path
            new_pos = get_position_on_path(v, new_idx) # convert index if the path to grid cell
            new_positions[vid] = new_pos 
            new_indices[vid] = new_idx

        # Collision counting
        collision_flags = {vid: False for vid in vehicle_ids} # did this vehicle collide
        collision_count = 0
        for i, vid1 in enumerate(vehicle_ids): 
            for vid2 in vehicle_ids[i + 1 :]: # everything after i in the list
                
                # every posible pair of vehicles
                v1 = [v for v in vehicles if v.vehicle_id == vid1][0]
                v2 = [v for v in vehicles if v.vehicle_id == vid2][0]
                
                # Do not count collisions for vehicles that have already reached their goals.
                if v1.has_reached_goal() or v2.has_reached_goal():
                    continue
                
                # Count a collision if two vehicles occupy the same position after their moves.
                if new_positions[vid1] == new_positions[vid2]:
                    collision_flags[vid1] = True
                    collision_flags[vid2] = True
                    collision_count += 1 # increment the collision count

        # move all vehicles, compute rewards and learn.
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0] # take first vehicle
            old_pos = v.current_position # where it was
            grid.move_vehicle(vid, new_positions[vid]) # update the grid
            v.update_position(new_positions[vid]) # update the vehicle
            paths[vid].append(new_positions[vid]) # record the paths
            path_index[vid] = new_indices[vid] # remember how far the vehicle it

            # REWARD by using dqn method.
            if train:
                true_next_sched_pos = get_position_on_path(v, step + 1) # schedule position at next step
                reward = agents[vid].compute_reward(
                    old_pos,
                    new_positions[vid],
                    true_next_sched_pos,
                    delay[vid],
                    collision_flags[vid],
                    v.goal, # score of what just happened
                )
                total_reward += reward # increment the total reward.

                next_obs = v.get_observation(grid) # sensor noise after move
                noisy_next_sched_pos = get_position_on_path(v, (step + 1) - delay[vid]) # get next position of vehicle on step and if there is a delay is still the same
                
                # build next state key.
                next_state = agents[vid].build_state_key(
                    next_obs, noisy_next_sched_pos, new_positions[vid], delay[vid]
                )
                # fill the buffer
                agents[vid].update_q_table(
                    prev_states[vid], prev_actions[vid], reward, next_state
                ) # store (state,action,reward,nextstate) in buffer and when there is 64+, do one learning step.

        # check if all vehicles have reached their goals.
        done = all(v.has_reached_goal() for v in vehicles)
        step += 1 # increment the time steps

    return paths, total_reward, collision_count # learning summary





# Main training loop over vehicle counts
def main():
    vehicle_counts = list(range(1, 9)) # all vehicles
    configs = [(sn, bd, cl) for sn in [0, 1] for bd in [0, 1] for cl in [0, 1]] # 8 combination of sensor noise
    all_results = [] # eveery experimental outcomes

    for num_vehicles in vehicle_counts:
        print(f"\n") 
        print(f"Training DQN for {num_vehicles} vehicles")
        print(f"\n")

        starts, goals = generate_scenario(num_vehicles) # generate starting positions
        vehicle_ids = sorted(starts.keys()) # aphabetic 
        grid = Grid(GRID_SIZE, GRID_SIZE) # create grid 10x10
        vehicles = [Vehicle(vid, starts[vid], goals[vid]) for vid in vehicle_ids] # create a vehicle

        planner = CBSPlanner(grid, vehicles) # create a CBS planner
        schedules = planner.plan(starts, goals) # schedules for each vehicle
        if schedules is None:
            print(f"CBS failed for {num_vehicles} vehicles, skipping.")
            continue
        distribute_schedules(vehicles, schedules) # distribute each schedule to it vehicle

        # Build json schedule.
        cbs_schedule_data = {} # CBS plan in json
        for sch in schedules:
            cbs_schedule_data[sch.vehicle_id] = [
                {"time": t, "position": pos} for pos, t in sch.steps # unpack positions and time step.
            ]

        # loop for each configurations
        for sn, bd, cl in configs:
            config_name = f"sensorNoise{sn}_brakingDelay{bd}_commLatency{cl}" # configuration names
            print(f"\n  Training DQN for config: Sensor={sn}, Brake={bd}, Comm={cl}")

            delays = {vid: 0 for vid in vehicle_ids} # no vevicle delayed
            delays[vehicle_ids[0]] = sn + bd + cl # first vehicle get delay

            # for all vehicles
            agents = {}
            for v in vehicles:
                #  Deep Q-Network Agent initialization for each vehicle
                agents[v.vehicle_id] = DQNAgent(
                    vehicle=v, # which vehicle
                    state_size=STATE_SIZE, # 14 inputs
                    action_size=3, # move, wait, catch up
                    lr=0.001, # learning rate
                    gamma=0.9, # discount factor
                    epsilon=1.0, # epsilon
                    epsilon_decay=0.995, # reduce epsilon by 0.5% per episode
                    batch_size=64, # learn from 64 memories at a time
                    target_update=100, # refresh the target network after 100 learning steps
                )

            # run the traing for 500 times
            for ep in range(TRAINING_EPISODES):
                run_episode(vehicles, grid, delays, agents, vehicle_ids, train=True)
                for agent in agents.values(): # just the agents
                    agent.decay_epsilon() # reduce epsilon

            for agent in agents.values():
                agent.epsilon = 0.0 # no more random moves, exploit what learned

            # one test episode, keep path and collision count
            eval_paths, _, eval_collisions = run_episode(
                vehicles, grid, delays, agents, vehicle_ids, train=False
            )

            # save the schedules
            actual_schedule = {}
            for vid in vehicle_ids:
                steps = [
                    {"time": t, "position": pos}
                    for t, pos in enumerate(eval_paths[vid]) # (0, first pos), (1, sec pos)
                ]
                actual_schedule[vid] = steps

            entry = { # dqn record
                "num_vehicles": num_vehicles,
                "configuration": {
                    "sensor_noise": sn,
                    "braking_delay": bd,
                    "comm_latency": cl,
                },
                "cbs_schedule": cbs_schedule_data, # output the CBS pure plan
                "actual_schedule": actual_schedule,
                "collisions": eval_collisions,
                "training_episodes": TRAINING_EPISODES,
            }
            all_results.append(entry)

            # print CBS Schedule
            print(f"\nCBS Schedule (for {num_vehicles} vehicles):")
            for vid, steps in cbs_schedule_data.items():
                print(f"  Vehicle {vid}: {steps}")
            
            # print DQN Schedule
            print(
                f"\nActual Schedule after DQN ({config_name}, {num_vehicles} vehicles):"
            )
            for vid, steps in actual_schedule.items():
                print(f"  Vehicle {vid}: {steps}")
            # print collisions
            print(f" Number of Collisions: {eval_collisions}")
    
    # write to json file.
    with open("dqn_simulation_results_all_vehicles.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(
        f"\nAll DQN results saved to dqn_simulation_results_all_vehicles.json (total {len(all_results)} entries)."
    )


# call main function
if __name__ == "__main__":
    main()
