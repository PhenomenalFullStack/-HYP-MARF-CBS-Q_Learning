# simulation

import json
from grid import Grid
from vehicle import Vehicle
from schedule import Schedule
from cbs_planner import CBSPlanner
from q_learning import QLearningAgent

# setting up the grid, start and vehicle IDs
GRID_SIZE = 10
STARTS = {'A': (5, 9), 'B': (9, 4), 'C': (4, 0), 'D': (1, 5)}
GOALS  = {'A': (5, 0), 'B': (0, 4), 'C': (9, 5), 'D': (9, 5)}
VEHICLE_IDS = ['A', 'B', 'C', 'D']
MAX_STEPS = 50 # number of steps each vehicle can take
TRAINING_EPISODES = 500 # traing episodes


# give each vehicle each schedule
def distribute_schedules(vehicles, schedules):
    for schedule in schedules: # loop through the schedule
        for vihicle in vehicles: # loop through the vehicles
            if vihicle.vehicle_id == schedule.vehicle_id: # check if the vehicle id is equal to the schedule id
                vihicle.set_schedule(schedule) # set that vehicle to its schedule
                break

# reset the simulation to start
def reset_simulation(vehicles, grid):
    # reset the grid
    for r in range(grid.height):
        for c in range(grid.width):
            grid.cells[r][c] = "empty"
    grid.vehicles.clear()
    # place the vehicle at the start
    for v in vehicles:
        v.current_position = v.start
        grid.place_vehicle(v.vehicle_id, v.start)

# get list of positions of a vehicle
def get_position_on_path(vehicle, index):
    if index < 0:
        return vehicle.start # return starting position
    if index >= len(vehicle.path):
        return vehicle.goal # return goal position
    return vehicle.path[index]


# train the agent for a max of time
def run_episode(vehicles, grid, delays, agents, time_limit=MAX_STEPS, train=True):
    # reset simulation to start
    reset_simulation(vehicles, grid)

    # create a path for each vehicle_id with each schedule
    paths = {v.vehicle_id: [v.start] for v in vehicles}
    done = False
    step = 0
    total_reward = 0.0

    # get a delay of vehicle due to disturbance else zero
    delay = {vid: delays.get(vid, 0) for vid in VEHICLE_IDS}
    
    # loop through the number of vehicle ids and map the vehicle ID to integer 0
    path_index = {vid: 0 for vid in VEHICLE_IDS}

    # empty previous states and actions
    prev_states = {}
    prev_actions = {} # action for vehicle at each time step

    # loop while vehicles have not reached their goals and steps is not max steps
    while not done and step < time_limit:
        scheduled_current = {} # create the current empty schedule
        for vid in VEHICLE_IDS: # loop through the veicle ids
            v = [v for v in vehicles if v.vehicle_id == vid][0] # get the first list on a vehicle object
            eff_idx = step - delay[vid] # how many steps the vehicle is currently behind schedule
            
            # the scheduled position that the vehicle should be at, given its current delay
            scheduled_current[vid] = get_position_on_path(v, eff_idx)


        # choosen action for each time step based on eploration vs exploitation
        actions = {}  # (A, 0)
        for vid in VEHICLE_IDS: # loop through the vehicles ids.
            v = [v for v in vehicles if v.vehicle_id == vid][0] # get the vehicle object using the id
            obs = v.get_observation(grid) # get the observation
            
            # build a key state
            state = agents[vid].build_state_key(obs, scheduled_current[vid], v.current_position, delay[vid])
            
            # train the agent if train = true
            # allow random Exploration through epsilon-greedy policy
            if train: 
                # let the agent select an action.
                action = agents[vid].select_action(obs, scheduled_current[vid], v.current_position, delay[vid])
            else:
                # allow Exploitation train = false 
                q_vals = agents[vid].get_q_values(state) # get the q_values
                action = max(range(len(q_vals)), key=lambda i: q_vals[i]) # get the action index with high q_value, takes the index i and look for the q_value in q_vals
            actions[vid] = action # select that action
            
            # store the current state string into the previous state and action into previous action
            if train:
                prev_states[vid] = state
                prev_actions[vid] = action

        # delay
        for vid in VEHICLE_IDS:
            if actions[vid] == 1: # index wait action.
                delay[vid] += 1
            if actions[vid] == 2: # index delay action
                if delay[vid] > 0:
                    delay[vid] -= 1  # reduce the delay

        # calculate new position
        new_positions = {}
        new_indices = {}
        
        # loop though the veicle ids, get the new path with delay
        for vid in VEHICLE_IDS:
            v = [v for v in vehicles if v.vehicle_id == vid][0] # create a vehicle object
            current_idx = path_index[vid] # create the current index action
            if actions[vid] == 0:      # move one step forward
                new_idx = current_idx + 1
            elif actions[vid] == 2:    # skip 2 step to catch up
                new_idx = current_idx + 2
            else:                      # wait
                new_idx = current_idx
                
            # check if not out of the grid 
            new_idx = min(new_idx, len(v.path) - 1) if v.path else current_idx
            
            # get positions using the vehicles using the index
            new_pos = get_position_on_path(v, new_idx) 
            new_positions[vid] = new_pos # get the grid position
            new_indices[vid] = new_idx # get the index of the vehicle in the path

        # check for collisions
        collision_flags = {vid: False for vid in VEHICLE_IDS} # set collision to false
        collision_count = 0 # set the collision count to zero
        for i, vid1 in enumerate(VEHICLE_IDS): # loop through the vehicle ids
            for vid2 in VEHICLE_IDS[i+1:]: # loop through the second vehicle ids
                if new_positions[vid1] == new_positions[vid2]: # check if the vehicles have collision
                    collision_flags[vid1] = True
                    collision_flags[vid2] = True
                    collision_count += 1 # count each pair collision at this step


        # Move vehicles, update grid, record paths, compute rewards, update the Q_TABLES
        for vid in VEHICLE_IDS:
            v = [v for v in vehicles if v.vehicle_id == vid][0] # create a vehicle object
            old_pos = v.current_position # save old position for reward calculation
            grid.move_vehicle(vid, new_positions[vid]) # move a vehicle
            v.update_position(new_positions[vid]) # update the vehicle position
            paths[vid].append(new_positions[vid]) # add to the path of the vehicle
            path_index[vid] = new_indices[vid] # update the index of the vehicle

            # reward and Q_TABLE update only if training.
            if train:
                # where the vehicle should be next step with zero delay
                true_next_sched_pos = get_position_on_path(v, step + 1)

                # where the vehicle thinks it should be next step, accounting for its perceived delay
                noisy_next_sched_pos = get_position_on_path(v, (step + 1) - delay[vid])

                # compute reward
                reward = agents[vid].compute_reward(
                    old_pos,
                    new_positions[vid],
                    true_next_sched_pos,
                    delay[vid],
                    collision_flags[vid],
                    GOALS[vid]
                )
                total_reward += reward # increment the reward

                # Updating the Q_TABLE
                next_obs = v.get_observation(grid) # get the next position observation
                
                # build the next state string
                next_state = agents[vid].build_state_key(next_obs, noisy_next_sched_pos, new_positions[vid], delay[vid])
                
                # update the Q_TABLE
                agents[vid].update_q_table(prev_states[vid], prev_actions[vid], reward, next_state)

        # exit the loop if the vehicle has reached its goals and steps are max steps
        done = all(v.has_reached_goal() for v in vehicles)
        step += 1

    # return the path and rewards
    return paths, total_reward, collision_count





# create Grid, Vehicle, GlobalPlan, QLeaningAgent, and train
def main():

    # create the grid
    grid = Grid(GRID_SIZE, GRID_SIZE) # 10x10
    
    # create a vehicles list objects
    vehicles = [Vehicle(vid, STARTS[vid], GOALS[vid]) for vid in VEHICLE_IDS]

    # generate the cbs schedules (list of schedules)
    planner = CBSPlanner(grid, vehicles)
    schedules = planner.plan(STARTS, GOALS) # generate a collision free schedule plan
    if schedules is None: # if schedule failed to be generated
        print("CBS failed, exiting.")
        return
    distribute_schedules(vehicles, schedules) # give each vehicle its schedule

    # create an empty dicionary
    cbs_schedule_data = {}
    for sch in schedules: # loop through the schedules
        cbs_schedule_data[sch.vehicle_id] = [{"time": t, "position": pos} for pos, t in sch.steps] # set the time and position

    # create the configurations
    configs = [(sn, bd, cl) for sn in [0,1] for bd in [0,1] for cl in [0,1]] #  the disturbances can only 0 or 1
    results = [] # create the empty list of the trained schedules

    # loop through the disturbances
    for sn, bd, cl in configs:
        config_name = f"sensorNoise{sn}_brakingDelay{bd}_commLatency{cl}" # get the disturbances names
        print(f"\nTraining configurations: Sensor Noise={sn}, BrakIng Delay={bd}, Communication Latency={cl}") # heading when traing
        # set the delay for the vehicle
        delays = {vid: 0 for vid in VEHICLE_IDS} # set all the delays to zero
        delays['A'] = sn + bd + cl # combine the delays

        # set the all the QLearningAgent vehicle agents
        agents = {v.vehicle_id: QLearningAgent(v, actions=3, lr=0.1, gamma=0.9, epsilon=1.0) for v in vehicles}

        # loop trough the number of episode to decay the epsilon, Exploration
        for ep in range(TRAINING_EPISODES):
            run_episode(vehicles, grid, delays, agents, train=True) # tain the agents by running one episodes
            for agent in agents.values(): 
                agent.decay_epsilon() # decay the episolon

        # loop through the agents, Exploitation
        for agent in agents.values():
            agent.epsilon = 0.0 # act greedy, choose best moves
        eval_paths, eval_collisions, _  = run_episode(vehicles, grid, delays, agents, train=False) # do not train, choose best path

        # empty schedule.
        actual_schedule = {}
        # loop through the schedule IDs
        for vid in VEHICLE_IDS:
            # get the list of time step and positision
            steps = [{"time": t, "position": pos} for t, pos in enumerate(eval_paths[vid])]
            actual_schedule[vid] = steps

        # get each schedule piece and the disturbances
        each_schedule_for_vehicles = {
            "configuration": {"sensor_noise": sn, "braking_delay": bd, "comm_latency": cl},
            "cbs_schedule": cbs_schedule_data,
            "actual_schedule": actual_schedule,
            "collisions": eval_collisions,
            "training_episodes": TRAINING_EPISODES,
        }
        results.append(each_schedule_for_vehicles) # list of schedules

        # print all the schedules cbs and q_learning in the cmd
        # cbs schedules
        print(f"\n CBS Schedule (Best):")
        for vid, steps in cbs_schedule_data.items():
            print(f"Vehicle {vid}: {steps}")
        # q_schedules
        print(f"\n Actual Schedule after Q-learning ({config_name}):")
        for vid, steps in actual_schedule.items():
            print(f"Vehicle {vid}: {steps}")

    # saving all the result schedules in a .json file
    with open("simulation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n Results saved to simulation_results.json")


if __name__ == "__main__":
    main()