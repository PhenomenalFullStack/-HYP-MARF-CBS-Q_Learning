# cbs_planner.py

# import statements
import heapq
import time


"""
this class initialise the parameters and get the position of the vehicle at a given time step.
"""
class Schedule:
    
    # constructor, initializing the parameters vehicle_id and steps 
    def __init__(self, vehicle_id, steps):
        self.vehicle_id = vehicle_id # veicle id A, B, C
        self.steps = steps           # ((0, (row, col)), (1, (row, col)), (2, (row, col)), ...)

    # this helps to get the position of the vehicle using the time step.
    def get_position_at(self, time_step):
        # initialize the previous position to None if there are no steps, otherwise set it to the first step's position.
        prev = self.steps[0][1] if self.steps else None
        
        # loop though the steps and check if the time requested time step is equal to the time step in the steps, 
        for t, pos in self.steps:       
            if t == time_step:          # if the requested time step is equal to the time step then return the current position.
                return pos              # return current rows and cols.
            
            elif t > time_step:         # if the requested time step is greater than the time step return the previous position.
                return prev             # return the previous rows and cols.
            
            # if the requested time step is less than the time step then set the previous position to the current position.
            prev = pos
            
        # gets the last position by iterating through the turple of time steps [-1] and get the position [1],
        # if there are no steps return None.
        return self.steps[-1][1] if self.steps else None



"""
This class represents a node in the CBS search tree, containing a constraints, schedules, and total cost of the schedules. 
"""
class CBSNode:
    
    # initialize the parameters constraints, schedules and total cost of the schedules.
    def __init__(self, constraints, schedules):
        self.constraints = constraints # list of constraints in the form of (vehicle_id, time_step, position)
        self.schedules = schedules     # (vehicle_id, steps) -> (vehicle_id: A, steps: (0, (row, col)), (1, (row, col)), (2, (row, col)), vehicle_id: B, steps: (0, (row, col)), (1, (row, col)), (2, (row, col)),,...)
        
        # loop through the schedules and get the number of steps in each schedule and sum them up to get the total cost of the schedules.
        self.total_cost = sum(len(s.steps) for s in schedules.values()) # we add the number of steps in each schedule to get the total cost of the schedules.

    # less-than method: this method is used to compare two CBSNode objects based on their total cost. In the min heap.
    # we use this method to sort the nodes in the priority queue based on their total cost.
    def __lt__(self, other):
        return self.total_cost < other.total_cost # if the total cost of the current node is less than the total cost of the other node, return True, otherwise return False.


# This class implements the Conflict-Based Search (CBS) algorithm 
# It takes a grid environment and a list of vehicle IDs as input, 
# and provides a method to plan paths for the vehicles from their start positions to their goal positions while avoiding conflicts.
class CBSPlanner:

    # constructor, initializing the parameters grid_env, vehicle_ids, max_nodes, max_a_star_iter and time_limit.
    def __init__(self, grid_env, vehicle_ids):
        self.grid = grid_env  # the grid environment, which contains information about the grid size, obstacles, and vehicle properties.
        self.vehicle_ids = vehicle_ids # the list of vehicle IDs for which paths need to be planned.
        self.max_nodes = 4000  # the maximum number of nodes to explore in the CBS search tree before giving up on finding a solution.
        self.max_a_star_iter = 4000  # the maximum number of iterations to perform in the A* search algorithm when finding a path for a vehicle, before giving up on finding a path.
        self.time_limit = 3.0 # the maximum time limit (in seconds) for the CBS algorithm to run before giving up on finding a solution.


    # this is the high-level of the CBS algorithm that resolves conflicts between vehicles by adding constraints and replanning paths for the vehicles. 
    # it uses a priority queue to explore nodes in the search tree based on their total cost, and returns a list of schedules for the vehicles if a solution is found.
    def plan(self, starts, goals):
        start_time = time.time() # if solution is not found
        root_schedules = {} # stores schedules
        
        # loop through each vehicles
        for vih_id in self.vehicle_ids: 
            delay = self.grid.vehicles[vih_id].get('delay_steps', 0) # get the delay for this vehicle
            path = self.find_path_for_vehicle(starts[vih_id], goals[vih_id], [], delay) # run the A-star to find the shortest path
            if path is None:
                print(f"CBS: No path for vehicle {vih_id} even without constraints.")
                return None
            root_schedules[vih_id] = Schedule(vih_id, path) # stores the schedule for that vehicle

        heap = [CBSNode([], root_schedules)] # initialise the priority queue - one node with no constraints.
        
        nodes_explored = 0
        # heap pop and push. 
        while heap and nodes_explored < self.max_nodes:
            # time out
            if time.time() - start_time > self.time_limit:
                print(f"CBS: time limit ({self.time_limit}s) exceeded, skip.")
                return None

            # take the node with the lowest cost out of the heap - there will always be one node in the heap
            node = heapq.heappop(heap)
            nodes_explored += 1

            # set the conflict using the find_first_conflict function
            conflict = self.find_first_conflict(node.schedules)
            if conflict is None:
                return list(node.schedules.values()) # return the values of the schedule with the conflicts (vih_di's, (row, cols), times_steps)

            vid_id_1, vid_id_2, pos, t = conflict # get the values of the conflicts.
            for constrained_vehicle_id in (vid_id_1, vid_id_2): # the vehicles that are constrained.
                child_constraints = node.constraints + [(constrained_vehicle_id, pos, t)] # create a new contraint by adding the contraints to the position and time_step
                child_schedules = dict(node.schedules) # copy the parant schedule
                start = starts[constrained_vehicle_id]
                goal = goals[constrained_vehicle_id]
                delay = self.grid.vehicles[constrained_vehicle_id].get('delay_steps', 0)
                veh_constraints = [(p, tm) for (v, p, tm) in child_constraints if v == constrained_vehicle_id] # use a four loop to extract the contraint for a specific vehicle
                new_path = self.find_path_for_vehicle(start, goal, veh_constraints, delay) # use the A* to find the path for the vehicle
                if new_path is not None:
                    child_schedules[constrained_vehicle_id] = Schedule(constrained_vehicle_id, new_path) # Update the schedule for that vehicle
                    heapq.heappush(heap, CBSNode(child_constraints, child_schedules)) # add a child node back to the priority queue

        print(f"CBS: no solution within {self.max_nodes} nodes or time limit.")
        return None

    """this is the low level A* search algorithm that finds a shortest path for one vehicle from its start position to its goal position avoiding the obsticles or contraints."""
    # it uses a min-heap priority queue to explore nodes in the search tree based on their f-cost (g-cost + h-cost), and returns a list of steps for the vehicle if a path is found.
    # f = g + h, where g is the actual cost from the start node to the current node, and h is the heuristic estimate of the cost from the current node to the goal node.
    # the min heap always pops the node with the lowest f-cost, which is the most promising node to explore next.
    def find_path_for_vehicle(self, start, goal, constraints, delay_steps=0):
        # convert the list of constraints into a set [(position(row, col), time_step), (position(row, col), time_step), ...]
        forbidden_set = set(constraints)

        # this is the heuristic function that calculates the Manhattan distance between the current position and the goal position (predicted cost).
        def heuristic(start_pos):
            return abs(start_pos[0] - goal[0]) + abs(start_pos[1] - goal[1]) # the (start_row - goal_row) + (start_col - goal_col).

        # if the start and goal positions are in the same row or column, we can restrict the search to that row or column, which can speed up the search.
        if start[0] == goal[0]:
            fixed_axis, direction = 'row', (1 if goal[1] > start[1] else -1)  # if the goal is to the right of the start, direction is 1 (right), otherwise -1 (left)
        elif start[1] == goal[1]:
            fixed_axis, direction = 'col', (1 if goal[0] > start[0] else -1)  # if the goal is below the start, direction is 1 (down), otherwise -1 (up)
        else:
            fixed_axis, direction = None, None # if the start and goal are not in the same row or column, we don't restrict the search to a specific axis.

    
        # this function returns the neighbouring positions of the current position, considering the fixed axis if applicable.
        def lane_neighbours(pos):
            if fixed_axis == 'row': # if the fixed axis is 'row', we only consider the current position and the position in the same row but in the direction of the goal.
                row, col = pos  # get the current row and column of the position
                cands = [(row, col)] # we set the current position as the first candidate
                if col != goal[1]: 
                    num_cols = col + direction
                    if 0 <= num_cols < self.grid.width:
                        cands.append((row, num_cols))
                        # e.g. if the current position is (5, 0) and the goal position is (5, 5), then the fixed axis is 'row' and the direction is 1 (right).
                        # row = 5
                        # 0 != 5 -> num_cols = 0 + 1 = 1
                        # 1 != 5 -> num_cols = 1 + 1 = 2
                        # 2 != 5 -> num_cols = 2 + 1 = 3
                        # 3 != 5 -> num_cols = 3 + 1 = 4
                        # 4 != 5 -> num_cols = 4 + 1 = 5
                        
            elif fixed_axis == 'col':
                row, col = pos
                cands = [(row, col)]
                if row != goal[0]:
                    num_rows = row + direction
                    if 0 <= num_rows < self.grid.height:
                        cands.append((num_rows, col))
                        # e.g. if the current position is (0, 5) and the goal position is (5, 5), then the fixed axis is 'col' and the direction is 1 (down).
                        # col = 5
                        # 0 != 5 -> num_rows = 0 + 1 = 1
                        # 1 != 5 -> num_rows = 1 + 1 = 2
                        # 2 != 5 -> num_rows = 2 + 1 = 3
                        # 3 != 5 -> num_rows = 3 + 1 = 4
                        # 4 != 5 -> num_rows = 4 + 1 = 5
            else:
                cands = self.get_neighbours(pos) + [pos] # if there is no fixed axis, we consider all valid neighbouring positions and the current position itself.
            return cands

        # this is the injection of the delay steps into the path, which is a list of tuples (time_step, position(row, col)).
        """"
            start = (5, 9)
            delay_steps = 2
            forced_wait_path = [(0, (5,9)), (1, (5,9)), (2, (5,9))]
        """
        forced_wait_path = [(time_step, start) for time_step in range(delay_steps + 1)] # how many steps to wait at the start position before moving towards the goal position.
        if start == goal:
            return forced_wait_path

        # initialize the min-heap priority queue with the heuristic cost, g-cost of 0, and the forced wait path.
        heap = [(heuristic(start), 0, start, delay_steps, forced_wait_path)] # h_cost, 0, start, delay_steps,
        visited = set()
        iter_count = 0

        # while the heap is not empty and the number of iterations is less than the maximum allowed iterations, 
        while heap and iter_count < self.max_a_star_iter:
            iter_count += 1 # how many iterations we have done so far, if it exceeds the max_a_star_iter, we will stop the search and return None.
            f_cost, g_cost, pos, num_steps, path = heapq.heappop(heap) # pop the node with the lowest f-cost from the heap, which contains the current position, g-cost, time step, and path taken to reach this position.
            
            # if the current position and time step have already been visited, we skip this node and continue to the next iteration.
            if (pos, num_steps) in visited:
                continue
            visited.add((pos, num_steps)) # add the current position and time step to the visited set.
            
            # check if current position is equal to the goal
            if pos == goal:
                return path
            next_time_step = num_steps + 1 # if not then we increment the number of steps
            
            # push the neighbouring positions of the current position into the heap, if they are not in the forbidden set or visited set.
            for nxt_position in lane_neighbours(pos): # the lane_neighbours(pos) updates the next position.
                
                # check if the next position and time step are in the forbidden set or visited set, if so, we skip this neighbour and continue to the next neighbour.
                if (nxt_position, next_time_step) in forbidden_set or (nxt_position, next_time_step) in visited:
                    continue
                
                new_g_cost = g_cost + 1 # increment the g_cost
                new_path = path + [(next_time_step, nxt_position)] # add the new turple of the next time step and position into the list of the path.
                
                # push the new node into the heap with the updated f-cost, g-cost, next position, next time step, and new path.
                heapq.heappush(heap, (new_g_cost + heuristic(nxt_position), new_g_cost, nxt_position, next_time_step, new_path))

        # check if the number of iterations has reached the maximum allowed iterations, if so, we print a message indicating that the A* search has reached the maximum iterations for the given start and goal positions.
        if iter_count >= self.max_a_star_iter:
            print(f"  A* max iterations reached for start {start} -> goal {goal}")
            
        # return nothing
        return None

    
    # helper function for low level A* that returns the valid neighbouring positions of a given position in the grid, considering the grid boundaries.
    # this only goes up, down, left, right and not diagonally.
    def get_neighbours(self, pos):
        row, col = pos # get the current row and column of the position
        valid = [] # initialize an empty list to store the valid neighbouring row and column positions
        
        # in the loop we can only move by the direction of the row and column, we can only move up, down, left, right and not diagonally.
        for num_rows, num_cols in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:  # safe list - move horizontal while keeping the vertical safe, or move vertical while keeping the horizontal safe
            if 0 <= num_rows < self.grid.height and 0 <= num_cols < self.grid.width:
                valid.append((num_rows, num_cols))
        return valid



    """To do, implement the logic for finding neighbouring positions. but we first check the center position of 
       the vehicle and when the vehicle is in the middle of the lane, we can move it diagonally.
       this will help when the vehicle is making a turn, to simulate the vehicle's movement more accurately. 
       the vehicle can move diagonally when it is in the middle of the lane, which allows for smoother turns and better path planning."""

    """ ((A, (1,1), 1), (A, (1,2), 2)) <-> ((B, (0,1), 1), (A, (1,1), 2)) """


    # helper function for the high-level conflict resolution of CBS algorithm 
    # that finds the first conflict between the schedules of the vehicles.
    def find_first_conflict(self, schedules):
        # schedules is a dictionary: {vehicle_id: Schedule_object}
        # Each Schedule object contains a list of (time, position) steps
        
        # Convert the dictionary values (Schedule objects) into a list
        sched_list = list(schedules.values()) # [Schedule("A"), Schedule("B"), Schedule("C"), Schedule("D")]
        
        # Get the total number of schedules (vehicles)
        n = len(sched_list) # n=4
        
        # Outer loop: iterate through each schedule (except the last one)
        for i in range(n): # i goes from 0 to n-2 (first vehicle to second-last vehicle)
            
            # Inner loop: compare schedule i with every schedule after it
            # j goes from i+1 to n-1 (vehicles after i)
            for j in range(i + 1, n):
                # Get the two schedules we are currently comparing
                schedule_1 = sched_list[i]
                schedule_2 = sched_list[j]
                
                # Extract all time steps from schedule_1
                # We use a set comprehension: {time for each (time, position) in steps}
                times1 = {t for t, _ in schedule_1.steps} # times1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
                
                # Extract all time steps from schedule_2
                times2 = {t for t, _ in schedule_2.steps} # times2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
                
                # Find time steps that appear in BOTH schedules
                # This uses set intersection: times1 & times2
                for time_step in times1 & times2:
                    # Get the position of vehicle 1 at time t
                    # get_position_at() returns (row, column) or None
                    p1 = schedule_1.get_position_at(time_step)
                    
                    # Get the position of vehicle 2 at time t
                    p2 = schedule_2.get_position_at(time_step)
                    
                    # Check if both vehicles are at the SAME position at time t
                    if p1 == p2 and p1 is not None:
                        # Conflict found
                        return (
                            schedule_1.vehicle_id,  # "A"
                            schedule_2.vehicle_id,  # "B"
                            p1,                     # (5, 5)   conflict position
                            time_step               # 4        conflict time
                        )
        
        # If we checked ALL pairs of schedules and found NO conflicts
        # Return None (collision-free)
        return None