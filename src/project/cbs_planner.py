# cbs_planner
# Conflict‑Based Search Global Planner.

import heapq
from schedule import Schedule

class CBSNode:
    def __init__(self, constraints, schedules):
        self.constraints = constraints
        self.schedules = schedules
        self.total_cost = self.calculate_total_cost() # set the total cost using the function

    # function to calculate the total cost of two nodes
    def calculate_total_cost(self):
        return sum(len(sch.steps) for sch in self.schedules.values()) # loop through the schedule values and get the length of the steps

    # how to compare two nodes and take the one with less total cost = length of steps
    def __lt__(self, other):
        return self.total_cost < other.total_cost


class CBSPlanner:
    def __init__(self, grid, vehicles):
        self.grid = grid
        self.vehicles = {v.vehicle_id: v for v in vehicles}

    # conflict resolution and collision free schedule generator.
    def plan(self, starts, goals):
        root_schedules = {} # root schedule with no constraints
        
        # loop through vehicles
        for vid in self.vehicles:
            # get the start and goals
            start = starts[vid]
            goal = goals[vid]
            
            # get the shortest path from start to goal, with no collisions
            path = self.find_path_for_vehicle(start, goal, [])
            
            # check if the path is valid
            if path is None:
                print(f"ERROR: No path for vehicle {vid}")
                return None
            
            # set the vehicle schedule
            root_schedules[vid] = Schedule(vid, path)

        # no constraint schedule.
        root_node = CBSNode([], root_schedules)
        pq = [root_node] # the first node in the list
        explored = 0 # set the visited nodes to zero

        while pq:
            # we pop that first node.
            node = heapq.heappop(pq)
            explored += 1 # increment the number of explored nodes

            # find the collisions in the node schedules
            conflict = self.find_first_conflict(list(node.schedules.values()))
            
            # if no collision is found
            if conflict is None:
                print(f"CBS solution found after {explored} nodes.")
                return list(node.schedules.values()) # return the schedeles

            # set the constraints if found
            v1, v2, pos, t = conflict
            
            # loop through the two constraints
            for constrained in (v1, v2):
                
                # create a constraint
                new_constraint = (constrained, pos, t)
                
                # copy the parants constraints and add a new one
                child_constraints = node.constraints + [new_constraint] 
                
                # copy the parent's schedules
                child_schedules = dict(node.schedules)

                # find the new path with the constraints
                start = starts[constrained]
                goal = goals[constrained]
                
                # create a new list with the constraints
                constraints_for_v = [(pos, t) for vid, pos, t in child_constraints if vid == constrained]
                
                # plan with constraints
                new_path = self.find_path_for_vehicle(start, goal, constraints_for_v)
                if new_path is not None:
                    # create a new schedule with constraints
                    child_schedules[constrained] = Schedule(constrained, new_path)
                    
                    # push the node into the heap
                    heapq.heappush(pq, CBSNode(child_constraints, child_schedules))

        print("CBS found no solution.")
        return None


    # plan the shotest path A* not worrying about collisions
    def find_path_for_vehicle(self, start, goal, constraints):
        forbidden = set(constraints) # check if they are fobbiden cells
        
        # f_cost, g_cost, current position, current time step, path so far (start, 0)
        start_entry = (self.manhattan(start, goal), 0, start, 0, [(start, 0)])
        pq = [start_entry] # add the entry into the heap
        visited = set() # create a set of visited nodes

        while pq:
            # pop from the first node in the heap
            f, g, pos, t, path = heapq.heappop(pq)
            
            # check if position is equal to goal
            if pos == goal:
                return path

            # get the state
            state = (pos, t)
            
            # check state is visited
            if state in visited:
                continue
            visited.add(state) # if not add the state into the set of state visited

            # increment to next time step
            next_t = t + 1
            
            # check if the next position is in the grid
            for nxt in self.grid.get_neighbours(pos) + [pos]:
                if (nxt, next_t) in forbidden:
                    continue # skip if the next position is in the forbidden list
                if (nxt, next_t) in visited:
                    continue # skip if the next position is in the visited list
                
                # add the g_cost by 1
                new_g = g + 1
                
                # add the new f_goal by the reduced manhattan distance
                new_f = new_g + self.manhattan(nxt, goal)
                
                # push the new node into the heap
                heapq.heappush(pq, (new_f, new_g, nxt, next_t, path + [(nxt, next_t)]))

        return None


    # find a conflict between two schedules
    def find_first_conflict(self, schedules):
        for i in range(len(schedules)):
            for j in range(i+1, len(schedules)):
                
                # set schedule 1 and 2
                s1, s2 = schedules[i], schedules[j]
                
                # get the time steps
                times1 = {t for _, t in s1.steps}
                times2 = {t for _, t in s2.steps}
                
                # loop through the time steps
                for t in times1 & times2:
                    
                    # get the positions
                    p1 = s1.get_position_at(t)
                    p2 = s2.get_position_at(t)
                    
                    # check is the positions are the same
                    if p1 == p2 and p1 is not None:
                        
                        # if yes return the vehicle 1 id and the vehicle 2 id and the time step
                        return (s1.vehicle_id, s2.vehicle_id, p1, t)
        return None

    # Calculates the Manhattan distance between two grid positions.
    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])