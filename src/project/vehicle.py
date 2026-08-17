# vehicle

class Vehicle:
    def __init__(self, vehicle_id, start, goal):
        self.vehicle_id = vehicle_id
        self.start = start
        self.goal = goal
        self.current_position = start # setting the starting point at the starts.
        self.schedule = None # the time step and the row and col
        self.path = [] # list of positions for vehicle

    # setting the schedule and path positions
    def set_schedule(self, schedule):
        self.schedule = schedule # set the schedule
        self.path = [pos for pos, _ in schedule.steps] # set the positions in the schedule

    # get observation around the car from the grid.
    def get_observation(self, grid):
        row, col = self.current_position # get the row and col that the vehicle is in.
        observation = [] # create an empty list of observation
        
        # 3by3 loop
        for changing_row in [-1, 0, 1]: # loop though up row, same row, and down row
            for changing_col in [-1, 0, 1]: # loop though left col, same col, right col
                # get the number of rows and cols and keep on sliding.
                new_row, new_col = row + changing_row, col + changing_col 
                # check if the observation is in the grid
                if grid.is_inside_grid((new_row, new_col)):
                    observation.append(1 if grid.is_occupied((new_row, new_col)) else 0) # append to the list of obversation 0 if empty and 1 if occupied
                else:
                    observation.append(0)
        return observation # return the list on the observation

    # update the position of the car
    def update_position(self, new_position):
        self.current_position = new_position

    # check if the vehicle has reached its goal
    def has_reached_goal(self):
        return self.current_position == self.goal