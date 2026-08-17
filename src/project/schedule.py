# schedule
# Time stamped plan for one vehicle.

class Schedule:
    def __init__(self, vehicle_id, steps):
        self.vehicle_id = vehicle_id
        self.steps = steps   # list of ((row, col), time_step)

    # get the position of the car using the time_step
    def get_position_at(self, time_step):
        for position, time_stepp in self.steps: # loop through the time steps on the vehicle schedule
            if time_stepp == time_step: # check if the given time step is equal to the schedule time step
                return position # return that positions
        # check if the the time step is not the more than the number of time steps
        if self.steps:
            last_pos, last_t = self.steps[-1] # get the last step and its position
            if time_step > last_t: # check if the given step is grater than the last step
                return last_pos # if it is greater then return the last position
        return None

    # get the remaining steps
    def get_remaining_steps(self, from_time_step):
        return [(pos, t) for pos, t in self.steps if t >= from_time_step] # loop through the steps while checking if the given time step is less than or equal to the schedule time step and return the position and time step

    # return true if the steps are finished
    def is_complete(self, current_time_step):
        if not self.steps: # check if steps is empty
            return True # return true if the steps empty
        # if not empty return last element in the steps ((row, col), time_step) and get the last step at 1
        # if the given step is greater than the last step return true
        return current_time_step >= self.steps[-1][1] 

    # print the schedule as a list of (timestep, (row,col))
    def print_schedule(self):
        print(f"\nSchedule for Vehicle {self.vehicle_id}:")
        
        # loop through the steps and get the timesteps and row and col list for a vehicle
        for positionnn, time_steppp in self.steps:
            print(f"  Time Step {time_steppp}: go to row {positionnn[0]}, col {positionnn[1]}")