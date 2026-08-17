# grid

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [["empty" for _ in range(width)] for _ in range(height)] # create the rows and cols and initialise them to empty
        self.vehicles = {} # vehicles id and positions.

    # place a vehicle on the grid at the start
    def place_vehicle(self, vehicle_id, position):
        row, col = position # set the row and cols.
        self.cells[row][col] = "occupied" # set the cells as occupied.
        self.vehicles[vehicle_id] = position # use the vehicle id to set the position.

    # move a vehicle on the grid.
    def move_vehicle(self, vehicle_id, new_position):
        old_row, old_col = self.vehicles[vehicle_id] # use the vehicle_id at get the position of the vehicle.
        self.cells[old_row][old_col] = "empty" # set that cell to empty.
        new_row, new_col = new_position # set the row and col to new position.
        self.cells[new_row][new_col] = "occupied" # set the new position to occupied.
        self.vehicles[vehicle_id] = new_position # use the id of the vehicle to set the position in the dictionary

    # check is the cell is occupied, used in vehicle
    def is_occupied(self, position):
        row, col = position
        return self.cells[row][col] == "occupied" # return true if the cell is occupied

    # check if the vehicle is in the grid or not, used in vehicle
    def is_inside_grid(self, position):
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width # return true if inside, false if it is outside the grid.

    # get the next row.
    def get_neighbours(self, position):
        row, col = position
        moves = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)] # move the vehicle in row or col only.
        return [p for p in moves if self.is_inside_grid(p)] # if the position is in the grid, return the position in any move.

    # get the cell state, occupied or empty.
    def get_cell_status(self, position):
        row, col = position
        return self.cells[row][col]

    # display the grid.
    def print_grid(self):
        display = [["." for _ in range(self.width)] for _ in range(self.height)] # display the dots in length of width and height
        for vehicle_id, pos in self.vehicles.items(): 
            r, c = pos # get the row and col
            display[r][c] = vehicle_id # display the vehicle ids
        print("\n  " + " ".join(str(c) for c in range(self.width)))
        for r, row in enumerate(display):
            print(str(r) + " " + " ".join(row))
        print()