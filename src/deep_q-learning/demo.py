import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

# Constants
GRID_SIZE = 10
CELL_SIZE = 50
CANVAS_SIZE = GRID_SIZE * CELL_SIZE

VEHICLE_COLORS = {
    'A': '#ff0000',
    'B': '#0044ff',
    'C': '#00aa00',
    'D': '#ff8800',
    'E': '#9900cc',
    'F': '#ff00ff',
    'G': '#090310',
    'H': '#00ccc2'
}

# Helper: build time‑position mapping
def build_time_positions(schedule):
    """
    Convert a schedule dict {vid: [{"time": t, "position": [r,c]}, ...]}
    into a list of dicts: for each time step, positions of all vehicles.
    Missing times are filled with the last known position.
    """
    # First, collect all time steps across vehicles
    all_times = set()
    for vid, steps in schedule.items():
        for step in steps:
            all_times.add(step['time'])
    if not all_times:
        return [], 0
    max_time = max(all_times)

    # Build position over time for each vehicle
    pos_by_vid_time = {}
    for vid, steps in schedule.items():
        pos_map = {}
        for step in steps:
            pos_map[step['time']] = tuple(step['position'])
        # Fill missing times with last known position
        last_pos = None
        filled = {}
        for t in range(max_time + 1):
            if t in pos_map:
                last_pos = pos_map[t]
            filled[t] = last_pos  # may be None if no position yet
        pos_by_vid_time[vid] = filled

    # Build time steps: list of dicts {vid: (row, col)}
    time_steps = []
    for t in range(max_time + 1):
        state = {}
        for vid in schedule.keys():
            pos = pos_by_vid_time[vid].get(t)
            if pos is not None:
                state[vid] = pos
        time_steps.append(state)

    return time_steps, max_time

# Main Demo
class DQNDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN Intersection Demo")

        # Load JSON data
        json_path = os.path.join(os.path.dirname(__file__), "dqn_simulation_results_all_vehicles.json")
        if not os.path.exists(json_path):
            messagebox.showerror("Error", f"JSON file not found: {json_path}")
            root.destroy()
            return
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {e}")
            root.destroy()
            return

        # GUI variables
        self.num_vehicles = tk.IntVar(value=4)
        self.sensor_noise = tk.IntVar(value=0)
        self.braking_delay = tk.IntVar(value=0)
        self.comm_latency = tk.IntVar(value=0)
        self.schedule_type = tk.StringVar(value="actual")  # "actual" or "cbs"

        self.time_index = tk.IntVar(value=0)
        self.max_time = 0
        self.time_steps = []
        self.vehicle_ids = []
        self.goals = {}
        self.collisions = 0
        self.is_playing = False
        self.after_id = None

        self.build_gui()
        self.draw_grid()

    def build_gui(self):
        # Control Frame
        ctrl = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        ctrl.pack(pady=10, fill='x')

        # Row 0
        ttk.Label(ctrl, text="Vehicles:").grid(row=0, column=0, padx=5)
        nv_spin = ttk.Spinbox(ctrl, from_=1, to=8, textvariable=self.num_vehicles, width=5)
        nv_spin.grid(row=0, column=1, padx=5)

        ttk.Label(ctrl, text="Sensor Noise:").grid(row=0, column=2, padx=5)
        sn_combo = ttk.Combobox(ctrl, values=[0, 1], textvariable=self.sensor_noise, width=5, state='readonly')
        sn_combo.grid(row=0, column=3, padx=5)

        ttk.Label(ctrl, text="Braking Delay:").grid(row=0, column=4, padx=5)
        bd_combo = ttk.Combobox(ctrl, values=[0, 1], textvariable=self.braking_delay, width=5, state='readonly')
        bd_combo.grid(row=0, column=5, padx=5)

        ttk.Label(ctrl, text="Comm Latency:").grid(row=0, column=6, padx=5)
        cl_combo = ttk.Combobox(ctrl, values=[0, 1], textvariable=self.comm_latency, width=5, state='readonly')
        cl_combo.grid(row=0, column=7, padx=5)

        # Row 1
        ttk.Label(ctrl, text="Schedule:").grid(row=1, column=0, padx=5)
        sch_combo = ttk.Combobox(ctrl, values=["DQN (actual)", "CBS"], 
                                 textvariable=self.schedule_type, width=15, state='readonly')
        sch_combo.grid(row=1, column=1, columnspan=2, padx=5)
        sch_combo.current(0)  # default DQN

        self.load_btn = ttk.Button(ctrl, text="Load", command=self.load_schedule)
        self.load_btn.grid(row=1, column=3, padx=20)

        # Collisions label
        self.coll_label = ttk.Label(ctrl, text="Collisions: -")
        self.coll_label.grid(row=1, column=4, columnspan=3, padx=5, sticky='w')

        # Canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(pady=10)
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()

        # Control Bar
        bar = ttk.Frame(self.root)
        bar.pack(pady=5, fill='x')

        self.prev_btn = ttk.Button(bar, text="Prev", command=self.prev_step, state='disabled')
        self.prev_btn.pack(side='left', padx=5)

        self.play_btn = ttk.Button(bar, text="Play", command=self.toggle_play, state='disabled')
        self.play_btn.pack(side='left', padx=5)

        self.next_btn = ttk.Button(bar, text="Next", command=self.next_step, state='disabled')
        self.next_btn.pack(side='left', padx=5)

        self.time_label = ttk.Label(bar, text="Time: 0 / 0")
        self.time_label.pack(side='left', padx=20)

        self.slider = ttk.Scale(bar, from_=0, to=0, orient='horizontal', 
                                variable=self.time_index, command=self.slider_changed)
        self.slider.pack(side='left', fill='x', expand=True, padx=10)

        # Status bar
        self.status = ttk.Label(self.root, text="Ready. Select parameters and click Load.", relief='sunken')
        self.status.pack(fill='x', pady=5)

    def draw_grid(self, state=None):
        """Draw the grid and vehicles. If state is None, draw empty grid."""
        self.canvas.delete('all')
        cell = CELL_SIZE
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            x = i * cell
            y = i * cell
            self.canvas.create_line(x, 0, x, CANVAS_SIZE, fill='#cccccc', tags='grid')
            self.canvas.create_line(0, y, CANVAS_SIZE, y, fill='#cccccc', tags='grid')
        # Draw intersection area (central 2x2)
        self.canvas.create_rectangle(4*cell, 4*cell, 6*cell, 6*cell, 
                                     outline='#ff0000', width=2, fill='', tags='intersection')
        # Draw goal markers if we have goals
        if hasattr(self, 'goals') and self.goals:
            for vid, (row, col) in self.goals.items():
                x = col * cell + cell // 2
                y = row * cell + cell // 2
                color = VEHICLE_COLORS.get(vid, '#888888')
                # Draw a star/diamond
                size = 10
                points = [x, y-size, x+size, y, x, y+size, x-size, y]
                self.canvas.create_polygon(points, outline=color, fill='', width=2, tags='goal')
        # Draw vehicles
        if state:
            for vid, (row, col) in state.items():
                x = col * cell + cell // 2
                y = row * cell + cell // 2
                radius = 15
                color = VEHICLE_COLORS.get(vid, '#888888')
                self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                        fill=color, outline='black', width=2, tags='vehicle')
                self.canvas.create_text(x, y, text=vid, font=('Arial', 10, 'bold'), 
                                        fill='white', tags='vehicle')

    def load_schedule(self):
        """Load the schedule matching the selected parameters."""
        nv = self.num_vehicles.get()
        sn = self.sensor_noise.get()
        bd = self.braking_delay.get()
        cl = self.comm_latency.get()
        sch_type = self.schedule_type.get()

        # Find matching entry
        entry = None
        for e in self.data:
            if (e.get('num_vehicles') == nv and
                e.get('configuration', {}).get('sensor_noise') == sn and
                e.get('configuration', {}).get('braking_delay') == bd and
                e.get('configuration', {}).get('comm_latency') == cl):
                entry = e
                break

        if entry is None:
            messagebox.showinfo("Not Found", f"No schedule found for vehicles={nv}, sn={sn}, bd={bd}, cl={cl}")
            return

        # Choose schedule type
        if sch_type == "CBS":
            schedule = entry.get('cbs_schedule')
        else:
            schedule = entry.get('actual_schedule')

        if not schedule:
            messagebox.showerror("Error", "Schedule not found in entry.")
            return

        # Build time steps
        self.time_steps, self.max_time = build_time_positions(schedule)
        self.vehicle_ids = list(schedule.keys())
        self.collisions = entry.get('collisions', 0)
        self.goals = {}
        # Compute goals from last position in schedule
        for vid, steps in schedule.items():
            if steps:
                last = steps[-1]
                self.goals[vid] = tuple(last['position'])

        # Update UI
        self.time_index.set(0)
        self.slider.config(to=self.max_time)
        self.time_label.config(text=f"Time: 0 / {self.max_time}")
        self.coll_label.config(text=f"Collisions: {self.collisions}")
        self.status.config(text=f"Loaded {len(self.vehicle_ids)} vehicles, {self.max_time} steps")

        # Enable controls
        self.prev_btn.config(state='normal')
        self.play_btn.config(state='normal')
        self.next_btn.config(state='normal')

        # Draw initial state
        if self.time_steps:
            self.draw_grid(self.time_steps[0])
        else:
            self.draw_grid()

    def update_frame(self, index):
        """Update canvas to time index."""
        if not self.time_steps:
            return
        if 0 <= index < len(self.time_steps):
            state = self.time_steps[index]
            self.draw_grid(state)
            self.time_label.config(text=f"Time: {index} / {self.max_time}")
            self.slider.set(index)

    def slider_changed(self, val):
        if not self.time_steps:
            return
        idx = int(round(float(val)))
        if idx != self.time_index.get():
            self.time_index.set(idx)
            self.update_frame(idx)

    def next_step(self):
        if not self.time_steps:
            return
        idx = self.time_index.get() + 1
        if idx > self.max_time:
            idx = self.max_time
        self.time_index.set(idx)
        self.update_frame(idx)

    def prev_step(self):
        if not self.time_steps:
            return
        idx = self.time_index.get() - 1
        if idx < 0:
            idx = 0
        self.time_index.set(idx)
        self.update_frame(idx)

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        if not self.time_steps or self.max_time == 0:
            return
        self.is_playing = True
        self.play_btn.config(text="Stop")
        self.prev_btn.config(state='disabled')
        self.next_btn.config(state='disabled')
        self._play_step()

    def _play_step(self):
        if not self.is_playing:
            return
        idx = self.time_index.get() + 1
        if idx > self.max_time:
            self.stop_play()
            return
        self.time_index.set(idx)
        self.update_frame(idx)
        self.after_id = self.root.after(500, self._play_step)

    def stop_play(self):
        self.is_playing = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.play_btn.config(text="Play")
        self.prev_btn.config(state='normal')
        self.next_btn.config(state='normal')

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = DQNDemo(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_play(), root.destroy()))
    root.mainloop()