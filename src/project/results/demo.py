# demo

import tkinter as tk
import json
from tkinter import messagebox

STARTS = {'A': (5, 9), 'B': (9, 4), 'C': (4, 0), 'D': (1, 5)}
VEHICLE_IDS = ['A', 'B', 'C', 'D']
VEHICLE_COLORS = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange'}

class IntersectionDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Intersection Simulation Global Plannner CBS + Q-Learning Reinforcement Demo")

        # Load JSON data
        try:
            with open("simulation_results.json", "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "simulation_results.json not found. Run simulation.py first.")
            root.destroy()
            return

        # GUI controls
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        self.sensor_var = tk.IntVar(value=0)
        self.brake_var = tk.IntVar(value=0)
        self.comm_var = tk.IntVar(value=0)

        tk.Checkbutton(control_frame, text="Sensor Noise", variable=self.sensor_var).grid(row=0, column=0, padx=10)
        tk.Checkbutton(control_frame, text="Braking Delay", variable=self.brake_var).grid(row=0, column=1, padx=10)
        tk.Checkbutton(control_frame, text="Communication Latency", variable=self.comm_var).grid(row=0, column=2, padx=10)

        self.run_btn = tk.Button(control_frame, text="Run", command=self.run_simulation)
        self.run_btn.grid(row=0, column=3, padx=20)

        # Canvas for grid
        self.canvas = tk.Canvas(root, width=500, height=500, bg='white')
        self.canvas.pack(pady=10)

        self.status_label = tk.Label(root, text="Ready")
        self.status_label.pack()

        # Animation state
        self.running = False
        self.after_id = None
        self.current_time = 0
        self.positions = {}
        self.vehicle_schedule = {}
        self.max_time = 0

        # Draw grid lines
        self.draw_grid()

    def draw_grid(self):
        """Draw the 10x10 grid."""
        self.canvas.delete("grid")
        cell_size = 50
        for i in range(11):
            x = i * cell_size
            y = i * cell_size
            self.canvas.create_line(x, 0, x, 500, tags="grid", fill="gray")
            self.canvas.create_line(0, y, 500, y, tags="grid", fill="gray")

    # Find the configuration entry matching the disturbance combination.
    def find_config(self, sensor, brake, comm):
        for entry in self.data:
            cfg = entry.get("configuration", {})
            if (cfg.get("sensor_noise") == sensor and
                cfg.get("braking_delay") == brake and
                cfg.get("comm_latency") == comm):
                return entry
        return None

    # Load the schedule for selected disturbances and start animation.
    def run_simulation(self):
        if self.running:
            return

        sensor = self.sensor_var.get()
        brake = self.brake_var.get()
        comm = self.comm_var.get()

        entry = self.find_config(sensor, brake, comm)
        if entry is None:
            messagebox.showinfo("No data", f"No configuration found for S={sensor}, B={brake}, C={comm}")
            return

        schedule_data = entry.get("actual_schedule", {})
        if not schedule_data:
            messagebox.showerror("Error", "No actual_schedule found.")
            return

        # Build a dictionary: vehicle_id -> list of (time, position) sorted by time
        self.vehicle_schedule = {}
        self.max_time = 0
        for vid in VEHICLE_IDS:
            steps = schedule_data.get(vid, [])
            if steps:
                # Sort by time (just in case)
                sorted_steps = sorted(steps, key=lambda x: x["time"])
                # Convert to list of (time, position) where position is a tuple
                time_pos_pairs = [(s["time"], tuple(s["position"])) for s in sorted_steps]
                self.vehicle_schedule[vid] = time_pos_pairs
                # Track the latest time
                last_time = time_pos_pairs[-1][0]
                if last_time > self.max_time:
                    self.max_time = last_time
            else:
                self.vehicle_schedule[vid] = []

        # Initialise positions at time 0 (or fallback to STARTS if no time 0)
        self.positions = {}
        for vid in VEHICLE_IDS:
            pairs = self.vehicle_schedule.get(vid, [])
            pos = None
            for t, p in pairs:
                if t == 0:
                    pos = p
                    break
            if pos is None:
                # fallback to start position
                pos = STARTS.get(vid)
            self.positions[vid] = pos

        self.current_time = 0
        self.running = True
        self.run_btn.config(state="disabled")
        self.status_label.config(text=f"Time {self.current_time}")

        # Draw the initial state
        self.draw_vehicles()

        # Start animation if there are future steps
        if self.max_time > 0:
            self.after_id = self.root.after(1000, self.animate_step)
        else:
            self.running = False
            self.run_btn.config(state="normal")
            self.status_label.config(text="No movement data.")

    # Advance to the next time step.
    def animate_step(self):
        
        if not self.running:
            return

        # Move to the next time
        next_time = self.current_time + 1
        if next_time <= self.max_time:
            # Update each vehicle's position to what it should be at next_time
            for vid in VEHICLE_IDS:
                pairs = self.vehicle_schedule.get(vid, [])
                # Find the position for this time (if any)
                pos = None
                for t, p in pairs:
                    if t == next_time:
                        pos = p
                        break
                if pos is not None:
                    self.positions[vid] = pos
                # else: keep previous position (vehicle waits)
            self.current_time = next_time
            self.draw_vehicles()
            self.status_label.config(text=f"Time {self.current_time} / {self.max_time}")
            # Schedule next step
            self.after_id = self.root.after(1000, self.animate_step)
        else:
            # End of animation
            self.running = False
            self.run_btn.config(state="normal")
            self.status_label.config(text="Simulation finished")

    def draw_vehicles(self):
        """Draw vehicles at their current positions."""
        self.canvas.delete("vehicle")
        cell_size = 50
        for vid, (row, col) in self.positions.items():
            x = col * cell_size + cell_size // 2
            y = row * cell_size + cell_size // 2
            radius = 15
            color = VEHICLE_COLORS.get(vid, 'black')
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill=color, outline='black', width=2, tags="vehicle")
            self.canvas.create_text(x, y, text=vid, font=("Arial", 12, "bold"),
                                    fill='white', tags="vehicle")

    def stop(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)


if __name__ == "__main__":
    root = tk.Tk()
    app = IntersectionDemo(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()