# ui_controller.py

import http.server
import socketserver
import json
import urllib.parse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories and file paths
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Baseline data (CBS results)
BASELINE_JSON = os.path.join(BASE_DIR, '..', 'baseline_lib', 'results', 'intersection_schedules.json')

# Hybrid data (QLearning + CBS results)
HYBRID_JSON = os.path.join(BASE_DIR, '..', 'project', 'simulation_results.json')

print(f"Baseline JSON path: {BASELINE_JSON}")
print(f"Hybrid JSON path: {HYBRID_JSON}")

# Load baseline data
try:
    with open(BASELINE_JSON, 'r') as f:
        BASELINE_DATA = json.load(f)
    print(f"Loaded {len(BASELINE_DATA)} baseline configurations.")
except FileNotFoundError:
    BASELINE_DATA = {}
    print("ERROR: baseline intersection_schedules.json not found. Run schedule_generator.py first.")

# Load hybrid data
try:
    with open(HYBRID_JSON, 'r') as f:
        HYBRID_DATA = json.load(f)
    print(f"Loaded {len(HYBRID_DATA)} hybrid configurations.")
except FileNotFoundError:
    HYBRID_DATA = []
    print("WARNING: hybrid simulation_results.json not found. Hybrid button will not work.")


# Helper functions to convert hybrid data to the common result format
def _empty_result(num_vehicles):
    return {
        "collisions": 0,
        "steps": 0,
        "total_time_steps": 0,
        "vehicles_reached_goal": 0,
        "vehicles_total": num_vehicles,
        "schedules": [],
        "grid_states": [],
        "delay_steps": 0,
        "disturbed_vehicle": None,
        "disturbed_vehicle_has_follower": False,
        "plan_found": None,
        "vehicles_reached_per_step": [],
    }


def _schedule_dict_to_list(schedule_dict):
    """Convert dict {vehicle_id: [steps]} to list of per-vehicle schedules."""
    out = []
    for vid, steps in schedule_dict.items():
        out.append({
            "vehicle_id": vid,
            "steps": [{"time": s["time"], "row": s["row"], "column": s["column"]} for s in steps]
        })
    return out


def _convert_hybrid_entry(entry, num_vehicles):
    """
    Convert a hybrid entry (from simulation_results.json) to the
    standard result format used by the UI.
    """
    actual_schedule = entry.get("actual_schedule", {})
    if not actual_schedule:
        return _empty_result(num_vehicles)

    schedule_list = []
    grid_states = []
    max_time = 0
    time_steps = {}

    for vid, steps in actual_schedule.items():
        vehicle_steps = []
        for step in steps:
            t = step["time"]
            pos = step["position"]  # [row, col]
            row, col = pos[0], pos[1]
            vehicle_steps.append({"time": t, "row": row, "column": col})
            time_steps.setdefault(t, {})[vid] = {"row": row, "column": col}
            if t > max_time:
                max_time = t
        schedule_list.append({"vehicle_id": vid, "steps": vehicle_steps})

    # Build grid_states list sorted by time
    grid_states = []
    for t in range(max_time + 1):
        state = time_steps.get(t, {})
        grid_states.append(state)

    vehicles_reached_goal = len(actual_schedule)
    vehicles_reached_per_step = [[] for _ in range(len(grid_states))]

    return {
        "collisions": entry.get("collisions", 0),
        "steps": max_time,
        "total_time_steps": max_time,
        "vehicles_reached_goal": vehicles_reached_goal,
        "vehicles_total": num_vehicles,
        "schedules": schedule_list,
        "grid_states": grid_states,
        "vehicles_reached_per_step": vehicles_reached_per_step,
        "delay_steps": 0,
        "disturbed_vehicle": None,
        "disturbed_vehicle_has_follower": False,
        "plan_found": True,
    }


# Result lookup functions
def find_baseline_result(params):
    """Find and return baseline (CBS) result for the given parameters."""
    num_vehicles = int(params.get('num_vehicles', 4))
    sensor_noise = float(params.get('sensor_noise', 0.0))
    comm_latency = float(params.get('comm_latency', 0.0))
    braking_delay = float(params.get('braking_delay', 0.0))

    match = None
    for key, entry in BASELINE_DATA.items():
        p = entry.get('parameters', {})
        if (p.get('num_vehicles') == num_vehicles and
                abs(p.get('sensor_noise', 0.0) - sensor_noise) < 0.001 and
                abs(p.get('comm_latency', 0.0) - comm_latency) < 0.001 and
                abs(p.get('braking_delay', 0.0) - braking_delay) < 0.001):
            match = entry
            break

    if match is None:
        print(f"No baseline match for params: num_vehicles={num_vehicles}, "
              f"sensor_noise={sensor_noise}, comm_latency={comm_latency}, braking_delay={braking_delay}")
        return _empty_result(num_vehicles)

    algo_result = match.get('cbs')
    if not algo_result:
        print("No CBS data found for matching baseline config")
        return _empty_result(num_vehicles)

    # Build vehicles_reached_per_step
    grid_states = algo_result.get("grid_states", [])
    vehicles_reached_per_step = []
    reached_goals = set()

    schedule = algo_result.get("schedule", {})
    goals = {}
    for vid, steps in schedule.items():
        if steps:
            goals[vid] = (steps[-1]["row"], steps[-1]["column"])

    for state in grid_states:
        reached_in_step = []
        for vid, pos in state.items():
            if vid in goals and pos["row"] == goals[vid][0] and pos["column"] == goals[vid][1]:
                if vid not in reached_goals:
                    reached_goals.add(vid)
                    reached_in_step.append(vid)
        vehicles_reached_per_step.append(reached_in_step)

    return {
        "collisions": algo_result.get("collisions", 0),
        "steps": algo_result.get("steps", 0),
        "total_time_steps": algo_result.get("steps", 0),
        "vehicles_reached_goal": algo_result.get("vehicles_reached_goal", 0),
        "vehicles_total": algo_result.get("vehicles_total", num_vehicles),
        "schedules": _schedule_dict_to_list(algo_result.get("schedule", {})),
        "grid_states": grid_states,
        "vehicles_reached_per_step": vehicles_reached_per_step,
        "delay_steps": algo_result.get("delay_steps", 0),
        "disturbed_vehicle": algo_result.get("disturbed_vehicle"),
        "disturbed_vehicle_has_follower": algo_result.get("disturbed_vehicle_has_follower", False),
        "plan_found": algo_result.get("plan_found"),
    }


def _map_ui_to_hybrid_value(ui_value):
    """
    Convert UI float (0.0 or positive) to 0/1 for hybrid JSON.
    sensor_noise: 0.0 -> 0, 0.1 -> 1
    braking_delay / comm_latency: 0.0 -> 0, any positive -> 1
    """
    if abs(ui_value) < 0.001:
        return 0
    return 1


def find_hybrid_result(params):
    """Find and return hybrid result for the given parameters."""
    num_vehicles = int(params.get('num_vehicles', 4))
    sensor_noise_ui = float(params.get('sensor_noise', 0.0))
    comm_latency_ui = float(params.get('comm_latency', 0.0))
    braking_delay_ui = float(params.get('braking_delay', 0.0))

    # Map UI values to hybrid integer values
    sensor_noise = _map_ui_to_hybrid_value(sensor_noise_ui)
    comm_latency = _map_ui_to_hybrid_value(comm_latency_ui)
    braking_delay = _map_ui_to_hybrid_value(braking_delay_ui)

    print(f"Hybrid lookup: sensor_noise={sensor_noise} (from {sensor_noise_ui}), "
          f"comm_latency={comm_latency} (from {comm_latency_ui}), "
          f"braking_delay={braking_delay} (from {braking_delay_ui})")

    # Search in HYBRID_DATA - data must match num_vehicles and configuration
    for entry in HYBRID_DATA:
        cfg = entry.get("configuration", {})
        if (entry.get("num_vehicles") == num_vehicles and
            cfg.get("sensor_noise", 0) == sensor_noise and
            cfg.get("braking_delay", 0) == braking_delay and
            cfg.get("comm_latency", 0) == comm_latency):
            # Found matching configuration
            return _convert_hybrid_entry(entry, num_vehicles)

    print(f"No hybrid match for num_vehicles={num_vehicles}, "
          f"sensor_noise={sensor_noise}, comm_latency={comm_latency}, braking_delay={braking_delay}")
    return _empty_result(num_vehicles)


# HTTP Request Handler
class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Serve static files
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            file_path = os.path.join(STATIC_DIR, rel)
            if os.path.isfile(file_path):
                if file_path.endswith(".css"):
                    ctype = "text/css"
                elif file_path.endswith(".js"):
                    ctype = "application/javascript"
                elif file_path.endswith(".json"):
                    ctype = "application/json"
                else:
                    ctype = "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-type", ctype)
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"File not found: {rel}")
                return

        # Serve baseline data file for charts
        if path == "/static/data/intersection_schedules.json" or path == "/intersection_schedules.json":
            if os.path.isfile(BASELINE_JSON):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                with open(BASELINE_JSON, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Data file not found")
                return

        # Serve hybrid data file for charts
        if path == "/simulation_results.json":
            if os.path.isfile(HYBRID_JSON):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                with open(HYBRID_JSON, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Hybrid data file not found")
                return

        # Serve index page
        if path == "/" or path == "/index.html":
            html_path = os.path.join(TEMPLATE_DIR, "index.html")
            if os.path.isfile(html_path):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "index.html not found")
                return

        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            params = json.loads(post_data.decode("utf-8"))
        except Exception:
            params = {}

        if path == "/run_cbs":
            result = find_baseline_result(params)
        elif path == "/run_hybrid":
            result = find_hybrid_result(params)
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))


# Server runner
def run_server(port=8080):
    with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
        print(f"\nAutonomous Navigation UI running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.shutdown()


if __name__ == "__main__":
    run_server()