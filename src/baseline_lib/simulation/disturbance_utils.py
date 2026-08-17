# disturbance_utils.py

def compute_delay_steps(sensor_noise, comm_latency, braking_delay, step_size=0.1):

    def steps(x):
        return int(round(x / step_size))

    return steps(sensor_noise) + steps(comm_latency) + steps(braking_delay)

def _lane_key_and_progress(start, goal):
    if start[0] == goal[0]:
        axis, fixed = "row", start[0]
        direction = 1 if goal[1] > start[1] else -1
        progress = start[1] * direction
    elif start[1] == goal[1]:
        axis, fixed = "col", start[1]
        direction = 1 if goal[0] > start[0] else -1
        progress = start[0] * direction
    else:
        axis, fixed, direction, progress = None, None, 0, 0
    return (axis, fixed, direction), progress

def pick_disturbed_vehicle(vehicle_ids, starts, goals):
    lanes = {}
    for vid in vehicle_ids:
        key, progress = _lane_key_and_progress(starts[vid], goals[vid])
        lanes.setdefault(key, []).append((progress, vid))

    best_leader, best_queue_len = None, 0
    for key, entries in lanes.items():
        if len(entries) >= 2:
            entries.sort(reverse=True)  # highest progress first = first one
            leader_vid = entries[0][1]
            if len(entries) > best_queue_len:
                best_queue_len = len(entries)
                best_leader = leader_vid

    if best_leader is not None:
        return best_leader, True

    return sorted(vehicle_ids)[0], False
