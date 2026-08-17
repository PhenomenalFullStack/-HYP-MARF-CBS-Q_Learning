# grid_env.py

import numpy as np

INTERSECTION_CELLS = {(4, 4), (4, 5), (5, 4), (5, 5)}

class GridEnv:

    # set up the grid size and disturbance parameters.
    def __init__(
        self,
        width=10,
        height=10,
        braking_delay_prob=0.0,
        sensor_noise=0.0,
        comm_latency=0.0,
    ):
        self.width = width
        self.height = height
        self.braking_delay_prob = braking_delay_prob
        self.sensor_noise = sensor_noise
        self.comm_latency = comm_latency
        self.vehicles = {}
        self.time_step = 0
        self.collisions = 0
        self.steps = 0

    # put the vehicle on the grid with start, goal, and delay.
    def add_vehicle(self, vehicle_id, start, goal, delay_steps=0):
        self.vehicles[vehicle_id] = {
            "start": start,
            "goal": goal,
            "current": start,
            "delay_steps": delay_steps,
            "reached": False,
        }

    # clear all vehicels back to starting.
    def reset(self):
        self.time_step = 0
        self.collisions = 0
        self.steps = 0
        for vehicle in self.vehicles.values():
            vehicle["current"] = vehicle["start"]
            vehicle["reached"] = False

    # 3x3 view per vehicle with optional sensor noise.
    def get_observations(self):
        obs = {}
        for vid, vehicle in self.vehicles.items():
            row, col = vehicle["current"]
            if self.sensor_noise > 0:
                nr = int(round(row + np.random.normal(0, self.sensor_noise)))
                nc = int(round(col + np.random.normal(0, self.sensor_noise)))
                nr = max(0, min(self.height - 1, nr))
                nc = max(0, min(self.width - 1, nc))
            else:
                nr, nc = row, col
            grid_obs = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = nr + dr, nc + dc
                    occupied = 0
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        for other_vid, other_v in self.vehicles.items():
                            if other_vid != vid and other_v["current"] == (rr, cc):
                                occupied = 1
                                break
                    grid_obs.append(occupied)
            obs[vid] = np.array(grid_obs)
        return obs

    # check if two vihicles share a cell.
    def _count_collisions(self):
        positions = {}
        for vid, v in self.vehicles.items():
            if v["reached"]:
                continue
            positions.setdefault(v["current"], []).append(vid)
        return sum(len(occ) - 1 for occ in positions.values() if len(occ) > 1)

    # move all vehicles and records collisions.
    def apply_positions(self, new_positions):
        for vid, pos in new_positions.items():
            self.vehicles[vid]["current"] = pos
            if pos == self.vehicles[vid]["goal"]:
                self.vehicles[vid]["reached"] = True
        collisions_this_step = self._count_collisions()
        self.collisions += collisions_this_step
        self.steps += 1
        self.time_step += 1
        done = all(v["reached"] for v in self.vehicles.values())
        return done, collisions_this_step

    # move vehicles according to cbs schedules, applies braking delay.
    def step_from_schedule(self, schedules):
        next_t = self.time_step + 1
        new_positions = {}
        for vid, steps_list in schedules.items():
            cur = self.vehicles[vid]["current"]
            sched_pos = cur
            for t, p in steps_list:
                if t == next_t:
                    sched_pos = p
                    break
                elif t > next_t:
                    break
                sched_pos = p

            is_disturbed_vehicle = self.vehicles[vid].get("delay_steps", 0) > 0
            if (
                is_disturbed_vehicle
                and self.braking_delay_prob > 0
                and np.random.rand() < self.braking_delay_prob
            ):
                new_positions[vid] = cur
            else:
                new_positions[vid] = sched_pos
        return self.apply_positions(new_positions)
