"""
environment.py

Continuous multi-vehicle intersection environment on a cross-shaped road.
"""

import numpy as np


class ContinuousIntersectionEnv:
    def __init__(self, config, cbs_schedule=None, noise_level=0.0):
        self.cfg = config
        self.cbs_schedule = cbs_schedule
        self.noise_level = noise_level

        self.num_vehicles = config.NUM_VEHICLES
        self.max_vehicles = config.MAX_VEHICLES
        self.max_steps = config.MAX_STEPS
        self.grid = config.GRID_SIZE
        self.dt = 1.0

        self.positions = None
        self.velocities = None
        self.goals = None
        self.step_count = 0
        self.done = False
        self.terminated = False
        self.prev_distances = None
        self.reached = None
        self.slot_ids = list(range(self.num_vehicles))

    
    def reset(self, slot=None):
        if self.num_vehicles == 1 and getattr(self.cfg, "SOLO_RANDOM_SLOT", False):
            k = int(np.random.randint(self.max_vehicles)) if slot is None else int(slot)
            self.slot_ids = [k]
        else:
            self.slot_ids = list(range(self.num_vehicles))

        starts = [self.cfg.START_POSITIONS[k] for k in self.slot_ids]
        goals = [self.cfg.GOAL_POSITIONS[k] for k in self.slot_ids]
        self.positions = np.array(starts, dtype=np.float32)
        self.velocities = np.zeros((self.num_vehicles, 2), dtype=np.float32)
        self.goals = np.array(goals, dtype=np.float32)
        self.step_count = 0
        self.done = False
        self.terminated = False
        self.reached = np.zeros(self.num_vehicles, dtype=bool)

        self.prev_distances = np.linalg.norm(self.goals - self.positions, axis=1)
        return self._get_all_states()

    
    @staticmethod
    def _nearness(d):
        return float(np.clip(2.0 - d, 0.0, 2.0))

    
    # ROAD / LANE helpers
    
    def _clip_to_road(self, new_positions, velocities):
        """Constrain vehicles to the road network."""
        rmin, rmax = self.cfg.ROAD_MIN, self.cfg.ROAD_MAX
        clipped = new_positions.copy()
        for i in range(self.num_vehicles):
            x, y = new_positions[i]
            on_h = rmin - 0.5 <= y <= rmax + 0.5
            on_v = rmin - 0.5 <= x <= rmax + 0.5

            if on_h or on_v:
                clipped[i, 0] = np.clip(x, 0.0, self.grid)
                clipped[i, 1] = np.clip(y, 0.0, self.grid)
            else:
                if abs(x - self.grid / 2) < abs(y - self.grid / 2):
                    clipped[i, 1] = rmax if y > rmax else rmin
                    clipped[i, 0] = np.clip(x, 0.0, self.grid)
                    velocities[i, 1] = 0.0
                else:
                    clipped[i, 0] = rmax if x > rmax else rmin
                    clipped[i, 1] = np.clip(y, 0.0, self.grid)
                    velocities[i, 0] = 0.0
        return clipped

    def _lane_deviation(self, i):
        slot = self.slot_ids[i]
        axis, center, _, _ = self.cfg.SLOT_DEFS[slot]
        x, y = self.positions[i]
        in_inter = (self.cfg.ROAD_MIN <= x <= self.cfg.ROAD_MAX
                    and self.cfg.ROAD_MIN <= y <= self.cfg.ROAD_MAX)
        if in_inter:
            return 0.0
        return abs(y - center) if axis == 0 else abs(x - center)

    def _off_road(self, i):
        x, y = self.positions[i]
        on_h = self.cfg.ROAD_MIN - 0.5 <= y <= self.cfg.ROAD_MAX + 0.5
        on_v = self.cfg.ROAD_MIN - 0.5 <= x <= self.cfg.ROAD_MAX + 0.5
        return not (on_h or on_v)

    
    def step(self, actions):
        actions = np.clip(actions, self.cfg.ACTION_LOW, self.cfg.ACTION_HIGH)

        active = ~self.reached
        actions[~active] = 0.0

        # Project each vehicle's action onto its lane direction (NEW)
        for i in range(self.num_vehicles):
            slot = self.slot_ids[i]
            axis = self.cfg.SLOT_DEFS[slot][0]
            if axis == 0:
                actions[i, 1] = 0.0
            else:
                actions[i, 0] = 0.0

        # dynamics
        self.velocities = self.velocities + actions * self.dt

        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        too_fast = speeds[:, 0] > self.cfg.MAX_SPEED
        if np.any(too_fast):
            self.velocities[too_fast] *= (self.cfg.MAX_SPEED / speeds[too_fast])

        if self.noise_level > 0:
            self.velocities += np.random.normal(
                0.0, self.noise_level * 0.1, size=self.velocities.shape
            ).astype(np.float32)

        # Zero lateral velocity so cars stay on their lane
        for i in range(self.num_vehicles):
            slot = self.slot_ids[i]
            axis = self.cfg.SLOT_DEFS[slot][0]
            if axis == 0:
                self.velocities[i, 1] = 0.0
            else:
                self.velocities[i, 0] = 0.0

        self.velocities[~active] = 0.0

        # move + constrain to road network
        proposed = self.positions + self.velocities * self.dt
        proposed = np.clip(proposed, 0.0, self.grid)
        self.positions = self._clip_to_road(proposed, self.velocities)

        # rewards
        rewards = np.zeros(self.num_vehicles, dtype=np.float32)
        collisions = self._check_collisions(active)
        distances = np.linalg.norm(self.goals - self.positions, axis=1)
        newly_reached = active & (distances < self.cfg.GOAL_RADIUS)

        for i in range(self.num_vehicles):
            if not active[i]:
                continue

            r = self.cfg.REWARD_STEP_PENALTY

            delta_d = self.prev_distances[i] - distances[i]
            r += self.cfg.REWARD_PROGRESS * delta_d

            r += self.cfg.REWARD_PROGRESS_BONUS * (
                self._nearness(distances[i]) - self._nearness(self.prev_distances[i])
            )

            if delta_d < 0.01 and not newly_reached[i]:
                r += getattr(self.cfg, "REWARD_NO_PROGRESS_PENALTY", 0.0)

            if newly_reached[i]:
                r += self.cfg.REWARD_GOAL

            if i in collisions:
                r += self.cfg.REWARD_COLLISION

            # Lane discipline
            dev = self._lane_deviation(i)
            r += self.cfg.LANE_DEVIATION_PENALTY * max(
                0.0, dev - self.cfg.LANE_HALF_WIDTH
            )
            if self._off_road(i):
                r += self.cfg.OFF_ROAD_PENALTY

            # Proximity penalty
            for j in range(self.num_vehicles):
                if j == i or not active[j]:
                    continue
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < self.cfg.REWARD_SAFE_DISTANCE:
                    r += self.cfg.REWARD_PROXIMITY_PENALTY * (
                        1.0 - d / self.cfg.REWARD_SAFE_DISTANCE
                    )

            # CBS schedule adherence
            if self.cbs_schedule is not None:
                sched_pos = self._get_scheduled_position(i)
                if sched_pos is not None:
                    dev_s = np.linalg.norm(self.positions[i] - sched_pos)
                    if dev_s < 0.5:
                        r += self.cfg.REWARD_SCHEDULE_BONUS
                    else:
                        r += self.cfg.REWARD_DEVIATION_PENALTY * dev_s

            rewards[i] = r

        self.reached = self.reached | newly_reached
        self.prev_distances = distances
        self.step_count += 1

        all_reached = bool(self.reached.all())
        crashed = len(collisions) > 0
        self.terminated = all_reached or (
            crashed and getattr(self.cfg, "TERMINATE_ON_COLLISION", True)
        )
        truncated = (self.step_count >= self.max_steps) and not self.terminated
        self.done = self.terminated or truncated

        info = {
            "collisions": collisions,
            "distances": distances,
            "all_reached": all_reached,
            "terminated": self.terminated,
            "active": active,
            "newly_reached": newly_reached,
            "reached": self.reached.copy(),
        }
        return self._get_all_states(), rewards, self.done, info

    
    def _get_scheduled_position(self, vehicle_id):
        if self.cbs_schedule is None:
            return None
        traj = self.cbs_schedule.get(self.slot_ids[vehicle_id])
        if traj is None or len(traj) == 0:
            return None
        idx = min(self.step_count, len(traj) - 1)
        return np.array(traj[idx], dtype=np.float32)

    def _check_collisions(self, active):
        collided = set()
        r = self.cfg.VEHICLE_RADIUS
        for i in range(self.num_vehicles):
            for j in range(i + 1, self.num_vehicles):
                if not (active[i] and active[j]):
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < 2 * r:
                    collided.add(i)
                    collided.add(j)
        return collided

    def _get_local_fov(self, vehicle_id):
        fov = np.zeros((self.cfg.FOV_SIZE, self.cfg.FOV_SIZE), dtype=np.float32)
        cx, cy = self.positions[vehicle_id]
        half = self.cfg.FOV_SIZE // 2

        for j in range(self.num_vehicles):
            if j == vehicle_id or self.reached[j]:
                continue
            dx = self.positions[j][0] - cx
            dy = self.positions[j][1] - cy
            gx = int(np.round(dx)) + half
            gy = int(np.round(dy)) + half
            if 0 <= gx < self.cfg.FOV_SIZE and 0 <= gy < self.cfg.FOV_SIZE:
                fov[gy, gx] = 1.0

        if self.noise_level > 0:
            fov += np.random.normal(0.0, self.noise_level, size=fov.shape).astype(np.float32)
            fov = np.clip(fov, 0.0, 1.0)
        return fov.flatten()

    def _get_state(self, vehicle_id):
        pos = self.positions[vehicle_id]
        vel = self.velocities[vehicle_id]
        goal = self.goals[vehicle_id]
        sched = self._get_scheduled_position(vehicle_id)
        if sched is None:
            sched = pos
        fov = self._get_local_fov(vehicle_id)

        veh_id = np.zeros(self.max_vehicles, dtype=np.float32)
        veh_id[self.slot_ids[vehicle_id]] = 1.0

        state = np.concatenate([
            pos / self.grid,
            vel / self.cfg.MAX_SPEED,
            goal / self.grid,
            sched / self.grid,
            fov,
            veh_id,
        ]).astype(np.float32)
        return state

    def _get_all_states(self):
        return np.stack([self._get_state(i) for i in range(self.num_vehicles)])