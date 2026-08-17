# scenario_generator.py

import numpy as np

def generate_scenario(num_vehicles, width=10, height=10, seed=None):

    if seed is not None:
        np.random.seed(seed)

    base_ids = ["A", "B", "C", "D"]

    sides = [
        # East edge: lane = row 5, depth = col (9 -> inward is -1)
        {
            "lane_axis": "row",
            "lane_value": 5,
            "start_depth_base": 9,
            "start_depth_dir": -1,
            "goal_depth_base": 0,
        },
        # South edge: lane = col 4, depth = row (9 -> inward is -1)
        {
            "lane_axis": "col",
            "lane_value": 4,
            "start_depth_base": 9,
            "start_depth_dir": -1,
            "goal_depth_base": 0,
        },
        # West edge: lane = row 4, depth = col (0 -> inward is +1)
        {
            "lane_axis": "row",
            "lane_value": 4,
            "start_depth_base": 0,
            "start_depth_dir": 1,
            "goal_depth_base": 9,
        },
        # North edge: lane = col 5, depth = row (1 -> inward is +1)
        {
            "lane_axis": "col",
            "lane_value": 5,
            "start_depth_base": 1,
            "start_depth_dir": 1,
            "goal_depth_base": 9,
        },
    ]

    # adding new vehicles moves the old vehicles inwards
    occupants = [["A"], ["B"], ["C"], ["D"]]

    if num_vehicles <= 4:
        ids = base_ids[:num_vehicles]
        occupants = [[vid] if vid in ids else [] for vid in base_ids]
    else:
        extra_count = num_vehicles - 4
        next_id_ord = ord("E")
        side_idx = 0
        for _ in range(extra_count):
            vid = chr(next_id_ord)
            next_id_ord += 1

            occupants[side_idx].insert(0, vid)
            side_idx = (side_idx + 1) % 4

    starts = {}
    goals = {}

    for side, queue in zip(sides, occupants):
        if not queue:
            continue

        lane_axis = side["lane_axis"]
        lane_value = side["lane_value"]
        start_base = side["start_depth_base"]
        start_dir = side["start_depth_dir"]
        goal_base = side["goal_depth_base"]
        goal_dir = -start_dir

        for offset, vid in enumerate(queue):
            start_depth = start_base + offset * start_dir
            goal_depth = goal_base + offset * goal_dir

            if lane_axis == "row":
                starts[vid] = (lane_value, start_depth)
                goals[vid] = (lane_value, goal_depth)
            else:
                starts[vid] = (start_depth, lane_value)
                goals[vid] = (goal_depth, lane_value)

    return starts, goals


if __name__ == "__main__":
    for n in [4, 5, 6, 7, 8]:
        s, g = generate_scenario(n)
        print(f"num_vehicles={n}")
        for vid in sorted(s):
            print(f"  {vid}: start={s[vid]} goal={g[vid]}")
        print()
