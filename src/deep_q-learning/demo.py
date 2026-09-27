"""
demo.py

Pygame visualisation of the DQN simulation results.
"""

import os
import sys
import json
import argparse

import pygame

# Constants
GRID_SIZE = 10

VEHICLE_COLORS = {
    'A': (220, 30, 40),
    'B': (40, 90, 220),
    'C': (40, 170, 60),
    'D': (245, 130, 40),
    'E': (150, 40, 200),
    'F': (230, 50, 210),
    'G': (60, 60, 60),
    'H': (40, 200, 200),
}

ROAD_MIN = 4
ROAD_MAX = 6

# Colours - matching the DDPG demo
COL_FORMATION       = (248, 243, 222)
COL_GRID            = (222, 216, 200)
COL_GRID_BORDER     = (180, 175, 160)
COL_ROAD            = (222, 217, 202)
COL_ROAD_EDGE       = (150, 145, 125)
COL_LANE_WHITE      = (255, 255, 255)
COL_LANE_YELLOW     = (235, 195, 30)
COL_INTERSECTION_BG = (175, 210, 245, 160)
COL_INTERSECTION_ED = (40, 90, 180)
COL_PANEL           = (250, 250, 250)
COL_PANEL_EDGE      = (200, 200, 200)
COL_TEXT            = (30, 30, 30)
COL_TEXT_DIM        = (120, 120, 120)
COL_START           = (100, 180, 100)
COL_GOAL            = (200, 100, 100)
COL_BTN_BG          = (235, 235, 235)
COL_BTN_HOVER       = (215, 215, 215)
COL_BTN_EDGE        = (150, 150, 150)
COL_SLIDER_TRACK    = (215, 215, 215)
COL_SLIDER_FILL     = (100, 150, 240)
COL_SLIDER_HANDLE   = (60, 100, 200)


# build_time_positions - unchanged from the original tkinter demo
def build_time_positions(schedule):
    all_times = set()
    for vid, steps in schedule.items():
        for step in steps:
            all_times.add(step['time'])
    if not all_times:
        return [], 0
    max_time = max(all_times)

    pos_by_vid_time = {}
    for vid, steps in schedule.items():
        pos_map = {}
        for step in steps:
            pos_map[step['time']] = tuple(step['position'])
        last_pos = None
        filled = {}
        for t in range(max_time + 1):
            if t in pos_map:
                last_pos = pos_map[t]
            filled[t] = last_pos
        pos_by_vid_time[vid] = filled

    time_steps = []
    for t in range(max_time + 1):
        state = {}
        for vid in schedule.keys():
            pos = pos_by_vid_time[vid].get(t)
            if pos is not None:
                state[vid] = pos
        time_steps.append(state)
    return time_steps, max_time


# Display init
def init_display(fullscreen):
    pygame.init()
    pygame.display.set_caption("DQN Intersection Demo")
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        info = pygame.display.Info()
        W = min(1400, int(info.current_w * 0.92))
        H = min(900, int(info.current_h * 0.92))
        screen = pygame.display.set_mode((W, H))
    return screen, screen.get_width(), screen.get_height()


# Layout
class Layout:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.panel_w = min(380, max(280, int(W * 0.26)))
        self.panel_x = W - self.panel_w

        canvas_w = W - self.panel_w
        pad = 40
        self.draw_size = min(canvas_w - 2 * pad, H - 2 * pad)
        self.draw_x = (canvas_w - self.draw_size) // 2
        self.draw_y = (H - self.draw_size) // 2
        self.cell = self.draw_size / GRID_SIZE

    def cell_center(self, row, col):
        px = self.draw_x + (col + 0.5) * self.cell
        py = self.draw_y + (row + 0.5) * self.cell
        return int(px), int(py)

    def y_of_row(self, row):
        return int(self.draw_y + row * self.cell)

    def x_of_col(self, col):
        return int(self.draw_x + col * self.cell)


# Car sprite
def make_car_sprite(length_px, width_px, colour):
    length_px = max(10, int(length_px))
    width_px = max(6, int(width_px))
    surf = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
    pygame.draw.rect(surf, colour, (0, 0, length_px, width_px),
                     border_radius=max(2, width_px // 4))
    dark = (max(colour[0] - 50, 0),
            max(colour[1] - 50, 0),
            max(colour[2] - 50, 0))
    cab_len = int(length_px * 0.45)
    cab_wid = int(width_px * 0.8)
    cab_x = (length_px - cab_len) // 2
    cab_y = (width_px - cab_wid) // 2
    pygame.draw.rect(surf, dark, (cab_x, cab_y, cab_len, cab_wid),
                     border_radius=max(2, width_px // 6))
    ws_w = max(2, int(length_px * 0.10))
    ws_x = int(length_px * 0.72)
    ws_h = int(width_px * 0.7)
    ws_y = (width_px - ws_h) // 2
    pygame.draw.rect(surf, (180, 220, 245), (ws_x, ws_y, ws_w, ws_h))
    hl = max(2, int(width_px * 0.18))
    pygame.draw.rect(surf, (255, 250, 200), (length_px - hl - 1, 1, hl, hl))
    pygame.draw.rect(surf, (255, 250, 200),
                     (length_px - hl - 1, width_px - hl - 1, hl, hl))
    tl = max(2, int(width_px * 0.18))
    pygame.draw.rect(surf, (255, 90, 90), (1, 1, tl, tl))
    pygame.draw.rect(surf, (255, 90, 90),
                     (1, width_px - tl - 1, tl, tl))
    return surf


# Main app
class DQNDemo:
    def __init__(self, data, screen, W, H):
        self.data = data
        self.screen = screen
        self.W = W
        self.H = H
        self.layout = Layout(W, H)

        # Parameters
        self.num_vehicles   = 4
        self.sensor_noise   = 0
        self.braking_delay  = 0
        self.comm_latency   = 0
        self.schedule_type  = "DQN (actual)"

        # Loaded schedule
        self.time_steps = []
        self.max_time = 0
        self.time_index = 0
        self.vehicle_ids = []
        self.starts = {}
        self.goals = {}
        self.collisions = 0
        self.status_msg = "Select parameters and click Load."
        self._headings = {}      

        # Playback
        self.playing = False
        self.animation_speed = 4.0
        self.last_time = 0
        self.accumulator = 0.0
        self.clock = pygame.time.Clock()

        # Fonts
        self.f    = pygame.font.SysFont("Segoe UI", 14)
        self.f_md = pygame.font.SysFont("Segoe UI", 16)
        self.f_lg = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.f_xl = pygame.font.SysFont("Segoe UI", 22, bold=True)

        # Sprites
        self._sprite_cache = {}

        # Build interactive controls
        self._build_controls()

    def _build_controls(self):
        px = self.layout.panel_x + 20
        pw = self.layout.panel_w - 40
        y = 130

        # Parameter cycle buttons (right-aligned in panel)
        self.ctrl_veh_minus = pygame.Rect(px + pw - 100, y, 30, 24)
        self.ctrl_veh_plus  = pygame.Rect(px + pw - 30, y, 30, 24)

        y += 36
        self.ctrl_noise   = pygame.Rect(px + pw - 100, y, 100, 24)
        y += 36
        self.ctrl_brake   = pygame.Rect(px + pw - 100, y, 100, 24)
        y += 36
        self.ctrl_comm    = pygame.Rect(px + pw - 100, y, 100, 24)
        y += 36
        self.ctrl_sched   = pygame.Rect(px + pw - 150, y, 150, 24)
        y += 44

        # Load button
        self.ctrl_load = pygame.Rect(px, y, pw, 34)

        # Bottom-anchored playback controls
        bottom_margin = 30
        btn_h = 36
        btn_gap = 8

        # Bottom row: Next / Reset
        btn2_y = self.H - bottom_margin - btn_h
        # Top row: Prev / Play
        btn1_y = btn2_y - btn_gap - btn_h

        # Slider sits above the buttons
        slider_y = btn1_y - 40

        # Labels
        self.time_label_y  = slider_y - 34 - 30   # higher
        self.speed_label_y = slider_y - 30        # just above slider

        self.slider_rect = pygame.Rect(px, slider_y, pw, 14)
        self.slider_handle = pygame.Rect(0, 0, 18, 22)
        self.dragging_slider = False

        btn_w = (pw - 10) // 2
        self.btn_prev  = pygame.Rect(px, btn1_y, btn_w, btn_h)
        self.btn_play  = pygame.Rect(px + btn_w + 10, btn1_y, btn_w, btn_h)
        self.btn_next  = pygame.Rect(px, btn2_y, btn_w, btn_h)
        self.btn_reset = pygame.Rect(px + btn_w + 10, btn2_y, btn_w, btn_h)

    # Main loop
    def run(self):
        running = True
        self.last_time = pygame.time.get_ticks()
        while running:
            now = pygame.time.get_ticks()
            dt = (now - self.last_time) / 1000.0
            self.last_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._toggle_play()
                    elif event.key == pygame.K_LEFT:
                        self._prev()
                    elif event.key == pygame.K_RIGHT:
                        self._next()
                    elif event.key == pygame.K_r:
                        self._reset()
                else:
                    self._handle_event(event)

            if self.playing and self.time_steps:
                self.accumulator += dt
                interval = 1.0 / max(self.animation_speed, 0.001)
                while self.accumulator >= interval:
                    self.accumulator -= interval
                    if self.time_index < self.max_time:
                        self.time_index += 1
                    else:
                        self.playing = False
                        self.accumulator = 0.0
                        break

            self._render()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

    def _handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            p = event.pos
            if self.ctrl_veh_minus.collidepoint(p):
                self.num_vehicles = max(1, self.num_vehicles - 1)
            elif self.ctrl_veh_plus.collidepoint(p):
                self.num_vehicles = min(8, self.num_vehicles + 1)
            elif self.ctrl_noise.collidepoint(p):
                self.sensor_noise = 1 - self.sensor_noise
            elif self.ctrl_brake.collidepoint(p):
                self.braking_delay = 1 - self.braking_delay
            elif self.ctrl_comm.collidepoint(p):
                self.comm_latency = 1 - self.comm_latency
            elif self.ctrl_sched.collidepoint(p):
                self.schedule_type = ("CBS" if self.schedule_type == "DQN (actual)"
                                      else "DQN (actual)")
            elif self.ctrl_load.collidepoint(p):
                self._load_schedule()
            elif self.btn_prev.collidepoint(p):
                self._prev()
            elif self.btn_play.collidepoint(p):
                self._toggle_play()
            elif self.btn_next.collidepoint(p):
                self._next()
            elif self.btn_reset.collidepoint(p):
                self._reset()
            elif self.slider_rect.inflate(0, 20).collidepoint(p):
                self.dragging_slider = True
                self._update_speed(p)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_slider = False
        elif event.type == pygame.MOUSEMOTION and self.dragging_slider:
            self._update_speed(event.pos)

    def _update_speed(self, pos):
        x = max(self.slider_rect.left, min(pos[0], self.slider_rect.right))
        frac = (x - self.slider_rect.left) / self.slider_rect.width
        self.animation_speed = 1.0 + frac * 29.0

    def _toggle_play(self):
        if not self.time_steps:
            return
        if not self.playing and self.time_index >= self.max_time:
            self.time_index = 0
            self.accumulator = 0.0
        self.playing = not self.playing

    def _next(self):
        if self.time_steps and self.time_index < self.max_time:
            self.time_index += 1

    def _prev(self):
        if self.time_steps and self.time_index > 0:
            self.time_index -= 1

    def _reset(self):
        self.playing = False
        self.time_index = 0
        self.accumulator = 0.0

    # Schedule loading - logic unchanged from the tkinter version
    def _load_schedule(self):
        nv = self.num_vehicles
        sn = self.sensor_noise
        bd = self.braking_delay
        cl = self.comm_latency
        sch_type = self.schedule_type

        entry = None
        for e in self.data:
            if (e.get('num_vehicles') == nv and
                e.get('configuration', {}).get('sensor_noise') == sn and
                e.get('configuration', {}).get('braking_delay') == bd and
                e.get('configuration', {}).get('comm_latency') == cl):
                entry = e
                break

        if entry is None:
            self.status_msg = (f"No schedule for v={nv}, sn={sn}, "
                               f"bd={bd}, cl={cl}")
            return

        if sch_type == "CBS":
            schedule = entry.get('cbs_schedule')
        else:
            schedule = entry.get('actual_schedule')

        if not schedule:
            self.status_msg = "Schedule not found in entry."
            return

        self.time_steps, self.max_time = build_time_positions(schedule)
        self.vehicle_ids = list(schedule.keys())
        self.collisions = entry.get('collisions', 0)

        # Extract start (first step) and goal (last step) positions
        self.starts = {}
        self.goals = {}
        for vid, steps in schedule.items():
            if steps:
                self.starts[vid] = tuple(steps[0]['position'])
                self.goals[vid] = tuple(steps[-1]['position'])

        self._headings = {}
        for vid, steps in schedule.items():
            sorted_steps = sorted(steps, key=lambda x: x['time'])

            pos_by_time = {s['time']: tuple(s['position']) for s in sorted_steps}
            positions = []
            last_pos = None
            for t in range(self.max_time + 1):
                if t in pos_by_time:
                    last_pos = pos_by_time[t]
                positions.append(last_pos)

            goal = tuple(sorted_steps[-1]['position']) if sorted_steps else (0, 0)

            headings = []
            last_dir = None
            for t in range(self.max_time + 1):
                pos = positions[t]
                if pos is None:
                    headings.append((0, 1))
                    continue

                # If the vehicle moved since the previous frame, update
                if t > 0 and positions[t - 1] is not None:
                    dr = pos[0] - positions[t - 1][0]
                    dc = pos[1] - positions[t - 1][1]
                    if dr != 0 or dc != 0:
                        last_dir = (dr, dc)

                # If we haven't seen any movement yet, aim toward the goal
                if last_dir is None:
                    gdr = goal[0] - pos[0]
                    gdc = goal[1] - pos[1]
                    if gdr != 0 or gdc != 0:
                        last_dir = (
                            1 if gdr > 0 else -1 if gdr < 0 else 0,
                            1 if gdc > 0 else -1 if gdc < 0 else 0,
                        )

                headings.append(last_dir if last_dir else (0, 1))

            self._headings[vid] = headings

        self.time_index = 0
        self.status_msg = (f"Loaded {len(self.vehicle_ids)} vehicles, "
                           f"{self.max_time} steps")

    # Rendering
    def _render(self):
        self.screen.fill(COL_FORMATION)
        self._draw_grid()
        self._draw_road()
        self._draw_intersection()
        self._draw_markers()
        self._draw_vehicles()
        self._draw_panel()
        pygame.display.flip()

    def _draw_grid(self):
        L = self.layout
        for i in range(GRID_SIZE + 1):
            x = L.x_of_col(i)
            pygame.draw.line(self.screen, COL_GRID,
                             (x, L.y_of_row(0)),
                             (x, L.y_of_row(GRID_SIZE)), 1)
            y = L.y_of_row(i)
            pygame.draw.line(self.screen, COL_GRID,
                             (L.x_of_col(0), y),
                             (L.x_of_col(GRID_SIZE), y), 1)
        # Outer border
        pygame.draw.rect(self.screen, COL_GRID_BORDER,
                         (L.x_of_col(0), L.y_of_row(0),
                          L.draw_size, L.draw_size), 1)

    def _draw_road(self):
        L = self.layout
        g = GRID_SIZE

        # Road surface (horizontal band)
        x1 = L.x_of_col(0); y1 = L.y_of_row(ROAD_MIN)
        x2 = L.x_of_col(g); y2 = L.y_of_row(ROAD_MAX)
        pygame.draw.rect(self.screen, COL_ROAD, (x1, y1, x2 - x1, y2 - y1))

        # Road surface (vertical band)
        x1 = L.x_of_col(ROAD_MIN); y1 = L.y_of_row(0)
        x2 = L.x_of_col(ROAD_MAX); y2 = L.y_of_row(g)
        pygame.draw.rect(self.screen, COL_ROAD, (x1, y1, x2 - x1, y2 - y1))

        # Road edges (dark grey)
        for r in (ROAD_MIN, ROAD_MAX):
            y = L.y_of_row(r)
            pygame.draw.line(self.screen, COL_ROAD_EDGE,
                             (L.x_of_col(0), y),
                             (L.x_of_col(ROAD_MIN), y), 2)
            pygame.draw.line(self.screen, COL_ROAD_EDGE,
                             (L.x_of_col(ROAD_MAX), y),
                             (L.x_of_col(g), y), 2)
        for c in (ROAD_MIN, ROAD_MAX):
            x = L.x_of_col(c)
            pygame.draw.line(self.screen, COL_ROAD_EDGE,
                             (x, L.y_of_row(0)),
                             (x, L.y_of_row(ROAD_MIN)), 2)
            pygame.draw.line(self.screen, COL_ROAD_EDGE,
                             (x, L.y_of_row(ROAD_MAX)),
                             (x, L.y_of_row(g)), 2)

        # Yellow dashed centre line - horizontal road
        mid_y = L.y_of_row((ROAD_MIN + ROAD_MAX) / 2)
        self._dash((L.x_of_col(0), mid_y),
                   (L.x_of_col(ROAD_MIN), mid_y),
                   COL_LANE_YELLOW, 3, 12, 8)
        self._dash((L.x_of_col(ROAD_MAX), mid_y),
                   (L.x_of_col(g), mid_y),
                   COL_LANE_YELLOW, 3, 12, 8)

        # Yellow dashed centre line - vertical road
        mid_x = L.x_of_col((ROAD_MIN + ROAD_MAX) / 2)
        self._dash((mid_x, L.y_of_row(0)),
                   (mid_x, L.y_of_row(ROAD_MIN)),
                   COL_LANE_YELLOW, 3, 12, 8)
        self._dash((mid_x, L.y_of_row(ROAD_MAX)),
                   (mid_x, L.y_of_row(g)),
                   COL_LANE_YELLOW, 3, 12, 8)

        # White inner edge lines (each side of the yellow line)
        off = L.cell * 0.45
        for sgn in (+1, -1):
            y = int(mid_y + sgn * off)
            pygame.draw.line(self.screen, COL_LANE_WHITE,
                             (L.x_of_col(0), y),
                             (L.x_of_col(ROAD_MIN), y), 2)
            pygame.draw.line(self.screen, COL_LANE_WHITE,
                             (L.x_of_col(ROAD_MAX), y),
                             (L.x_of_col(g), y), 2)
            x = int(mid_x + sgn * off)
            pygame.draw.line(self.screen, COL_LANE_WHITE,
                             (x, L.y_of_row(0)),
                             (x, L.y_of_row(ROAD_MIN)), 2)
            pygame.draw.line(self.screen, COL_LANE_WHITE,
                             (x, L.y_of_row(ROAD_MAX)),
                             (x, L.y_of_row(g)), 2)

    def _dash(self, p1, p2, colour, width, dash, gap):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist == 0:
            return
        ux, uy = dx / dist, dy / dist
        pos = 0.0
        while pos < dist:
            sx = x1 + ux * pos;  sy = y1 + uy * pos
            ex = x1 + ux * min(pos + dash, dist)
            ey = y1 + uy * min(pos + dash, dist)
            pygame.draw.line(self.screen, colour, (sx, sy), (ex, ey), width)
            pos += dash + gap

    def _draw_intersection(self):
        L = self.layout
        x1 = L.x_of_col(ROAD_MIN); y1 = L.y_of_row(ROAD_MIN)
        x2 = L.x_of_col(ROAD_MAX); y2 = L.y_of_row(ROAD_MAX)
        rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)

        surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        surf.fill(COL_INTERSECTION_BG)
        self.screen.blit(surf, rect.topleft)
        pygame.draw.rect(self.screen, COL_INTERSECTION_ED, rect, 3)

        label = self.f_lg.render("Coordination Zone", True, COL_INTERSECTION_ED)
        lrect = label.get_rect(center=rect.center)
        bg = pygame.Surface((lrect.w + 14, lrect.h + 8), pygame.SRCALPHA)
        bg.fill((255, 255, 255, 210))
        self.screen.blit(bg, (lrect.x - 7, lrect.y - 4))
        self.screen.blit(label, lrect)

        form = self.f_lg.render("Formation Zone", True, (150, 140, 100))
        frect = form.get_rect(
            center=(L.draw_x + L.draw_size * 0.20,
                    L.draw_y + L.draw_size * 0.82)
        )
        self.screen.blit(form, frect)

    def _draw_markers(self):
        """Green start markers + red goal markers."""
        L = self.layout
        r = max(6, int(L.cell * 0.32))

        # Start markers
        for vid, (row, col) in self.starts.items():
            cx, cy = L.cell_center(row, col)
            pygame.draw.circle(self.screen, COL_START, (cx, cy), r, 2)

        # Goal markers
        for vid, (row, col) in self.goals.items():
            cx, cy = L.cell_center(row, col)
            pygame.draw.circle(self.screen, COL_GOAL, (cx, cy), r, 2)
            pygame.draw.circle(self.screen, COL_GOAL, (cx, cy), 2)

    def _draw_vehicles(self):
        if not self.time_steps:
            return
        L = self.layout
        state = self.time_steps[self.time_index]

        car_len = int(L.cell * 0.72)
        car_wid = int(L.cell * 0.42)

        for vid, (row, col) in state.items():
            cx, cy = L.cell_center(row, col)
            colour = VEHICLE_COLORS.get(vid, (150, 150, 150))

            # Heading from the precomputed table
            headings = self._headings.get(vid, [])
            if self.time_index < len(headings):
                dr, dc = headings[self.time_index]
            else:
                dr, dc = 0, 1   # fallback east

            if dr == 0 and dc == 0:
                dr, dc = 0, 1

            # to convert(row, col) space into screen space.
            angle_deg = pygame.math.Vector2(dc, -dr).as_polar()[1]

            sprite = self._get_sprite(vid, colour, car_len, car_wid, angle_deg)
            rect = sprite.get_rect(center=(cx, cy))
            self.screen.blit(sprite, rect)

            # Vehicle ID label with dark halo
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                s = self.f_md.render(vid, True, (0, 0, 0))
                self.screen.blit(s, s.get_rect(center=(cx + ox, cy + oy)))
            s = self.f_md.render(vid, True, (255, 255, 255))
            self.screen.blit(s, s.get_rect(center=(cx, cy)))

    def _get_sprite(self, vid, colour, length_px, width_px, angle_deg):
        bucket = (int(round(angle_deg)) // 5) * 5
        key = (vid, bucket)
        if key not in self._sprite_cache:
            base = make_car_sprite(length_px, width_px, colour)
            self._sprite_cache[key] = pygame.transform.rotate(base, bucket)
        return self._sprite_cache[key]

    # Panel
    def _draw_panel(self):
        L = self.layout
        pygame.draw.rect(self.screen, COL_PANEL,
                         (L.panel_x, 0, L.panel_w, self.H))
        pygame.draw.line(self.screen, COL_PANEL_EDGE,
                         (L.panel_x, 0), (L.panel_x, self.H), 1)

        px = L.panel_x + 20
        pw = L.panel_w - 40
        y = 20

        # Header
        self.screen.blit(self.f_xl.render("DQN Intersection", True, COL_TEXT),
                         (px, y)); y += 30
        self.screen.blit(self.f.render("Multi-Agent RL for Autonomous Vehicles",
                                       True, COL_TEXT_DIM),
                         (px, y)); y += 26
        pygame.draw.line(self.screen, COL_PANEL_EDGE,
                         (px, y), (px + pw, y), 1); y += 14

        # Parameters
        self.screen.blit(self.f_lg.render("Parameters", True, COL_TEXT),
                         (px, y)); y += 26

        self.screen.blit(self.f.render("Vehicles:", True, COL_TEXT),
                         (px, y))
        self._draw_small_btn(self.ctrl_veh_minus, "-")
        self.screen.blit(self.f_md.render(str(self.num_vehicles), True, COL_TEXT),
                         (self.ctrl_veh_minus.right + 10, y + 2))
        self._draw_small_btn(self.ctrl_veh_plus, "+")
        y += 36

        self.screen.blit(self.f.render("Sensor Noise:", True, COL_TEXT),
                         (px, y))
        self._draw_cycle_btn(self.ctrl_noise, str(self.sensor_noise))
        y += 36

        self.screen.blit(self.f.render("Braking Delay:", True, COL_TEXT),
                         (px, y))
        self._draw_cycle_btn(self.ctrl_brake, str(self.braking_delay))
        y += 36

        self.screen.blit(self.f.render("Comm Latency:", True, COL_TEXT),
                         (px, y))
        self._draw_cycle_btn(self.ctrl_comm, str(self.comm_latency))
        y += 36

        self.screen.blit(self.f.render("Schedule:", True, COL_TEXT),
                         (px, y))
        self._draw_cycle_btn(self.ctrl_sched, self.schedule_type)
        y += 44

        self._draw_wide_btn(self.ctrl_load, "Load Schedule")
        y += 50

        coll_col = (200, 40, 40) if self.collisions > 0 else COL_TEXT_DIM
        self.screen.blit(
            self.f.render(f"Collisions: {self.collisions if self.time_steps else '-'}",
                          True, coll_col),
            (px, y)); y += 24
        self.screen.blit(
            self.f.render(self.status_msg[:36], True, COL_TEXT_DIM),
            (px, y)); y += 30

        pygame.draw.line(self.screen, COL_PANEL_EDGE,
                         (px, y), (px + pw, y), 1); y += 14

        # Legend
        self.screen.blit(self.f_lg.render("Legend", True, COL_TEXT),
                         (px, y)); y += 26
        for colour, label in [
            (COL_START, "Starting Position"),
            (COL_GOAL, "Goal Position"),
            (COL_INTERSECTION_ED, "Coordination Zone"),
            (COL_LANE_YELLOW, "Lane centre"),
        ]:
            pygame.draw.rect(self.screen, colour, (px, y + 3, 16, 16))
            self.screen.blit(self.f.render(label, True, COL_TEXT),
                             (px + 24, y))
            y += 24

        # Time label
        self.screen.blit(
            self.f_lg.render(f"Time: {self.time_index} / {self.max_time}",
                             True, COL_TEXT),
            (px, self.time_label_y))

        # Speed label
        self.screen.blit(self.f.render("Speed:", True, COL_TEXT),
                         (px, self.speed_label_y))
        speed_txt = self.f.render(f"{self.animation_speed:.0f} steps/sec",
                                  True, COL_TEXT_DIM)
        self.screen.blit(
            speed_txt,
            (px + pw - speed_txt.get_width(), self.speed_label_y))

        # Slider track
        pygame.draw.rect(self.screen, COL_SLIDER_TRACK,
                         self.slider_rect, border_radius=7)
        frac = (self.animation_speed - 1) / 29.0
        filled_w = int(self.slider_rect.width * frac)
        if filled_w > 0:
            pygame.draw.rect(self.screen, COL_SLIDER_FILL,
                             (self.slider_rect.left, self.slider_rect.top,
                              filled_w, self.slider_rect.height),
                             border_radius=7)
        handle_x = self.slider_rect.left + filled_w
        self.slider_handle.center = (handle_x, self.slider_rect.centery)
        pygame.draw.rect(self.screen, COL_SLIDER_HANDLE,
                         self.slider_handle, border_radius=5)
        pygame.draw.rect(self.screen, (30, 60, 140),
                         self.slider_handle, 1, border_radius=5)

        # Buttons
        self._draw_btn(self.btn_prev, "Prev")
        self._draw_btn(self.btn_play,
                       "Pause" if self.playing else "Play")
        self._draw_btn(self.btn_next, "Next")
        self._draw_btn(self.btn_reset, "Reset")

    # Button primitives
    def _draw_btn(self, rect, text):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen,
                         COL_BTN_HOVER if hover else COL_BTN_BG,
                         rect, border_radius=6)
        pygame.draw.rect(self.screen, COL_BTN_EDGE, rect, 1, border_radius=6)
        surf = self.f.render(text, True, COL_TEXT)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_small_btn(self, rect, text):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen,
                         COL_BTN_HOVER if hover else COL_BTN_BG,
                         rect, border_radius=5)
        pygame.draw.rect(self.screen, COL_BTN_EDGE, rect, 1, border_radius=5)
        surf = self.f_md.render(text, True, COL_TEXT)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_cycle_btn(self, rect, text):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen,
                         COL_BTN_HOVER if hover else COL_BTN_BG,
                         rect, border_radius=5)
        pygame.draw.rect(self.screen, COL_BTN_EDGE, rect, 1, border_radius=5)
        surf = self.f.render(text, True, COL_TEXT)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_wide_btn(self, rect, text):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        bg = (200, 220, 245) if hover else (220, 235, 250)
        pygame.draw.rect(self.screen, bg, rect, border_radius=6)
        pygame.draw.rect(self.screen, (80, 120, 200), rect, 1, border_radius=6)
        surf = self.f_lg.render(text, True, (20, 50, 110))
        self.screen.blit(surf, surf.get_rect(center=rect.center))



# Entry point
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "dqn_simulation_results_all_vehicles.json"))
    p.add_argument("--fullscreen", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    json_path = args.json
    if not os.path.exists(json_path):
        print(f"[error] JSON file not found: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to load JSON: {e}")
        sys.exit(1)

    screen, W, H = init_display(args.fullscreen)
    print(f"[info] Display: {W} x {H}")
    print(f"[info] Loaded {len(data)} entries from {json_path}")

    DQNDemo(data, screen, W, H).run()


if __name__ == "__main__":
    main()