"""
demo.py

Full-screen Pygame visualisation of the trained DDPG policy.
"""

import os
import sys
import argparse

import numpy as np
import pygame

import config
from environment import ContinuousIntersectionEnv
from ddpg_agent import DDPGAgent


# Colours
COL_FORMATION       = (248, 243, 222)
COL_ROAD            = (222, 217, 202)
COL_ROAD_EDGE       = (150, 145, 125)
COL_LANE_WHITE      = (255, 255, 255)
COL_LANE_YELLOW     = (235, 195, 30)
COL_INTERSECTION    = (175, 210, 245)
COL_INTERSECTION_BG = (175, 210, 245, 150)
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

CAR_COLOURS = [
    (218, 60, 75), (60, 130, 230), (60, 180, 90), (240, 140, 50),
    (145, 60, 180), (60, 200, 210), (220, 60, 180), (200, 200, 40),
]


# Display
def init_display(fullscreen):
    pygame.init()
    pygame.display.set_caption("DDPG Intersection Demo")
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        info = pygame.display.Info()
        W = int(info.current_w * 0.92)
        H = int(info.current_h * 0.92)
        screen = pygame.display.set_mode((W, H))
    return screen, screen.get_width(), screen.get_height()


# Layout
class Layout:
    def __init__(self, W, H, grid_size):
        self.W = W
        self.H = H
        self.grid = grid_size

        self.panel_w = min(380, max(280, int(W * 0.22)))
        self.panel_x = W - self.panel_w

        canvas_area_w = W - self.panel_w
        canvas_area_h = H
        pad = 30
        self.draw_size = min(canvas_area_w, canvas_area_h) - 2 * pad
        self.draw_x = (canvas_area_w - self.draw_size) // 2
        self.draw_y = (canvas_area_h - self.draw_size) // 2
        self.scale = self.draw_size / grid_size

    def world_to_screen(self, x, y):
        px = self.draw_x + (x / self.grid) * self.draw_size
        py = self.draw_y + self.draw_size - (y / self.grid) * self.draw_size
        return int(px), int(py)

    def world_len(self, length):
        return length * self.scale


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

    # Windshield
    ws_w = max(2, int(length_px * 0.10))
    ws_x = int(length_px * 0.72)
    ws_h = int(width_px * 0.7)
    ws_y = (width_px - ws_h) // 2
    pygame.draw.rect(surf, (180, 220, 245), (ws_x, ws_y, ws_w, ws_h))

    # Headlights
    hl = max(2, int(width_px * 0.18))
    pygame.draw.rect(surf, (255, 250, 200), (length_px - hl - 1, 1, hl, hl))
    pygame.draw.rect(surf, (255, 250, 200),
                     (length_px - hl - 1, width_px - hl - 1, hl, hl))
    # Tail lights
    tl = max(2, int(width_px * 0.18))
    pygame.draw.rect(surf, (255, 90, 90), (1, 1, tl, tl))
    pygame.draw.rect(surf, (255, 90, 90),
                     (1, width_px - tl - 1, tl, tl))
    return surf


# Episode runner
def run_episode(agent, env, max_steps):
    states = env.reset()
    snapshots = [{
        "positions": env.positions.copy(),
        "velocities": env.velocities.copy(),
        "reached": env.reached.copy(),
        "collisions": set(),
        "step": 0,
    }]
    for step in range(max_steps):
        actions = np.stack([
            agent.select_action(states[i], add_noise=False)
            for i in range(env.num_vehicles)
        ])
        states, rewards, done, info = env.step(actions)
        snapshots.append({
            "positions": env.positions.copy(),
            "velocities": env.velocities.copy(),
            "reached": info["reached"].copy(),
            "collisions": set(info["collisions"]),
            "step": step + 1,
        })
        if done:
            break
    return snapshots


# Main app
class DDPGDemo:
    def __init__(self, snapshots, env, args, screen, W, H):
        self.snapshots = snapshots
        self.env = env
        self.args = args
        self.screen = screen
        self.W = W
        self.H = H
        self.layout = Layout(W, H, env.grid)

        self.frame_idx = 0
        self.playing = False
        self.clock = pygame.time.Clock()
        self.animation_speed = 6.0
        self.last_time = pygame.time.get_ticks()
        self.accumulator = 0.0

        self.f_sm = pygame.font.SysFont("Segoe UI", 12)
        self.f    = pygame.font.SysFont("Segoe UI", 14)
        self.f_md = pygame.font.SysFont("Segoe UI", 16)
        self.f_lg = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.f_xl = pygame.font.SysFont("Segoe UI", 22, bold=True)

        self.car_length_px = self.layout.world_len(config.VEHICLE_RADIUS * 2.8)
        self.car_width_px  = self.layout.world_len(config.VEHICLE_RADIUS * 1.7)

        # Buttons (panel bottom)
        px = self.layout.panel_x + 20
        pw = self.layout.panel_w - 40
        bw = (pw - 10) // 2
        bh = 36
        self.btn_prev  = pygame.Rect(px, H - 200, bw, bh)
        self.btn_play  = pygame.Rect(px + bw + 10, H - 200, bw, bh)
        self.btn_next  = pygame.Rect(px, H - 155, bw, bh)
        self.btn_reset = pygame.Rect(px + bw + 10, H - 155, bw, bh)

        # Speed slider
        self.slider_rect = pygame.Rect(px, H - 80, pw, 12)
        self.slider_handle = pygame.Rect(0, 0, 18, 22)
        self.dragging_slider = False

        self._sprite_cache = {}

    def _get_sprite(self, slot, angle_rad):
        deg = int(np.degrees(angle_rad)) % 360
        bucket = (deg // 5) * 5
        key = (slot, bucket)
        if key not in self._sprite_cache:
            base = make_car_sprite(self.car_length_px, self.car_width_px,
                                   CAR_COLOURS[slot % len(CAR_COLOURS)])
            self._sprite_cache[key] = pygame.transform.rotate(base, bucket)
        return self._sprite_cache[key]

    def run(self):
        running = True
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

            if self.playing:
                self.accumulator += dt
                interval = 1.0 / max(self.animation_speed, 0.001)
                while self.accumulator >= interval:
                    self.accumulator -= interval
                    if self.frame_idx < len(self.snapshots) - 1:
                        self.frame_idx += 1
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
            if self.btn_prev.collidepoint(event.pos):   self._prev()
            elif self.btn_play.collidepoint(event.pos): self._toggle_play()
            elif self.btn_next.collidepoint(event.pos): self._next()
            elif self.btn_reset.collidepoint(event.pos):self._reset()
            elif self.slider_rect.inflate(0, 20).collidepoint(event.pos):
                self.dragging_slider = True
                self._update_speed(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_slider = False
        elif event.type == pygame.MOUSEMOTION and self.dragging_slider:
            self._update_speed(event.pos)

    def _update_speed(self, pos):
        x = max(self.slider_rect.left, min(pos[0], self.slider_rect.right))
        frac = (x - self.slider_rect.left) / self.slider_rect.width
        self.animation_speed = 1.0 + frac * 29.0

    def _toggle_play(self):
        if not self.playing and self.frame_idx >= len(self.snapshots) - 1:
            self.frame_idx = 0
            self.accumulator = 0.0
        self.playing = not self.playing

    def _next(self):
        if self.frame_idx < len(self.snapshots) - 1: self.frame_idx += 1

    def _prev(self):
        if self.frame_idx > 0: self.frame_idx -= 1

    def _reset(self):
        self.playing = False
        self.frame_idx = 0
        self.accumulator = 0.0

    # Rendering
    def _render(self):
        self.screen.fill(COL_FORMATION)
        self._draw_road()
        self._draw_intersection()
        self._draw_markers()
        self._draw_cars()
        self._draw_panel()
        pygame.display.flip()

    def _draw_road(self):
        L = self.layout
        g = config.GRID_SIZE
        rmin, rmax = config.ROAD_MIN, config.ROAD_MAX
        c = config.CENTER

        # Road surface (horizontal band)
        x1, y1 = L.world_to_screen(0, rmax)
        x2, y2 = L.world_to_screen(g, rmin)
        pygame.draw.rect(self.screen, COL_ROAD, (x1, y1, x2 - x1, y2 - y1))

        # Road surface (vertical band)
        x1, y1 = L.world_to_screen(rmax, 0)
        x2, y2 = L.world_to_screen(rmin, g)
        pygame.draw.rect(self.screen, COL_ROAD, (x1, y1, x2 - x1, y2 - y1))

        # Road edges (dark grey)
        for y_l in (rmin, rmax):
            for xa, xb in [(0, rmin), (rmax, g)]:
                p1 = L.world_to_screen(xa, y_l)
                p2 = L.world_to_screen(xb, y_l)
                pygame.draw.line(self.screen, COL_ROAD_EDGE, p1, p2, 2)
        for x_l in (rmin, rmax):
            for ya, yb in [(0, rmin), (rmax, g)]:
                p1 = L.world_to_screen(x_l, ya)
                p2 = L.world_to_screen(x_l, yb)
                pygame.draw.line(self.screen, COL_ROAD_EDGE, p1, p2, 2)

        # Yellow dashed centre line separating directions
        for xa, xb in [(0.5, rmin - 0.5), (rmax + 0.5, g - 0.5)]:
            p1 = L.world_to_screen(xa, c)
            p2 = L.world_to_screen(xb, c)
            self._dash(p1, p2, COL_LANE_YELLOW, 3, 12, 8)
        for ya, yb in [(0.5, rmin - 0.5), (rmax + 0.5, g - 0.5)]:
            p1 = L.world_to_screen(c, ya)
            p2 = L.world_to_screen(c, yb)
            self._dash(p1, p2, COL_LANE_YELLOW, 3, 12, 8)

        # White edge lines (inner edge of each lane)
        for y_l in (rmin + 0.15, rmax - 0.15):
            for xa, xb in [(0.5, rmin), (rmax, g - 0.5)]:
                p1 = L.world_to_screen(xa, y_l)
                p2 = L.world_to_screen(xb, y_l)
                pygame.draw.line(self.screen, COL_LANE_WHITE, p1, p2, 2)
        for x_l in (rmin + 0.15, rmax - 0.15):
            for ya, yb in [(0.5, rmin), (rmax, g - 0.5)]:
                p1 = L.world_to_screen(x_l, ya)
                p2 = L.world_to_screen(x_l, yb)
                pygame.draw.line(self.screen, COL_LANE_WHITE, p1, p2, 2)

    def _dash(self, p1, p2, colour, width, dash, gap):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist == 0: return
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
        x1, y1 = L.world_to_screen(config.ROAD_MIN, config.ROAD_MAX)
        x2, y2 = L.world_to_screen(config.ROAD_MAX, config.ROAD_MIN)
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

        # Formation Zone label
        form = self.f_lg.render("Formation Zone", True, (140, 130, 90))
        frect = form.get_rect(center=(L.draw_x + L.draw_size * 0.20,
                                      L.draw_y + L.draw_size * 0.82))
        self.screen.blit(form, frect)

    def _draw_markers(self):
        L = self.layout
        r = max(4, int(L.world_len(0.45)))
        for slot in range(self.env.num_vehicles):
            sx, sy = L.world_to_screen(*config.START_POSITIONS[slot])
            pygame.draw.circle(self.screen, COL_START, (sx, sy), r, 2)
            gx, gy = L.world_to_screen(*config.GOAL_POSITIONS[slot])
            pygame.draw.circle(self.screen, COL_GOAL, (gx, gy), r, 2)
            pygame.draw.circle(self.screen, COL_GOAL, (gx, gy), 2)

    def _draw_cars(self):
        snap = self.snapshots[self.frame_idx]
        L = self.layout
        for v in range(self.env.num_vehicles):
            slot = self.env.slot_ids[v]
            pos = snap["positions"][v]
            vel = snap["velocities"][v]
            sx, sy = L.world_to_screen(pos[0], pos[1])

            speed = float(np.hypot(vel[0], vel[1]))
            if speed > 0.05:
                ang = float(np.arctan2(vel[1], vel[0]))
            else:
                goal = config.GOAL_POSITIONS[slot]
                ang = float(np.arctan2(goal[1] - pos[1], goal[0] - pos[0]))

            sprite = self._get_sprite(slot, ang)
            rect = sprite.get_rect(center=(sx, sy))

            reached = bool(snap["reached"][v])
            if reached:
                copy = sprite.copy(); copy.set_alpha(150)
                self.screen.blit(copy, rect)
            else:
                self.screen.blit(sprite, rect)

            if v in snap["collisions"]:
                rr = int(L.world_len(config.VEHICLE_RADIUS)) + 6
                pygame.draw.circle(self.screen, (240, 60, 60), (sx, sy), rr, 3)

            # ID label with a dark halo
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                s = self.f_md.render(str(v), True, (0, 0, 0))
                self.screen.blit(s, s.get_rect(center=(sx + ox, sy + oy)))
            s = self.f_md.render(str(v), True, (255, 255, 255))
            self.screen.blit(s, s.get_rect(center=(sx, sy)))

    def _draw_panel(self):
        L = self.layout
        pygame.draw.rect(self.screen, COL_PANEL,
                         (L.panel_x, 0, L.panel_w, self.H))
        pygame.draw.line(self.screen, COL_PANEL_EDGE,
                         (L.panel_x, 0), (L.panel_x, self.H), 1)

        px = L.panel_x + 20
        y = 20
        pw = L.panel_w - 40

        self.screen.blit(self.f_xl.render("DDPG Intersection", True, COL_TEXT), (px, y)); y += 34
        self.screen.blit(self.f.render("Multi-Agent RL for Autonomous Vehicles",
                                       True, COL_TEXT_DIM), (px, y)); y += 30
        pygame.draw.line(self.screen, COL_PANEL_EDGE, (px, y), (px + pw, y), 1); y += 16

        self.screen.blit(self.f_lg.render("Parameters", True, COL_TEXT), (px, y)); y += 26
        self.screen.blit(self.f.render(f"Vehicles: {self.env.num_vehicles}", True, COL_TEXT), (px, y)); y += 22
        self.screen.blit(self.f.render(f"Sensor Noise: {self.args.noise}", True, COL_TEXT), (px, y)); y += 22

        snap = self.snapshots[self.frame_idx]
        n_reached = int(snap["reached"].sum())
        self.screen.blit(self.f.render(
            f"Reached: {n_reached}/{self.env.num_vehicles}",
            True, (40, 130, 40)), (px, y)); y += 22

        n_coll = len(snap["collisions"])
        self.screen.blit(self.f.render(
            f"Collisions: {n_coll}",
            True, (200, 40, 40) if n_coll > 0 else COL_TEXT_DIM), (px, y)); y += 22

        self.screen.blit(self.f.render(
            f"Step: {snap['step']} / {self.snapshots[-1]['step']}",
            True, COL_TEXT), (px, y)); y += 30

        pygame.draw.line(self.screen, COL_PANEL_EDGE, (px, y), (px + pw, y), 1); y += 16

        self.screen.blit(self.f_lg.render("Legend", True, COL_TEXT), (px, y)); y += 26
        for colour, label in [
            (COL_START, "Start position"),
            (COL_GOAL,  "Goal position"),
            (COL_INTERSECTION_ED, "Coordination zone"),
            (COL_LANE_YELLOW, "Lane centre")
        ]:
            pygame.draw.rect(self.screen, colour, (px, y + 3, 16, 16))
            self.screen.blit(self.f.render(label, True, COL_TEXT), (px + 24, y))
            y += 22

        # Speed slider
        sy = self.H - 80
        self.screen.blit(self.f_lg.render("Speed", True, COL_TEXT), (px, sy - 30))
        self.screen.blit(self.f.render(
            f"{self.animation_speed:.0f} steps/sec",
            True, COL_TEXT_DIM), (px + 70, sy - 28))

        pygame.draw.rect(self.screen, COL_SLIDER_TRACK,
                         self.slider_rect, border_radius=6)
        frac = (self.animation_speed - 1) / 29.0
        filled_w = int(self.slider_rect.width * frac)
        if filled_w > 0:
            pygame.draw.rect(self.screen, COL_SLIDER_FILL,
                             (self.slider_rect.left, self.slider_rect.top,
                              filled_w, self.slider_rect.height),
                             border_radius=6)
        handle_x = self.slider_rect.left + filled_w
        self.slider_handle.center = (handle_x, self.slider_rect.centery)
        pygame.draw.rect(self.screen, COL_SLIDER_HANDLE,
                         self.slider_handle, border_radius=5)
        pygame.draw.rect(self.screen, (30, 60, 140),
                         self.slider_handle, 1, border_radius=5)

        # Buttons
        self._draw_btn(self.btn_prev,  "Prev")
        self._draw_btn(self.btn_play,  "Pause" if self.playing else "Play")
        self._draw_btn(self.btn_next,  "Next")
        self._draw_btn(self.btn_reset, "Reset")

    def _draw_btn(self, rect, text):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, COL_BTN_HOVER if hover else COL_BTN_BG,
                         rect, border_radius=6)
        pygame.draw.rect(self.screen, COL_BTN_EDGE, rect, 1, border_radius=6)
        surf = self.f.render(text, True, COL_TEXT)
        self.screen.blit(surf, surf.get_rect(center=rect.center))


# Entry point
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vehicles", type=int, default=config.NUM_VEHICLES)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--model", type=str,
                   default=os.path.join(config.MODEL_DIR, "best"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fullscreen", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    screen, W, H = init_display(args.fullscreen)
    print(f"[info] Display: {W} x {H}")

    config.NUM_VEHICLES = args.vehicles
    env = ContinuousIntersectionEnv(config, noise_level=args.noise)
    agent = DDPGAgent(config.STATE_DIM, config.ACTION_DIM, config)

    if os.path.exists(args.model + "_actor.pth"):
        agent.load(args.model)
        print(f"[info] Loaded model from {args.model}")
    else:
        print(f"[warn] No model at {args.model} — using untrained network")

    print(f"[info] Running: {args.vehicles} vehicles, noise={args.noise}")
    snapshots = run_episode(agent, env, config.MAX_STEPS)
    print(f"[info] Rolled out {snapshots[-1]['step']} steps")

    DDPGDemo(snapshots, env, args, screen, W, H).run()


if __name__ == "__main__":
    main()