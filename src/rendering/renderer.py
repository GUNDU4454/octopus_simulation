"""
renderer.py  —  Top-down debug visualiser

Painter's order:
  1.  Sandy seabed background
  2.  Subtle depth grid
  3.  Tentacles — coloured by grip state + contraction heat
  4.  Grip contact points with slip-state indicator
  5.  Tension force arrows per planted arm
  6.  Body net force accumulation arrow
  7.  Octopus body disc
  8.  Per-arm contraction bars (above body)
  9.  HUD: step, speed, planted count, distance, tension bars
"""

import pygame
import numpy as np
import math
from src.entities.tentacle import GripState
from src.locomotion.tentacle_controller import Role

# ── Colour palette ────────────────────────────────────────────────────
BG_SAND       = ( 38,  48,  28)
GRID          = ( 48,  60,  36)
BODY_FILL     = ( 60,  90, 120)
BODY_EDGE     = ( 30,  55,  80)
HEADING_COL   = (200, 220, 255)

ARM_FREE      = ( 80, 130, 180)   # reaching / retracting
ARM_PLANTED   = (220,  80,  60)   # gripping seabed
ARM_RETRACT   = (160, 160,  60)
ARM_SLIPPING  = (255, 200,  40)   # amber — grip under stress

# Segment contraction heat tint (blended in based on contraction level)
CONTRACT_HOT  = (255,  65,  20)   # red-orange for fully contracted muscle

GRIP_DOT      = (255, 100,  60)   # planted anchor ring
SLIP_DOT      = (255, 210,  30)   # slipping anchor ring
FORCE_ARM     = (255, 200,  50)   # per-arm tension arrow
FORCE_BODY    = ( 80, 255, 160)   # body net force accumulation arrow
VELOCITY_VEC  = (100, 220, 100)

HUD_TEXT      = (180, 200, 160)
HUD_DIM       = ( 70,  85,  60)
HUD_WARN      = (255, 180,  40)
HUD_BG        = (  8,  14,  10)

SUCKER        = (200, 160,  90)
SUCKER_DARK   = (120,  80,  30)

CONTRACT_BAR_FULL = (220,  60,  30)
CONTRACT_BAR_LOW  = ( 40, 110, 180)
TENSION_BAR       = (220, 160,  40)

PUSH_ARM      = ( 70, 200, 235)   # per-arm push (extension) arrow — cyan
TARGET_COL    = (255, 235,  90)   # crawl target marker / direction line
COM_COL       = (255, 120, 200)   # centre-of-mass crosshair
FORWARD_COL   = (120, 200, 255)   # body-forward (current heading) arrow
TORQUE_COL    = (255, 140, 220)   # net rotation-torque turn indicator

# Role → colour.  Roles describe what each arm is *doing* this frame and drive
# the arm tint + the debug panel, so the behaviour is readable at a glance.
ROLE_COLORS = {
    Role.PULLING:     (235,  90,  60),   # red-orange — dragging body to target
    Role.PUSHING:     ( 70, 200, 235),   # cyan       — shoving body forward
    Role.ANCHORING:   ( 80, 200, 120),   # green      — stable holding grip
    Role.STABILIZING: (185, 115, 225),   # purple     — damping rotation
    Role.SEARCHING:   (205, 195,  95),   # yellow     — reaching for a grip
    Role.IDLE:        ( 90, 120, 150),   # dim blue   — relaxed
}


def role_color(role: str) -> tuple:
    return ROLE_COLORS.get(role, ARM_FREE)


class Renderer:

    GRID_STEP        = 80
    ARM_FORCE_SCALE  = 0.30   # px per N for per-arm arrows
    BODY_FORCE_SCALE = 0.15   # px per N for body net force arrow

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        self.screen = screen
        self.width  = width
        self.height = height

        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_md = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_xs = pygame.font.SysFont("monospace", 11)

        self._bg = self._build_seabed()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def draw(self, octopus, step: int, manager=None, debug: bool = True):
        self.screen.blit(self._bg, (0, 0))
        self._draw_grid()

        # Target + heading are useful even with the debug overlay off.
        self._draw_target(octopus, manager)

        self._draw_tentacles(octopus, debug)
        if debug:
            self._draw_grip_contacts(octopus)
            self._draw_tension_vectors(octopus)
            self._draw_push_vectors(octopus)
            self._draw_body_net_force(octopus)
            self._draw_body_orientation(octopus, manager)

        self._draw_body(octopus)

        if debug:
            self._draw_com(octopus)
            self._draw_arm_labels(octopus)
            self._draw_contraction_bars(octopus)
            self._draw_hud(octopus, step)
            self._draw_role_panel(octopus, manager)
        else:
            self._draw_minimal_hint()

    # ------------------------------------------------------------------
    # Seabed
    # ------------------------------------------------------------------

    def _build_seabed(self) -> pygame.Surface:
        surf = pygame.Surface((self.width, self.height))
        surf.fill(BG_SAND)
        rng = np.random.default_rng(42)
        for _ in range(1800):
            x = int(rng.uniform(0, self.width))
            y = int(rng.uniform(0, self.height))
            r = int(rng.uniform(1, 3))
            shade = int(rng.uniform(28, 58))
            pygame.draw.circle(surf, (shade, shade+10, shade-8), (x, y), r)
        return surf

    def _draw_grid(self):
        for x in range(0, self.width, self.GRID_STEP):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.GRID_STEP):
            pygame.draw.line(self.screen, GRID, (0, y), (self.width, y), 1)

    # ------------------------------------------------------------------
    # Tentacles with contraction heat-map
    # ------------------------------------------------------------------

    def _draw_tentacles(self, octopus, debug: bool = True):
        for arm in octopus.arms:
            positions = arm.get_positions()
            n = len(positions)
            if n < 2:
                continue

            # Base colour: by role when debugging and a role is active (shows
            # behaviour); otherwise by raw grip state (cleaner look, and the
            # fallback the RL training view uses since it sets no roles).
            role = getattr(arm, "role", Role.IDLE)
            if debug and role != Role.IDLE:
                base = role_color(role)
            elif arm.state == GripState.PLANTED:    base = ARM_PLANTED
            elif arm.state == GripState.SLIPPING:   base = ARM_SLIPPING
            elif arm.state == GripState.RETRACTING: base = ARM_RETRACT
            else:                                   base = ARM_FREE

            for i in range(n - 1):
                p1 = positions[i]
                p2 = positions[i + 1]
                r1 = arm.radii[i]
                r2 = arm.radii[i + 1]
                pct = i / (n - 1)

                # Contraction heat: blend base colour toward CONTRACT_HOT
                seg_c = arm.contractions[i] if i < arm.n else 0.0
                seg_col = self._heat_blend(base, CONTRACT_HOT, float(seg_c) * 0.75)

                # Darken toward tip
                shade = 1.0 - pct * 0.40
                seg_col = tuple(max(0, int(c * shade)) for c in seg_col)

                self._tapered_segment(p1, p2, r1, r2, seg_col)

                # Suckers every 2 segments if wide enough
                if i % 2 == 1 and r1 > 3:
                    mx = (p1[0] + p2[0]) * 0.5
                    my = (p1[1] + p2[1]) * 0.5
                    sr = max(1, int(r1 * 0.45))
                    pygame.draw.circle(self.screen, SUCKER,
                                       (int(mx), int(my)), sr)
                    pygame.draw.circle(self.screen, SUCKER_DARK,
                                       (int(mx), int(my)), sr, 1)

    @staticmethod
    def _heat_blend(col_a: tuple, col_b: tuple, t: float) -> tuple:
        t = max(0.0, min(1.0, t))
        return tuple(int(col_a[i] * (1 - t) + col_b[i] * t) for i in range(3))

    def _tapered_segment(self, p1, p2, r1: float, r2: float, col: tuple):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        L = math.hypot(dx, dy)
        if L < 0.5:
            return
        nx, ny = -dy/L, dx/L
        pts = [
            (p1[0]+nx*r1, p1[1]+ny*r1),
            (p2[0]+nx*r2, p2[1]+ny*r2),
            (p2[0]-nx*r2, p2[1]-ny*r2),
            (p1[0]-nx*r1, p1[1]-ny*r1),
        ]
        pygame.draw.polygon(self.screen, col,
                            [(int(x), int(y)) for x, y in pts])
        pygame.draw.circle(self.screen, col,
                           (int(p1[0]), int(p1[1])), max(1, int(r1)))
        pygame.draw.circle(self.screen, col,
                           (int(p2[0]), int(p2[1])), max(1, int(r2)))

    # ------------------------------------------------------------------
    # Grip contacts
    # ------------------------------------------------------------------

    def _draw_grip_contacts(self, octopus):
        for arm in octopus.arms:
            if not arm.is_anchored:
                continue
            gx, gy = int(arm.grip_point[0]), int(arm.grip_point[1])
            slip_f  = arm.grip_slip_fraction()

            if arm.state == GripState.SLIPPING:
                # Slip indicator: amber ring, size grows with slip fraction
                outer_r = int(8 + slip_f * 6)
                col     = SLIP_DOT
                pygame.draw.circle(self.screen, col, (gx, gy), outer_r, 2)
                pygame.draw.circle(self.screen, col, (gx, gy), max(2, outer_r - 4))
                # Slip bar: short line showing drift direction
                tip = arm.points[-1]
                drift = arm.grip_point - tip.pos
                ddist = float(np.linalg.norm(drift))
                if ddist > 0.5:
                    ex = int(gx + drift[0] / ddist * 12)
                    ey = int(gy + drift[1] / ddist * 12)
                    pygame.draw.line(self.screen, (255, 255, 100), (gx, gy), (ex, ey), 2)
            else:
                # Solid planted anchor
                pygame.draw.circle(self.screen, GRIP_DOT, (gx, gy), 8, 2)
                pygame.draw.circle(self.screen, GRIP_DOT, (gx, gy), 4)
                pygame.draw.circle(self.screen, (255, 220, 180), (gx, gy), 2)

    # ------------------------------------------------------------------
    # Tension vectors — per planted arm
    # ------------------------------------------------------------------

    def _draw_tension_vectors(self, octopus):
        for arm in octopus.arms:
            if not arm.is_anchored:
                continue
            f = arm.tension_force()
            fmag = float(np.linalg.norm(f))
            if fmag < 1.0:
                continue
            root = arm.root_pos()
            rx, ry = int(root[0]), int(root[1])
            # Arrow scaled by force magnitude
            scale = self.ARM_FORCE_SCALE * min(fmag, 600.0) / max(fmag, 1.0)
            ex = int(rx + f[0] * scale)
            ey = int(ry + f[1] * scale)

            # Colour intensity by tension: dim → bright
            intensity = min(1.0, fmag / 300.0)
            col = tuple(int(FORCE_ARM[c] * (0.4 + 0.6 * intensity)) for c in range(3))
            pygame.draw.line(self.screen, col, (rx, ry), (ex, ey), 2)
            self._arrow_head(rx, ry, ex, ey, col, size=7)

            # Tension magnitude label
            lbl = self.font_xs.render(f"{fmag:.0f}", True, col)
            self.screen.blit(lbl, (ex + 4, ey - 6))

    # ------------------------------------------------------------------
    # Push vectors — per pushing arm (body-reaction force away from anchor)
    # ------------------------------------------------------------------

    def _draw_push_vectors(self, octopus):
        for arm in octopus.arms:
            fmag = getattr(arm, "_last_push_mag", 0.0)
            if fmag < 1.0:
                continue
            f = arm.push_force()
            fmag = float(np.linalg.norm(f))
            if fmag < 1.0:
                continue
            root = arm.root_pos()
            rx, ry = int(root[0]), int(root[1])
            scale = self.ARM_FORCE_SCALE * min(fmag, 600.0) / max(fmag, 1.0)
            ex = int(rx + f[0] * scale)
            ey = int(ry + f[1] * scale)
            pygame.draw.line(self.screen, PUSH_ARM, (rx, ry), (ex, ey), 2)
            self._arrow_head(rx, ry, ex, ey, PUSH_ARM, size=7)
            lbl = self.font_xs.render(f"{fmag:.0f}", True, PUSH_ARM)
            self.screen.blit(lbl, (ex + 4, ey - 6))

    # ------------------------------------------------------------------
    # Crawl target + move-direction indicator
    # ------------------------------------------------------------------

    def _draw_target(self, octopus, manager):
        if manager is None:
            return
        target = manager.target
        if target is None:
            return
        tx, ty = int(target[0]), int(target[1])
        bx, by = int(octopus.pos[0]), int(octopus.pos[1])

        # Dotted line body → target.
        self._dotted_line((bx, by), (tx, ty), TARGET_COL, gap=10)

        # Pulsing target ring + crosshair.
        pulse = 10 + int(4 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.006)))
        pygame.draw.circle(self.screen, TARGET_COL, (tx, ty), pulse, 2)
        pygame.draw.line(self.screen, TARGET_COL, (tx - 9, ty), (tx + 9, ty), 1)
        pygame.draw.line(self.screen, TARGET_COL, (tx, ty - 9), (tx, ty + 9), 1)
        lbl = self.font_xs.render("target", True, TARGET_COL)
        self.screen.blit(lbl, (tx + 12, ty - 6))

        # Short heading arrow on the body, pointing at the target.
        d = target - octopus.pos
        dist = float(np.linalg.norm(d))
        if dist > 1.0:
            d = d / dist
            ax = int(bx + d[0] * (octopus.RADIUS + 22))
            ay = int(by + d[1] * (octopus.RADIUS + 22))
            pygame.draw.line(self.screen, TARGET_COL, (bx, by), (ax, ay), 2)
            self._arrow_head(bx, by, ax, ay, TARGET_COL, size=8)

    def _dotted_line(self, p1, p2, col, gap: int = 10):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        steps = int(dist // gap)
        for s in range(0, steps, 2):
            t1 = s / max(steps, 1)
            t2 = min(1.0, (s + 1) / max(steps, 1))
            ax = int(x1 + (x2 - x1) * t1); ay = int(y1 + (y2 - y1) * t1)
            bx = int(x1 + (x2 - x1) * t2); by = int(y1 + (y2 - y1) * t2)
            pygame.draw.line(self.screen, col, (ax, ay), (bx, by), 1)

    # ------------------------------------------------------------------
    # Centre-of-mass indicator
    # ------------------------------------------------------------------

    def _draw_com(self, octopus):
        cx, cy = int(octopus.pos[0]), int(octopus.pos[1])
        pygame.draw.circle(self.screen, COM_COL, (cx, cy), 4, 1)
        pygame.draw.line(self.screen, COM_COL, (cx - 7, cy), (cx + 7, cy), 1)
        pygame.draw.line(self.screen, COM_COL, (cx, cy - 7), (cx, cy + 7), 1)
        lbl = self.font_xs.render("COM", True, COM_COL)
        self.screen.blit(lbl, (cx + 8, cy + 6))

    # ------------------------------------------------------------------
    # Per-arm number labels (above each arm's mid-point)
    # ------------------------------------------------------------------

    def _draw_arm_labels(self, octopus):
        for i, arm in enumerate(octopus.arms):
            pts = arm.get_positions()
            mid = pts[len(pts) // 2]
            col = role_color(getattr(arm, "role", Role.IDLE))
            lbl = self.font_md.render(str(i), True, (15, 15, 15))
            bg  = lbl.get_rect()
            bg.center = (int(mid[0]), int(mid[1]))
            pygame.draw.circle(self.screen, col, bg.center, 9)
            pygame.draw.circle(self.screen, (15, 15, 15), bg.center, 9, 1)
            self.screen.blit(lbl, (bg.center[0] - lbl.get_width() // 2,
                                   bg.center[1] - lbl.get_height() // 2))

    # ------------------------------------------------------------------
    # Per-tentacle role panel (right side)
    # ------------------------------------------------------------------

    def _draw_role_panel(self, octopus, manager):
        rows = manager.role_summary() if manager is not None else None
        if rows is None:
            return

        line_h = 30
        pad    = 8
        w      = 218
        h      = len(rows) * line_h + pad * 2 + 16
        x0     = self.width - w - 10
        y0     = 10

        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((8, 14, 10, 200))
        self.screen.blit(panel, (x0, y0))

        title = self.font_md.render("TENTACLES", True, HUD_TEXT)
        self.screen.blit(title, (x0 + pad, y0 + pad))

        for r in rows:
            ry  = y0 + pad + 16 + r["idx"] * line_h
            col = role_color(r["role"])

            # Number swatch.
            pygame.draw.circle(self.screen, col, (x0 + pad + 7, ry + 8), 8)
            num = self.font_sm.render(str(r["idx"]), True, (15, 15, 15))
            self.screen.blit(num, (x0 + pad + 7 - num.get_width() // 2,
                                   ry + 8 - num.get_height() // 2))

            # Role + grip line.
            grip = "grip" if r["gripping"] else "----"
            head = self.font_sm.render(
                f"{r['role']:<11} {grip}", True, col
            )
            self.screen.blit(head, (x0 + pad + 22, ry))

            # Detail line: force (tension or push), extension.
            force = r["push"] if r["role"] == Role.PUSHING else r["tension"]
            ftag  = "push" if r["role"] == Role.PUSHING else "pull"
            detail = self.font_xs.render(
                f"{ftag} {force:>4.0f}N  ext {r['extension']*100:>3.0f}%"
                f"  dir {r['reach_deg']:>+4.0f}°",
                True, HUD_TEXT
            )
            self.screen.blit(detail, (x0 + pad + 22, ry + 14))

    # ------------------------------------------------------------------
    # Minimal hint when debug overlay is off
    # ------------------------------------------------------------------

    def _draw_minimal_hint(self):
        hint = self.font_xs.render(
            "[ D ] debug   [ click ] target   [ r ] reset   [ space ] pause",
            True, (90, 105, 80)
        )
        self.screen.blit(hint, (12, self.height - 18))

    # ------------------------------------------------------------------
    # Body net force accumulation arrow
    # ------------------------------------------------------------------

    def _draw_body_net_force(self, octopus):
        f    = octopus.body_net_force
        fmag = float(np.linalg.norm(f))
        bx   = int(octopus.pos[0])
        by   = int(octopus.pos[1])

        # Net force arrow (green) — scaled
        if fmag > 2.0:
            scale = self.BODY_FORCE_SCALE
            ex = int(bx + f[0] * scale)
            ey = int(by + f[1] * scale)
            pygame.draw.line(self.screen, FORCE_BODY, (bx, by), (ex, ey), 3)
            self._arrow_head(bx, by, ex, ey, FORCE_BODY, size=9)

        # Body velocity arrow (blue-green) — separate
        vx = int(bx + octopus.vel[0] * 1.2)
        vy = int(by + octopus.vel[1] * 1.2)
        if abs(vx - bx) + abs(vy - by) > 3:
            pygame.draw.line(self.screen, VELOCITY_VEC, (bx, by), (vx, vy), 2)
            self._arrow_head(bx, by, vx, vy, VELOCITY_VEC, size=7)

    # ------------------------------------------------------------------
    # Body orientation: current forward vector + rotation-torque indicator
    # ------------------------------------------------------------------

    def _draw_body_orientation(self, octopus, manager):
        cx, cy = octopus.pos[0], octopus.pos[1]
        r      = octopus.RADIUS

        # Body forward vector — where the body is currently facing.
        fl = r + 46
        fx = int(cx + math.cos(octopus.angle) * fl)
        fy = int(cy + math.sin(octopus.angle) * fl)
        pygame.draw.line(self.screen, FORWARD_COL, (int(cx), int(cy)), (fx, fy), 2)
        self._arrow_head(int(cx), int(cy), fx, fy, FORWARD_COL, size=8)
        self.screen.blit(self.font_xs.render("forward", True, FORWARD_COL),
                         (fx + 5, fy - 6))

        # Rotation-torque indicator — a curved arrow whose sweep direction shows
        # which way the net tentacle torque is turning the body (same x=cos /
        # y=sin convention as the physics, so it matches the real rotation).
        tau = octopus.body_net_torque
        if abs(tau) > 8.0:
            arc_r     = r + 16
            sweep     = min(2.2, 0.4 + abs(tau) / 500.0)
            direction = 1.0 if tau > 0.0 else -1.0   # angle increases for +torque
            start     = octopus.angle - direction * sweep   # lead end ends near forward
            steps     = 16
            pts = [
                (int(cx + math.cos(start + direction * sweep * k / steps) * arc_r),
                 int(cy + math.sin(start + direction * sweep * k / steps) * arc_r))
                for k in range(steps + 1)
            ]
            pygame.draw.lines(self.screen, TORQUE_COL, False, pts, 3)
            self._arrow_head(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
                             TORQUE_COL, size=8)
            self.screen.blit(self.font_xs.render("turn", True, TORQUE_COL),
                             (pts[-1][0] + 4, pts[-1][1] - 6))

        # Heading-error readout (how far the body still has to rotate).
        drive = getattr(manager, "drive", None) if manager is not None else None
        if drive is not None and getattr(drive, "has_target", False):
            txt = (f"heading err {math.degrees(drive.heading_error):>+4.0f}°"
                   f"  turn {drive.turn_gain*100:>3.0f}%")
            self.screen.blit(self.font_xs.render(txt, True, FORWARD_COL),
                             (int(cx) - 70, int(cy) + r + 16))

    def _arrow_head(self, x1, y1, x2, y2, col, size=7):
        dx, dy = x2-x1, y2-y1
        L = math.hypot(dx, dy)
        if L < 1:
            return
        dx, dy = dx/L, dy/L
        lx, ly = -dy, dx
        pts = [
            (x2, y2),
            (int(x2 - dx*size + lx*size*0.5), int(y2 - dy*size + ly*size*0.5)),
            (int(x2 - dx*size - lx*size*0.5), int(y2 - dy*size - ly*size*0.5)),
        ]
        pygame.draw.polygon(self.screen, col, pts)

    # ------------------------------------------------------------------
    # Body disc
    # ------------------------------------------------------------------

    def _draw_body(self, octopus):
        cx, cy = int(octopus.pos[0]), int(octopus.pos[1])
        r      = octopus.RADIUS

        sh = pygame.Surface((r*2+12, r*2+12), pygame.SRCALPHA)
        pygame.draw.circle(sh, (0, 0, 0, 60), (r+6, r+8), r+2)
        self.screen.blit(sh, (cx-r-6, cy-r-6))

        pygame.draw.circle(self.screen, BODY_FILL, (cx, cy), r)

        for ring in range(3):
            rr = r - ring * 4
            if rr > 0:
                shade_surf = pygame.Surface((rr*2, rr*2), pygame.SRCALPHA)
                pygame.draw.circle(shade_surf, (0, 0, 0, 20+ring*15),
                                   (rr, rr), rr, rr//3)
                self.screen.blit(shade_surf, (cx-rr, cy-rr))

        pygame.draw.circle(self.screen, BODY_EDGE, (cx, cy), r, 2)

        hx = cx + int(math.cos(octopus.angle) * r * 0.75)
        hy = cy + int(math.sin(octopus.angle) * r * 0.75)
        pygame.draw.line(self.screen, HEADING_COL, (cx, cy), (hx, hy), 3)
        pygame.draw.circle(self.screen, HEADING_COL, (hx, hy), 4)

        perp = octopus.angle + math.pi / 2
        for side in (-1, 1):
            ex = cx + int(math.cos(octopus.angle)*r*0.45
                          + math.cos(perp)*r*0.35*side)
            ey = cy + int(math.sin(octopus.angle)*r*0.45
                          + math.sin(perp)*r*0.35*side)
            pygame.draw.circle(self.screen, (10, 10, 15), (ex, ey), 5)
            pygame.draw.circle(self.screen, (180, 150, 40), (ex, ey), 4)
            pygame.draw.circle(self.screen, (5, 2, 8),      (ex, ey), 2)
            pygame.draw.circle(self.screen, (240, 230, 200), (ex-1, ey-1), 1)

    # ------------------------------------------------------------------
    # Per-arm contraction bars (small radial bars near body)
    # ------------------------------------------------------------------

    def _draw_contraction_bars(self, octopus):
        cx, cy = octopus.pos[0], octopus.pos[1]
        bar_dist  = octopus.RADIUS + 14   # radius at which bars appear
        bar_len   = 12

        for arm in octopus.arms:
            angle = arm.world_angle
            base_x = cx + math.cos(angle) * bar_dist
            base_y = cy + math.sin(angle) * bar_dist
            tip_x  = base_x + math.cos(angle) * bar_len * arm.avg_contraction()
            tip_y  = base_y + math.sin(angle) * bar_len * arm.avg_contraction()

            # Background track
            track_end_x = base_x + math.cos(angle) * bar_len
            track_end_y = base_y + math.sin(angle) * bar_len
            pygame.draw.line(self.screen, (40, 50, 35),
                             (int(base_x), int(base_y)),
                             (int(track_end_x), int(track_end_y)), 3)

            if arm.avg_contraction() > 0.02:
                c = arm.avg_contraction()
                bar_col = self._heat_blend(CONTRACT_BAR_LOW, CONTRACT_BAR_FULL, c)
                pygame.draw.line(self.screen, bar_col,
                                 (int(base_x), int(base_y)),
                                 (int(tip_x), int(tip_y)), 3)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, octopus, step: int):
        fmag = float(np.linalg.norm(octopus.body_net_force))
        lines = [
            f"step    {step:>6}",
            f"speed   {octopus.speed:>6.1f} px/s",
            f"anchored {octopus.planted_count}/8",
            f"dist    {octopus.total_distance:>6.0f} px",
            f"net_F   {fmag:>6.0f} N",
            f"net_T   {octopus.body_net_torque:>+6.0f}",
        ]

        pad = 10
        lh  = 18
        w   = 168
        h   = len(lines) * lh + pad * 2

        hud = pygame.Surface((w, h), pygame.SRCALPHA)
        hud.fill((8, 14, 10, 190))
        self.screen.blit(hud, (pad, pad))

        for i, line in enumerate(lines):
            surf = self.font_sm.render(line, True, HUD_TEXT)
            self.screen.blit(surf, (pad*2, pad*2 + i*lh))

        # Per-arm strip at bottom
        strip_y = self.height - 46
        for i, arm in enumerate(octopus.arms):
            x = 12 + i * 44

            # Role background
            bg = role_color(getattr(arm, "role", Role.IDLE))
            pygame.draw.rect(self.screen, bg, (x, strip_y, 36, 14))

            # Contraction level bar overlay (filled bottom portion)
            c = arm.avg_contraction()
            if c > 0.01:
                bar_w = int(36 * c)
                bar_col = self._heat_blend(CONTRACT_BAR_LOW, CONTRACT_BAR_FULL, c)
                bar_surf = pygame.Surface((bar_w, 14), pygame.SRCALPHA)
                bar_surf.fill((*bar_col, 140))
                self.screen.blit(bar_surf, (x, strip_y))

            # Slip indicator (yellow X overlay)
            if arm.state == GripState.SLIPPING:
                sf = arm.grip_slip_fraction()
                warn = self.font_xs.render("!", True, (255, 60, 30))
                self.screen.blit(warn, (x + 26, strip_y))

            lbl = self.font_xs.render(str(i), True, (20, 20, 20))
            self.screen.blit(lbl, (x + 13, strip_y + 1))

        # Tension magnitude bars per arm (row below state strip)
        bar_y = strip_y + 16
        max_t = 400.0
        for i, arm in enumerate(octopus.arms):
            x = 12 + i * 44
            t = arm._last_tension_mag
            bar_w = int(min(36, 36 * t / max_t))
            pygame.draw.rect(self.screen, (30, 35, 25), (x, bar_y, 36, 6))
            if bar_w > 0:
                pygame.draw.rect(self.screen, TENSION_BAR, (x, bar_y, bar_w, 6))

        # Role legend
        legend = self.font_xs.render(
            "pull  push  anchor  stabilize  search   [ contraction ]  [ tension ]",
            True, (70, 80, 60)
        )
        self.screen.blit(legend, (12, self.height - 22))

        # Force legend
        fleg_col  = self.font_xs.render("── net force", True, FORCE_BODY)
        vleg_col  = self.font_xs.render("── velocity", True, VELOCITY_VEC)
        tleg_col  = self.font_xs.render("── arm pull", True, FORCE_ARM)
        pleg_col  = self.font_xs.render("── arm push", True, PUSH_ARM)
        self.screen.blit(fleg_col,  (self.width - 250, self.height - 72))
        self.screen.blit(vleg_col,  (self.width - 250, self.height - 58))
        self.screen.blit(tleg_col,  (self.width - 250, self.height - 44))
        self.screen.blit(pleg_col,  (self.width - 250, self.height - 30))

        ctrl = self.font_xs.render(
            "[ click ] target   [ d ] debug   [ r ] reset   [ space ] pause   [ esc/q ] quit",
            True, (50, 62, 42)
        )
        self.screen.blit(ctrl, (self.width - 560, self.height - 14))
