import sys
import math
import pygame
import numpy as np
import torch
from typing import List, Tuple, Optional

# ── Window ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 960, 720
FPS = 30
OCT_CX, OCT_CY = 480, 390       # octopus center on screen

# ── Palette ───────────────────────────────────────────────────────────────────
SAND_BASE     = (198, 172, 118)
ROCK_DARK     = (108,  98,  82)
ALGAE_GREEN   = ( 62, 102,  64)
WATER_TINT    = ( 12,  42,  88, 38)   # RGBA overlay
OCTOPUS_SKIN  = (158,  82,  42)
SKIN_HIGHLIGHT= (188, 118,  72)
EYE_OUTER     = ( 28,  18,   8)
EYE_INNER     = (  8,   8,   8)
EYE_GLINT     = (210, 210, 210)


# ── Ocean floor ───────────────────────────────────────────────────────────────

def _generate_floor(width: int, height: int) -> pygame.Surface:
    """
    Builds a one-time procedural sandy ocean floor.
    Uses numpy noise so each run looks slightly different.
    """
    rng = np.random.default_rng(42)   # fixed seed = consistent look

    # Start with base sand color + per-pixel noise
    noise = rng.integers(-18, 18, (height, width, 3), dtype=np.int32)
    base  = np.array(SAND_BASE, dtype=np.int32)
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Scatter dark rocks and green algae patches
    for _ in range(40):
        rx   = rng.integers(0, width)
        ry   = rng.integers(0, height)
        rr   = rng.integers(12, 55)
        kind = rng.choice(["rock", "algae"])
        color = np.array(ROCK_DARK if kind == "rock" else ALGAE_GREEN, dtype=np.int32)
        y_idx, x_idx = np.ogrid[:height, :width]
        mask = (x_idx - rx)**2 + (y_idx - ry)**2 < rr**2
        blend = rng.integers(30, 70) / 100
        pixels[mask] = np.clip(
            pixels[mask] * (1 - blend) + color * blend, 0, 255
        ).astype(np.uint8)

    # pygame surfarray expects (width, height, 3)
    surf = pygame.Surface((width, height))
    pygame.surfarray.blit_array(surf, pixels.transpose(1, 0, 2))

    # Semi-transparent water-blue tint
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill(WATER_TINT)
    surf.blit(overlay, (0, 0))
    return surf


# ── Chromatophore ─────────────────────────────────────────────────────────────

class Chromatophore:
    """
    One pigment cell in the octopus skin.
    In real octopi these expand (darken/colour) or contract (pale).
    We simulate that with smooth radius + colour interpolation.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y   = y
        self.color        = OCTOPUS_SKIN
        self.target_color = OCTOPUS_SKIN
        self.radius        = 3.0
        self.target_radius = 3.0

    def set_target(self, color: Tuple[int, int, int], expanded: bool):
        self.target_color  = color
        self.target_radius = 6.0 if expanded else 2.0

    def update(self):
        lerp = 0.14   # smoothing speed — lower = slower, dreamier
        self.color = tuple(
            int(self.color[i] + (self.target_color[i] - self.color[i]) * lerp)
            for i in range(3)
        )
        self.radius += (self.target_radius - self.radius) * lerp

    def draw(self, surf: pygame.Surface):
        r = max(1, int(self.radius))
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), r)


# ── Arm geometry ──────────────────────────────────────────────────────────────

def _arm_polygon(
    cx: float, cy: float,
    angle: float,
    length: float,
    base_width: float,
    segments: int = 16,
) -> List[Tuple[int, int]]:
    """
    Returns a closed polygon for one tapered, gently curving arm.
    The S-curve is achieved by a sine offset perpendicular to the arm axis.
    """
    left, right = [], []
    perp = angle + math.pi / 2

    for i in range(segments + 1):
        t      = i / segments
        dist   = length * t
        curve  = math.sin(t * math.pi) * 18   # max lateral bend at midpoint
        width  = base_width * (1.0 - t * 0.92)

        bx = cx + math.cos(angle) * dist + math.cos(perp) * curve * 0.35
        by = cy + math.sin(angle) * dist + math.sin(perp) * curve * 0.35

        ox = math.cos(perp) * width
        oy = math.sin(perp) * width

        left.append((int(bx + ox), int(by + oy)))
        right.append((int(bx - ox), int(by - oy)))

    return left + list(reversed(right))


# ── Main simulation class ─────────────────────────────────────────────────────

class OctopusSimulation:
    ARM_LENGTH    = 170
    ARM_BASE_W    = 16
    MANTLE_RX     = 72
    MANTLE_RY     = 58
    CELL_SPACING  = 9      # pixels between chromatophores

    def __init__(self, agent=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Octopus Camouflage Simulation")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("monospace", 13)

        self.agent = agent
        self.floor = _generate_floor(WIDTH, HEIGHT)

        # 8 arm angles evenly spaced, first arm pointing straight up
        self.arm_angles = [
            i * (2 * math.pi / 8) - math.pi / 2
            for i in range(8)
        ]

        self.cells: List[Chromatophore] = []
        self._populate_cells()

        # RL wiring
        self.env = self.obs = None
        if agent is not None:
            from .environment import OctopusEnv
            self.env = OctopusEnv()
            self.obs, _ = self.env.reset()

        self._tick = 0   # used for the demo animation when no agent is loaded

    # ── Cell placement ────────────────────────────────────────────────────────

    def _in_mantle(self, x: float, y: float) -> bool:
        dx = (x - OCT_CX) / self.MANTLE_RX
        dy = (y - OCT_CY) / self.MANTLE_RY
        return dx * dx + dy * dy <= 1.0

    def _in_arm(self, x: float, y: float, angle: float) -> bool:
        """Approximate arm containment: distance to arm centre-line < half-width."""
        for t in np.linspace(0.05, 1.0, 30):
            dist  = self.ARM_LENGTH * t
            curve = math.sin(t * math.pi) * 18
            perp  = angle + math.pi / 2
            ax = OCT_CX + math.cos(angle) * dist + math.cos(perp) * curve * 0.35
            ay = OCT_CY + math.sin(angle) * dist + math.sin(perp) * curve * 0.35
            hw = self.ARM_BASE_W * (1.0 - t * 0.92) + 2
            if (x - ax) ** 2 + (y - ay) ** 2 < hw * hw:
                return True
        return False

    def _populate_cells(self):
        s = self.CELL_SPACING
        x_range = range(OCT_CX - self.ARM_LENGTH - 20, OCT_CX + self.ARM_LENGTH + 20, s)
        y_range = range(OCT_CY - self.ARM_LENGTH - 20, OCT_CY + self.ARM_LENGTH + 20, s)

        for x in x_range:
            for y in y_range:
                if self._in_mantle(x, y):
                    self.cells.append(Chromatophore(x, y))
                    continue
                for angle in self.arm_angles:
                    if self._in_arm(x, y, angle):
                        self.cells.append(Chromatophore(x, y))
                        break

    # ── Pattern source ────────────────────────────────────────────────────────

    def _demo_pattern(self) -> np.ndarray:
        """
        Slow colour wave used when no trained agent is loaded.
        Makes the octopus look alive even before training.
        """
        self._tick += 1
        t = self._tick / 60
        pattern = np.zeros((3, 64, 64), np.float32)
        y, x = np.mgrid[0:64, 0:64] / 63
        pattern[0] = 0.55 + 0.25 * np.sin(t + x * 4)
        pattern[1] = 0.35 + 0.15 * np.cos(t * 0.7 + y * 3)
        pattern[2] = 0.20 + 0.10 * np.sin(t * 1.4 + (x + y) * 2)
        return pattern

    def _agent_pattern(self) -> np.ndarray:
        action, _ = self.agent.predict(self.obs, deterministic=True)
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.obs, _ = self.env.reset()

        with torch.no_grad():
            z = torch.tensor(self.env.latent_vector).unsqueeze(0)
            p = self.env.generator(z)
            p = (p.squeeze(0).numpy() + 1) / 2
        return p.astype(np.float32)

    # ── Update chromatophores ─────────────────────────────────────────────────

    def _apply_pattern(self, pattern: np.ndarray):
        """Sample the 64×64 ML pattern onto each cell by position."""
        x_min = OCT_CX - self.ARM_LENGTH - 10
        x_max = OCT_CX + self.ARM_LENGTH + 10
        y_min = OCT_CY - self.ARM_LENGTH - 10
        y_max = OCT_CY + self.ARM_LENGTH + 10
        span_x = x_max - x_min
        span_y = y_max - y_min

        for cell in self.cells:
            px = int(np.clip((cell.x - x_min) / span_x * 63, 0, 63))
            py = int(np.clip((cell.y - y_min) / span_y * 63, 0, 63))
            r = int(pattern[0, py, px] * 255)
            g = int(pattern[1, py, px] * 255)
            b = int(pattern[2, py, px] * 255)
            cell.set_target((r, g, b), expanded=(r + g + b) > 360)
            cell.update()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_body(self):
        """Draws the base octopus shape (arms + mantle) before chromatophores."""
        # Arms (drawn first so mantle sits on top)
        for angle in self.arm_angles:
            pts = _arm_polygon(OCT_CX, OCT_CY, angle,
                               self.ARM_LENGTH, self.ARM_BASE_W)
            pygame.draw.polygon(self.screen, OCTOPUS_SKIN, pts)
            pygame.draw.polygon(self.screen, OCTOPUS_SKIN, pts, 1)

        # Webbing between arm bases — small filled circles at base
        for angle in self.arm_angles:
            mid_angle = angle + math.pi / 8
            wx = OCT_CX + math.cos(mid_angle) * 30
            wy = OCT_CY + math.sin(mid_angle) * 30
            pygame.draw.circle(self.screen, OCTOPUS_SKIN, (int(wx), int(wy)), 14)

        # Mantle (body)
        pygame.draw.ellipse(self.screen, OCTOPUS_SKIN,
                            (OCT_CX - self.MANTLE_RX,
                             OCT_CY - self.MANTLE_RY,
                             self.MANTLE_RX * 2,
                             self.MANTLE_RY * 2))
        # Subtle highlight on mantle
        pygame.draw.ellipse(self.screen, SKIN_HIGHLIGHT,
                            (OCT_CX - 44, OCT_CY - 32, 88, 64))

        # Eyes
        for sign in (-1, 1):
            ex, ey = OCT_CX + sign * 30, OCT_CY - 12
            pygame.draw.circle(self.screen, EYE_OUTER, (ex, ey), 11)
            pygame.draw.circle(self.screen, EYE_INNER,  (ex, ey),  7)
            pygame.draw.circle(self.screen, EYE_GLINT,  (ex - 3, ey - 3), 2)

    def _draw_hud(self, cell_count: int):
        lines = [
            f"Chromatophores: {cell_count}",
            "ESC — quit",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(surf, (12, 10 + i * 18))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            self.screen.blit(self.floor, (0, 0))
            self._draw_body()

            pattern = self._agent_pattern() if self.agent else self._demo_pattern()
            self._apply_pattern(pattern)

            for cell in self.cells:
                cell.draw(self.screen)

            self._draw_hud(len(self.cells))
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_simulation(agent=None):
    sim = OctopusSimulation(agent=agent)
    sim.run()


if __name__ == "__main__":
    run_simulation()
