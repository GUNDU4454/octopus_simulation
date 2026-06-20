import pygame
import pymunk
import pymunk.pygame_util
import math

BG_DEEP    = ( 8,  15,  35)
SEABED_DARK= (55,  40,  18)
SEABED_SURF= (90,  65,  28)
BODY_FILL  = (210,  85,  35)
BODY_RIM   = (255, 140,  60)
SEG_ROOT   = (100, 155, 215)
SEG_TIP    = ( 45,  80, 145)
SEG_GRIP   = (230,  55,  55)
SEG_EXTEND = ( 60, 200,  90)
SEG_PULL   = ( 60, 210, 220)
SEG_SELECT = (255, 220,  40)
GRIP_ANCHOR= (220,  50,  50)
HUD_WHITE  = (210, 220, 230)
HUD_DIM    = ( 90, 105, 125)
HUD_RED    = (220,  70,  70)
HUD_GREEN  = ( 70, 200,  90)
HUD_BLUE   = ( 70, 140, 220)
HUD_YELLOW = (240, 210,  50)


class DebugRenderer:
    def __init__(self, screen, width, height, seabed_y):
        self.screen   = screen
        self.W        = width
        self.H        = height
        self.seabed_y = seabed_y
        self.t        = 0.0
        self.font_sm  = pygame.font.SysFont("monospace", 12)
        self.font_md  = pygame.font.SysFont("monospace", 14, bold=True)
        pymunk.pygame_util.positive_y_is_up = True
        self._draw_opts = pymunk.pygame_util.DrawOptions(screen)
        self._bg = self._build_bg()

    def ws(self, pos):
        return (int(pos[0]), self.H - int(pos[1]))

    def _build_bg(self):
        surf = pygame.Surface((self.W, self.H))
        surf.fill(BG_DEEP)
        return surf

    def draw(self, space, octopus, controller):
        self.t += 0.016
        self.screen.blit(self._bg, (0, 0))
        self._draw_seabed()
        for i, tent in enumerate(octopus.tentacles):
            self._draw_tentacle(tent, i, i == controller.selected)
        self._draw_body(octopus)
        if controller.show_pymunk:
            space.debug_draw(self._draw_opts)
        if controller.show_forces:
            self._draw_forces(octopus)
        self._draw_hud(octopus, controller)
        self._draw_controls()
        self._draw_strip(octopus, controller)

    def _draw_seabed(self):
        sy = self.H - int(self.seabed_y)

        for y in range(sy):
            t = y / sy
            r = int(8  + t * 15)
            g = int(15 + t * 35)
            b = int(35 + t * 60)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.W, y))

        for y in range(sy, self.H):
            t = (y - sy) / max(self.H - sy, 1)
            r = int(120 - t * 40)
            g = int(90  - t * 30)
            b = int(40  - t * 20)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.W, y))

        pygame.draw.line(self.screen, (160, 120, 55), (0, sy), (self.W, sy), 3)

        import random
        rng = random.Random(42)
        for i in range(120):
            dx = rng.randint(0, self.W)
            dy = sy + rng.randint(2, 20)
            dr = rng.randint(1, 4)
            shade = rng.randint(100, 160)
            pygame.draw.circle(self.screen, (shade, shade-20, shade-60), (dx, dy), dr)

        for i in range(20):
            rx = rng.randint(0, self.W)
            ry = sy + rng.randint(5, 25)
            rw = rng.randint(4, 12)
            rh = rng.randint(3, 8)
            pygame.draw.ellipse(self.screen, (80, 65, 35), (rx, ry, rw, rh))
    
    def _draw_tentacle(self, tent, idx, selected):
        positions = [self.ws(p) for p in tent.get_positions()]
        n = len(positions)
        if n < 2:
            return
        phase   = tent.state.phase
        gripped = tent.state.is_gripped

        def scheme(pct):
            if selected:       return SEG_SELECT
            if phase == "gripping":  return SEG_GRIP
            if phase == "extending": return SEG_EXTEND
            if phase == "pulling":   return SEG_PULL
            r = int(SEG_ROOT[0]*(1-pct) + SEG_TIP[0]*pct)
            g = int(SEG_ROOT[1]*(1-pct) + SEG_TIP[1]*pct)
            b = int(SEG_ROOT[2]*(1-pct) + SEG_TIP[2]*pct)
            return (r, g, b)

        for i in range(n - 1):
            pct = i / max(n-1, 1)
            w   = max(1, int(5 - pct * 3.5))
            pygame.draw.line(self.screen, scheme(pct),
                             positions[i], positions[i+1], w)

        for i, pos in enumerate(positions):
            pct = i / max(n-1, 1)
            r   = max(2, int(6 - pct * 3))
            if i == n-1 and gripped:
                pygame.draw.circle(self.screen, SEG_GRIP, pos, r+3)
                pygame.draw.circle(self.screen, (255,120,120), pos, r+3, 2)
            else:
                pygame.draw.circle(self.screen, scheme(pct), pos, r)

        if gripped:
            tip    = positions[-1]
            anchor = (tip[0], self.H - int(self.seabed_y))
            pygame.draw.line(self.screen, GRIP_ANCHOR, tip, anchor, 2)
            pygame.draw.circle(self.screen, GRIP_ANCHOR, anchor, 5)

        root = positions[0]
        lbl  = self.font_sm.render(str(idx+1), True,
                                   SEG_SELECT if selected else HUD_DIM)
        self.screen.blit(lbl, (root[0]-5, root[1]-16))

    def _draw_body(self, octopus):
        bpos = self.ws(octopus.body.position)
        r    = 24
        ang  = -octopus.body.angle
        sh   = pygame.Surface((r*2+12, r*2+12), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0,0,0,60), (0,0,r*2+12,r*2+12))
        self.screen.blit(sh, (bpos[0]-r-6, bpos[1]-r+6))
        pygame.draw.circle(self.screen, BODY_FILL, bpos, r)
        pygame.draw.circle(self.screen, BODY_RIM,  bpos, r, 2)
        dx = int(math.cos(ang)*(r-4))
        dy = int(math.sin(ang)*(r-4))
        pygame.draw.line(self.screen, (255,180,80), bpos,
                         (bpos[0]+dx, bpos[1]+dy), 3)
        perp_x = -math.sin(ang)
        perp_y =  math.cos(ang)
        for side in (-1, 1):
            ex = int(bpos[0] + math.cos(ang)*10 + perp_x*9*side)
            ey = int(bpos[1] + math.sin(ang)*10 + perp_y*9*side)
            pygame.draw.circle(self.screen, (200,160,30), (ex,ey), 5)
            pygame.draw.circle(self.screen, (20,10,5),    (ex,ey), 3)
        vel = octopus.body.velocity
        spd = vel.length
        if spd > 5:
            scale = min(spd*0.25, 40)
            vx = int(bpos[0] + vel.x/spd*scale)
            vy = int(bpos[1] - vel.y/spd*scale)
            pygame.draw.line(self.screen, (180,220,255), bpos, (vx,vy), 2)

    def _draw_forces(self, octopus):
        for tent in octopus.tentacles:
            if tent.state.is_gripped and tent.tip:
                tip  = self.ws(tent.tip.position)
                body = self.ws(octopus.body.position)
                pygame.draw.line(self.screen, (255,200,50), tip, body, 1)

    def _draw_hud(self, octopus, ctrl):
        bp  = octopus.body.position
        bv  = octopus.body.velocity
        sel = ctrl.selected
        t   = octopus.tentacles[sel]
        lines = [
            ("OCTOPUS SIM",                       self.font_md, HUD_WHITE),
            (f"pos  ({bp.x:7.1f}, {bp.y:6.1f})", self.font_sm, HUD_DIM),
            (f"vel  ({bv.x:7.1f}, {bv.y:6.1f})", self.font_sm, HUD_DIM),
            (f"dist  {octopus.distance_travelled():+7.1f} px", self.font_sm, HUD_GREEN),
            (f"grips {octopus.gripped_count} / 8", self.font_sm, HUD_BLUE),
            ("",                                  self.font_sm, HUD_DIM),
            (f"arm #{sel+1}  {t.state.phase}",    self.font_sm, HUD_YELLOW),
            (f"gripped  {t.state.is_gripped}",    self.font_sm,
             HUD_RED if t.state.is_gripped else HUD_DIM),
            ("",                                  self.font_sm, HUD_DIM),
            (f"AUTO {'ON' if ctrl.auto_crawl else 'off'}",
             self.font_md, HUD_GREEN if ctrl.auto_crawl else HUD_DIM),
        ]
        y = 10
        for text, font, col in lines:
            if text:
                self.screen.blit(font.render(text, True, col), (12, y))
            y += 17

    def _draw_controls(self):
        controls = [
            ("1-8  select arm",  HUD_DIM),
            ("TAB  cycle arm",   HUD_DIM),
            ("E    extend",      SEG_EXTEND),
            ("C    contract",    SEG_ROOT),
            ("G    grip",        SEG_GRIP),
            ("R    release",     HUD_DIM),
            ("P    pull",        SEG_PULL),
            ("Q    relax all",   HUD_DIM),
            ("SPC  auto-crawl",  HUD_GREEN),
            ("F1   pymunk dbg",  HUD_DIM),
            ("F2   forces",      HUD_DIM),
            ("ESC  quit",        HUD_DIM),
        ]
        y = 10
        for text, col in controls:
            self.screen.blit(self.font_sm.render(text, True, col),
                             (self.W - 175, y))
            y += 17

    def _draw_strip(self, octopus, ctrl):
        sy = self.H - 32
        for i, tent in enumerate(octopus.tentacles):
            x    = 10 + i * 44
            rect = pygame.Rect(x, sy, 38, 22)
            if tent.state.is_gripped:        bg = SEG_GRIP
            elif tent.state.phase == "extending": bg = SEG_EXTEND
            elif tent.state.phase == "pulling":   bg = SEG_PULL
            elif tent.state.phase == "contracting": bg = (80,110,180)
            else:                                 bg = (25,35,60)
            pygame.draw.rect(self.screen, bg, rect)
            border = SEG_SELECT if i == ctrl.selected else (60,75,100)
            pygame.draw.rect(self.screen, border, rect, 2)
            self.screen.blit(
                self.font_sm.render(str(i+1), True, HUD_WHITE),
                (x+14, sy+4)
            )
