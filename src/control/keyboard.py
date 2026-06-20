import pygame
from src.entities.octopus import Octopus

FRONT_ARMS = [7, 0, 1]
SIDE_ARMS  = [2, 6]
BACK_ARMS  = [3, 4, 5]


class KeyboardController:

    PHASE_DURATIONS = [0.45, 0.20, 0.70, 0.35]

    def __init__(self, octopus):
        self.octopus    = octopus
        self.selected   = 0
        self.auto_crawl = False
        self._ac_timer  = 0.0
        self._ac_phase  = 0
        self._ac_group  = 0
        self.show_pymunk  = False
        self.show_forces  = False
        self.show_indices = True

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN:
            return
        k   = event.key
        oct_ = self.octopus
        t   = oct_.tentacles[self.selected]
        gnd = oct_.ground_body
        sea = oct_.seabed_y

        sel_keys = [
            pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
            pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
        ]
        for i, key in enumerate(sel_keys):
            if k == key:
                self.selected = i
                return

        if k == pygame.K_TAB:
            self.selected = (self.selected + 1) % 8
        elif k == pygame.K_e:
            t.extend()
        elif k == pygame.K_c:
            t.contract()
        elif k == pygame.K_g:
            t.grip(gnd, sea)
        elif k == pygame.K_r:
            t.release()
        elif k == pygame.K_p:
            t.pull()
        elif k == pygame.K_q:
            for tent in oct_.tentacles:
                tent.release()
                tent.relax()
        elif k == pygame.K_SPACE:
            self.auto_crawl = not self.auto_crawl
            if self.auto_crawl:
                self._ac_timer = 0.0
                self._ac_phase = 0
                self._ac_group = 0
        elif k == pygame.K_F1:
            self.show_pymunk = not self.show_pymunk
        elif k == pygame.K_F2:
            self.show_forces = not self.show_forces
        elif k == pygame.K_F3:
            self.show_indices = not self.show_indices

    def update(self, dt):
        if self.auto_crawl:
            self._run_auto_crawl(dt)

    def _run_auto_crawl(self, dt):
        self._ac_timer += dt
        if self._ac_timer < self.PHASE_DURATIONS[self._ac_phase]:
            self._apply_phase_forces()
            return
        self._ac_timer = 0.0
        self._ac_phase = (self._ac_phase + 1) % 4
        if self._ac_phase == 0:
            self._ac_group = 1 - self._ac_group

    def _apply_phase_forces(self):
        oct_ = self.octopus
        gnd  = oct_.ground_body
        sea  = oct_.seabed_y

        if self._ac_group == 0:
            active  = [oct_.tentacles[i] for i in FRONT_ARMS]
            passive = [oct_.tentacles[i] for i in BACK_ARMS]
        else:
            active  = [oct_.tentacles[i] for i in BACK_ARMS]
            passive = [oct_.tentacles[i] for i in FRONT_ARMS]

        side = [oct_.tentacles[i] for i in SIDE_ARMS]

        if self._ac_phase == 0:
            for t in active:  t.extend(0.9)
            for t in passive: t.relax()
            for t in side:    t.relax()
        elif self._ac_phase == 1:
            for t in active:
                if not t.state.is_gripped:
                    t.grip(gnd, sea)
            for t in side: t.grip(gnd, sea)
        elif self._ac_phase == 2:
            for t in active:
                t.contract(1.0)
                t.pull(1.0)
            for t in passive:
                t.release()
                t.extend(0.5)
        elif self._ac_phase == 3:
            for t in active:  t.release(); t.relax()
            for t in side:    t.release()
            for t in passive: t.relax()