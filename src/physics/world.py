import pymunk

GRAVITY     = (0, -120)
DAMPING     = 0.92
DRAG_LINEAR = 0.6
DRAG_QUAD   = 0.003
TIMESTEP    = 1 / 180
SUBSTEPS    = 3
SEABED_Y    = 60
SEABED_FRICTION   = 3.5
SEABED_ELASTICITY = 0.05
CAT_GROUND   = 0b001
CAT_TENTACLE = 0b010
CAT_BODY     = 0b100


class World:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.space  = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.damping = DAMPING
        self._build_seabed()
        self._build_walls()

    def _build_seabed(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        seg  = pymunk.Segment(
            body,
            (-self.width * 2, SEABED_Y),
            ( self.width * 3, SEABED_Y),
            4,
        )
        seg.friction   = SEABED_FRICTION
        seg.elasticity = SEABED_ELASTICITY
        seg.filter     = pymunk.ShapeFilter(categories=CAT_GROUND)
        self.space.add(body, seg)
        self.ground_body = body
        self.seabed_y    = SEABED_Y

    def _build_walls(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        lw = pymunk.Segment(body, (0, 0), (0, self.height * 2), 3)
        rw = pymunk.Segment(body, (self.width, 0), (self.width, self.height * 2), 3)
        for w in (lw, rw):
            w.friction   = 0.2
            w.elasticity = 0.3
        self.space.add(body, lw, rw)

    def _apply_drag(self):
        for body in self.space.bodies:
            if body.body_type != pymunk.Body.DYNAMIC:
                continue
            v   = body.velocity
            spd = v.length
            if spd < 1e-4:
                continue
            mag  = DRAG_LINEAR * spd + DRAG_QUAD * spd * spd
            drag = -v.normalized() * mag
            body.apply_force_at_world_point(drag, body.position)

    def step(self, dt=TIMESTEP):
        self._apply_drag()
        self.space.step(dt)
        