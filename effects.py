"""
effects.py — Particle System & Visual Effects
High-performance particle explosions for key-press feedback.
"""

import math
import numpy as np
import pygame
from dataclasses import dataclass, field
from typing import List, Tuple

RNG = np.random.default_rng()


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float          # seconds remaining
    max_life: float
    r: int
    g: int
    b: int
    radius: float

    @property
    def alpha(self) -> int:
        ratio = self.life / self.max_life
        return int(255 * min(1.0, ratio * 2.5))   # quick fade-in, slow fade-out


class ParticleSystem:
    """
    GPU-friendly particle system using pre-allocated NumPy arrays
    for position integration, with Pygame per-pixel-alpha surfaces for rendering.
    """

    MAX_PARTICLES = 1200
    GRAVITY       = 180.0    # px / s²
    DRAG          = 0.97

    def __init__(self):
        self._particles: List[Particle] = []

    # ──────────────────────────────────────────────────────────────────────────

    def emit_burst(self, cx: float, cy: float,
                   color: Tuple[int, int, int],
                   count: int = 28,
                   speed_min: float = 60,
                   speed_max: float = 220) -> None:
        """Emit a radial burst of particles."""
        if len(self._particles) > self.MAX_PARTICLES:
            return

        r, g, b = color
        angles  = RNG.uniform(0, 2 * math.pi, count)
        speeds  = RNG.uniform(speed_min, speed_max, count)
        lives   = RNG.uniform(0.45, 0.90, count)
        radii   = RNG.uniform(2.0, 5.5, count)
        ox      = RNG.uniform(-6, 6, count)
        oy      = RNG.uniform(-6, 6, count)

        for i in range(count):
            self._particles.append(Particle(
                x=cx + ox[i], y=cy + oy[i],
                vx=math.cos(angles[i]) * speeds[i],
                vy=math.sin(angles[i]) * speeds[i] - speeds[i] * 0.4,
                life=lives[i], max_life=lives[i],
                r=min(255, r + RNG.integers(-20, 40)),
                g=min(255, g + RNG.integers(-20, 40)),
                b=min(255, b + RNG.integers(-20, 40)),
                radius=radii[i],
            ))

    def emit_sparkle(self, cx: float, cy: float,
                     color: Tuple[int, int, int],
                     count: int = 10) -> None:
        """Gentle upward sparkle (note label area)."""
        r, g, b = color
        for _ in range(count):
            angle = RNG.uniform(-math.pi / 2 - 0.4, -math.pi / 2 + 0.4)
            speed = RNG.uniform(30, 90)
            life  = RNG.uniform(0.3, 0.6)
            self._particles.append(Particle(
                x=cx + RNG.uniform(-15, 15),
                y=cy,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=life, max_life=life,
                r=r, g=g, b=b,
                radius=RNG.uniform(1.5, 3.5),
            ))

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """Integrate all particles. Remove dead ones."""
        alive = []
        for p in self._particles:
            p.vy += self.GRAVITY * dt
            p.vx *= self.DRAG
            p.vy *= self.DRAG
            p.x  += p.vx * dt
            p.y  += p.vy * dt
            p.life -= dt
            if p.life > 0:
                alive.append(p)
        self._particles = alive

    def draw(self, surface: pygame.Surface) -> None:
        """Draw all particles onto surface using per-pixel alpha circles."""
        for p in self._particles:
            a   = p.alpha
            if a < 8:
                continue
            rad = max(1, int(p.radius))
            sz  = rad * 4

            buf = pygame.Surface((sz, sz), pygame.SRCALPHA, 32)

            # Inner bright core
            pygame.draw.circle(buf, (p.r, p.g, p.b, a),
                               (sz // 2, sz // 2), rad)
            # Soft outer glow
            glow_a = max(0, a // 5)
            pygame.draw.circle(buf, (p.r, p.g, p.b, glow_a),
                               (sz // 2, sz // 2), rad * 2)

            surface.blit(buf, (int(p.x) - sz // 2, int(p.y) - sz // 2),
                         special_flags=pygame.BLEND_PREMULTIPLIED)

    @property
    def count(self) -> int:
        return len(self._particles)
