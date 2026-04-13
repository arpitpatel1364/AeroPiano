"""
piano.py — Piano Key Definitions, Layout, Hit-Detection & Rendering
Renders a photorealistic floating piano with 3-D depth and glow effects.
"""

import math
import time
import pygame
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─── Note / frequency helpers ─────────────────────────────────────────────────

CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
WHITE_SEMITONES  = [0, 2, 4, 5, 7, 9, 11]
BLACK_SEMITONES  = [1, 3, 6, 8, 10]
# x-offset (in white-key units) of each black key from the left of its octave
BLACK_X_OFFSETS  = [0.67, 1.67, 3.70, 4.67, 5.67]


def midi_freq(n: int) -> float:
    return 440.0 * 2.0 ** ((n - 69) / 12.0)


# ─── Key data ─────────────────────────────────────────────────────────────────

@dataclass
class PianoKey:
    name:     str        # e.g. "C4"
    freq:     float
    is_black: bool
    white_col: float     # x position in white-key-width units

    # Runtime state
    pressed:    bool  = False
    press_time: float = 0.0
    press_alpha: float = 0.0   # 0 → 1 glow strength
    finger_ids: set   = field(default_factory=set)  # which finger indices are on it

    def hit(self, finger_id: int) -> bool:
        """Returns True if this is a *new* press (edge trigger) or adding a finger."""
        self.finger_ids.add(finger_id)
        if not self.pressed:
            self.pressed    = True
            self.press_time = time.time()
            self.press_alpha = 1.0
            return True
        return False

    def release(self, finger_id: int) -> bool:
        """Returns True if the key is fully released (all fingers gone)."""
        if finger_id in self.finger_ids:
            self.finger_ids.remove(finger_id)
        if not self.finger_ids:
            self.pressed = False
            return True
        return False

    def update(self, dt: float) -> None:
        if not self.pressed:
            self.press_alpha = max(0.0, self.press_alpha - dt * 3.5)


# ─── Build key list ────────────────────────────────────────────────────────────

def build_keys(num_octaves: int = 2, start_midi: int = 60) -> List[PianoKey]:
    keys: List[PianoKey] = []
    for oct_i in range(num_octaves):
        base = start_midi + oct_i * 12
        oct_white_offset = oct_i * 7
        # White keys
        for w_i, semi in enumerate(WHITE_SEMITONES):
            midi = base + semi
            keys.append(PianoKey(
                name=CHROMATIC[semi] + str(4 + oct_i),
                freq=midi_freq(midi),
                is_black=False,
                white_col=oct_white_offset + w_i,
            ))
        # Black keys
        for b_i, (semi, x_off) in enumerate(
                zip(BLACK_SEMITONES, BLACK_X_OFFSETS)):
            midi = base + semi
            keys.append(PianoKey(
                name=CHROMATIC[semi] + str(4 + oct_i),
                freq=midi_freq(midi),
                is_black=True,
                white_col=oct_white_offset + x_off,
            ))
    return keys


# ─── Colours ──────────────────────────────────────────────────────────────────

C_WK_TOP       = (248, 250, 255)
C_WK_MID       = (220, 224, 240)
C_WK_SHAD      = (190, 195, 215)
C_WK_EDGE      = (150, 155, 175)
C_WK_3D        = (120, 125, 145)
C_WK_PRESSED   = (160, 190, 255)
C_WK_PR_BRIGHT = (210, 230, 255)

C_BK_FACE      = (16, 16, 28)
C_BK_TOP       = (50, 52, 70)
C_BK_EDGE      = (8,  8, 16)
C_BK_PRESSED   = (80, 115, 240)
C_BK_PR_BRIGHT = (140, 170, 255)

C_BODY_TOP     = (30, 28, 50)
C_BODY_FACE    = (20, 18, 38)
C_BODY_EDGE    = (60, 55, 100)

C_GLOW_1       = (120, 160, 255)
C_GLOW_2       = (200, 140, 255)

FINGER_COLORS = [
    (255,  80, 120),  # thumb
    (80,  200, 255),  # index
    (80,  255, 160),  # middle
    (255, 200,  60),  # ring
    (255, 120,  60),  # pinky
]


# ─── Piano Renderer ───────────────────────────────────────────────────────────

class PianoRenderer:
    """
    Renders a floating, zoomable, 3-D styled piano onto a Pygame surface.
    Also handles hit-detection given fingertip pixel coordinates.
    """

    NUM_WHITE = 14          # 2 octaves

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.zoom      = 1.0          # user-controlled
        self.target_zoom = 1.0
        self.offset_y  = 0.0         # vertical drag offset
        self.target_offset_y = 0.0

        # Base (unscaled) white key dimensions
        self.BASE_WK_W  = 58
        self.BASE_WK_H  = 185
        self.BASE_BK_W  = 36
        self.BASE_BK_H  = 115
        self.BASE_3D_H  = 14         # depth of the 3-D bottom edge
        self.BASE_LID_H = 22         # piano lid above keys

        self.keys = build_keys(num_octaves=2, start_midi=60)
        self._label_font  = None
        self._info_font   = None
        self._glow_cache: dict = {}

    def init_fonts(self) -> None:
        pygame.font.init()
        self._label_font = pygame.font.SysFont("Arial", 13, bold=True)
        self._info_font  = pygame.font.SysFont("Arial", 11)

    # ── Derived geometry ──────────────────────────────────────────────────────

    @property
    def wk_w(self) -> float:
        return self.BASE_WK_W * self.zoom

    @property
    def wk_h(self) -> float:
        return self.BASE_WK_H * self.zoom

    @property
    def bk_w(self) -> float:
        return self.BASE_BK_W * self.zoom

    @property
    def bk_h(self) -> float:
        return self.BASE_BK_H * self.zoom

    @property
    def depth_h(self) -> float:
        return self.BASE_3D_H * self.zoom

    @property
    def lid_h(self) -> float:
        return self.BASE_LID_H * self.zoom

    @property
    def piano_total_w(self) -> float:
        return self.wk_w * self.NUM_WHITE

    @property
    def piano_left(self) -> float:
        return (self.screen_w - self.piano_total_w) / 2.0

    @property
    def piano_top(self) -> float:
        base = self.screen_h * 0.52 + self.offset_y
        return base

    # ── Key screen rect ───────────────────────────────────────────────────────

    def key_rect(self, key: PianoKey) -> Tuple[float, float, float, float]:
        """Return (x, y, w, h) in screen pixels for a key's top face."""
        x = self.piano_left + key.white_col * self.wk_w
        y = self.piano_top
        if key.is_black:
            w, h = self.bk_w, self.bk_h
            x += (self.wk_w - self.bk_w) * 0.5  # centre black key on white
        else:
            w, h = self.wk_w - 1, self.wk_h
        return x, y, w, h

    # ── Hit detection ─────────────────────────────────────────────────────────

    def get_hit_key(self, px: int, py: int) -> Optional[PianoKey]:
        """
        Return the key under pixel (px, py), or None.
        Black keys have priority over white.
        Hit zone is extended ±30 px above the key top.
        """
        HIT_MARGIN = int(30 * self.zoom)
        # Check black keys first (they sit on top)
        for key in self.keys:
            if not key.is_black:
                continue
            x, y, w, h = self.key_rect(key)
            if (x - 2 <= px <= x + w + 2 and
                    y - HIT_MARGIN <= py <= y + h):
                return key
        # Then white keys
        for key in self.keys:
            if key.is_black:
                continue
            x, y, w, h = self.key_rect(key)
            if (x <= px <= x + w and
                    y - HIT_MARGIN <= py <= y + h):
                return key
        return None

    # ── Zoom & offset smoothing ───────────────────────────────────────────────

    def set_target_zoom(self, z: float) -> None:
        self.target_zoom = max(0.50, min(2.2, z))

    def update(self, dt: float) -> None:
        self.zoom     += (self.target_zoom - self.zoom) * min(1.0, dt * 8.0)
        self.offset_y += (self.target_offset_y - self.offset_y) * min(1.0, dt * 6.0)
        for key in self.keys:
            key.update(dt)

    # ── Glow surface helper ───────────────────────────────────────────────────

    def _glow_surf(self, w: int, h: int,
                   color: Tuple[int, int, int], alpha: int) -> pygame.Surface:
        """Return a pre-computed radial glow surface (cached)."""
        cache_key = (w, h, color, alpha)
        if cache_key in self._glow_cache:
            return self._glow_cache[cache_key]

        surf = pygame.Surface((w * 3, h * 2), pygame.SRCALPHA, 32)
        cx, cy = w * 3 // 2, h // 2
        steps = 12
        for i in range(steps, 0, -1):
            a = int(alpha * (i / steps) ** 2.2)
            rw = int(w * 1.4 * i / steps)
            rh = int(h * 0.7 * i / steps)
            pygame.draw.ellipse(
                surf,
                (*color, a),
                (cx - rw, cy - rh, rw * 2, rh * 2),
            )
        self._glow_cache[cache_key] = surf
        return surf

    # ── Core draw ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface,
             active_fingers: List[Tuple[int, int, int]]) -> None:
        """
        Render the full piano.
        active_fingers: list of (px, py, finger_idx) for drawing cursor dots.
        """
        self._draw_body(surface)
        self._draw_white_keys(surface)
        self._draw_black_keys(surface)
        self._draw_note_labels(surface)
        self._draw_finger_cursors(surface, active_fingers)

    # ── Body (piano cabinet) ──────────────────────────────────────────────────

    def _draw_body(self, surface: pygame.Surface) -> None:
        lx = int(self.piano_left - 10 * self.zoom)
        ty = int(self.piano_top  - self.lid_h)
        total_w = int(self.piano_total_w + 20 * self.zoom)
        total_h = int(self.wk_h + self.depth_h + self.lid_h + 10 * self.zoom)

        # Body Surface with alpha
        body_surf = pygame.Surface((total_w, total_h), pygame.SRCALPHA, 32)
        
        # FULLY TRANSPARENT Body (very faint glass tint)
        body_alpha = 35
        pygame.draw.rect(body_surf, (20, 30, 60, body_alpha), (0, 0, total_w, total_h), 
                         border_radius=int(8*self.zoom))
        
        # Border Glow
        for i in range(3):
            ga = int(100 / (i + 1))
            pygame.draw.rect(body_surf, (80, 140, 255, ga), (0, 0, total_w, total_h), 
                             width=1+i, border_radius=int(8*self.zoom))

        # Lid (top strip - also glassy)
        lid_h = int(self.lid_h)
        pygame.draw.rect(body_surf, (60, 100, 255, 60), (0, 0, total_w, lid_h),
                         border_radius=int(8*self.zoom))
        
        # Brand strip highlight
        pygame.draw.rect(body_surf, (255, 255, 255, 60), (10, 4, total_w - 20, 2))
        
        surface.blit(body_surf, (lx, ty))

        # Shadow beneath piano (subtler)
        shadow = pygame.Surface((total_w + 40, 30), pygame.SRCALPHA, 32)
        for i in range(15):
            a = int(80 * (1 - i / 15) ** 2)
            pygame.draw.ellipse(shadow, (0, 0, 0, a),
                                (i, i, total_w + 40 - 2*i, 30 - 2*i))
        surface.blit(shadow, (lx - 20, ty + total_h + 5))

    # ── White keys ────────────────────────────────────────────────────────────

    def _draw_white_keys(self, surface: pygame.Surface) -> None:
        for key in self.keys:
            if key.is_black:
                continue
            x, y, w, h = self.key_rect(key)
            xi, yi, wi, hi = int(x), int(y), int(w), int(h)
            di = int(self.depth_h)

            pressed = key.pressed
            alpha   = key.press_alpha

            # ── Glow ──
            if alpha > 0.02:
                gcolor = C_GLOW_1 if not pressed else C_GLOW_2
                ga = int(180 * alpha)
                gs = self._glow_surf(wi, hi, gcolor, ga)
                surface.blit(gs, (xi - wi, yi - hi // 2),
                             special_flags=pygame.BLEND_PREMULTIPLIED)

            # ── 3-D bottom depth ──
            pts_3d = [
                (xi,      yi + hi),
                (xi + wi, yi + hi),
                (xi + wi + 2, yi + hi + di),
                (xi - 2,  yi + hi + di),
            ]
            pygame.draw.polygon(surface, C_WK_3D, pts_3d)

            # ── Main key face ──
            # FULLY TRANSPARENT key face (faint glow tint)
            key_surf = pygame.Surface((wi, hi), pygame.SRCALPHA, 32)
            k_alpha = 15 if not pressed else 60
            face_col = (*C_WK_PRESSED[:3], k_alpha) if pressed else (200, 220, 255, k_alpha)
            pygame.draw.rect(key_surf, face_col, (0, 0, wi, hi))
            
            # Border Glow
            glow_c = (100, 160, 255) if not pressed else (240, 60, 255)
            glow_w = 2 if not pressed else 3
            for i in range(glow_w):
                ga = int((140 if pressed else 90) / (i + 1))
                pygame.draw.rect(key_surf, (*glow_c, ga), (0, 0, wi, hi), width=1+i)
            
            # Subtle vertical gradient
            grad_h = max(1, hi // 4)
            for row in range(grad_h):
                aa = int(40 * (1 - row / grad_h))
                pygame.draw.line(key_surf, (*glow_c, aa), (0, row), (wi, row))
            
            surface.blit(key_surf, (xi, yi))

            # ── Key separator / edge ── (fainter)
            pygame.draw.rect(surface, (*C_WK_EDGE, 40), (xi, yi, wi, hi), 1)

            # ── Note name at bottom ──
            if self._label_font and not key.is_black:
                note_name = key.name.replace('#', '♯')
                col = (100, 120, 180) if not pressed else (200, 210, 255)
                txt = self._label_font.render(note_name, True, col)
                tw = txt.get_width()
                txt.set_alpha(180 if not pressed else 255)
                surface.blit(txt, (xi + (wi - tw) // 2, yi + hi - 22))

    # ── Black keys ────────────────────────────────────────────────────────────

    def _draw_black_keys(self, surface: pygame.Surface) -> None:
        for key in self.keys:
            if not key.is_black:
                continue
            x, y, w, h = self.key_rect(key)
            xi, yi, wi, hi = int(x), int(y), int(w), int(h)
            di = int(self.depth_h * 0.6)

            pressed = key.pressed
            alpha   = key.press_alpha

            # ── Glow ──
            if alpha > 0.02:
                ga = int(220 * alpha)
                gs = self._glow_surf(wi, hi, C_GLOW_1, ga)
                surface.blit(gs, (xi - wi, yi - hi // 3),
                             special_flags=pygame.BLEND_PREMULTIPLIED)

            # ── Drop shadow on white keys below ──
            shad = pygame.Surface((wi + 6, 12), pygame.SRCALPHA, 32)
            shad.fill((0, 0, 0, 80))
            surface.blit(shad, (xi - 3, yi + hi + 2))

            # ── 3-D bottom ──
            pts_3d = [
                (xi,      yi + hi),
                (xi + wi, yi + hi),
                (xi + wi + 1, yi + hi + di),
                (xi - 1,  yi + hi + di),
            ]
            pygame.draw.polygon(surface, C_BK_EDGE, pts_3d)

            # ── Main face ──
            # FULLY TRANSPARENT black keys
            bk_surf = pygame.Surface((wi, hi), pygame.SRCALPHA, 32)
            bk_alpha = 30 if not pressed else 80
            face_col = (50, 80, 160, bk_alpha) if pressed else (15, 20, 45, bk_alpha)
            pygame.draw.rect(bk_surf, face_col, (0, 0, wi, hi),
                             border_radius=int(3 * self.zoom))

            # Border Glow
            glow_c = (80, 140, 255) if not pressed else (180, 80, 255)
            for i in range(2):
                ga = int(120 / (i + 1))
                pygame.draw.rect(bk_surf, (*glow_c, ga), (0, 0, wi, hi), 
                                 width=1+i, border_radius=int(3 * self.zoom))

            # Top highlight strip
            hl_h = max(2, int(hi * 0.12))
            pygame.draw.rect(bk_surf, (255, 255, 255, 40), (2, 2, wi - 4, hl_h),
                             border_radius=2)
            
            surface.blit(bk_surf, (xi, yi))

    # ── Note labels that float above pressed keys ──────────────────────────────

    def _draw_note_labels(self, surface: pygame.Surface) -> None:
        if not self._label_font:
            return
        for key in self.keys:
            if key.press_alpha < 0.05:
                continue
            x, y, w, h = self.key_rect(key)
            cx = int(x + w / 2)
            label_y = int(y - 38 * self.zoom * key.press_alpha)

            a = int(255 * key.press_alpha)
            note_str = key.name.replace('#', '♯')
            txt = self._label_font.render(note_str, True, (220, 235, 255))
            tw, th = txt.get_width(), txt.get_height()

            # Background pill
            pill = pygame.Surface((tw + 14, th + 8), pygame.SRCALPHA, 32)
            pygame.draw.rect(pill, (60, 90, 200, a // 2),
                             (0, 0, tw + 14, th + 8), border_radius=6)
            surface.blit(pill, (cx - (tw + 14) // 2, label_y - th // 2 - 4))

            txt_surf = txt.copy()
            txt_surf.set_alpha(a)
            surface.blit(txt_surf, (cx - tw // 2, label_y - th // 2))

    # ── Finger cursor dots ────────────────────────────────────────────────────

    def _draw_finger_cursors(self, surface: pygame.Surface,
                             fingers: List[Tuple[int, int, int]]) -> None:
        for (px, py, fid) in fingers:
            color = FINGER_COLORS[fid % len(FINGER_COLORS)]
            # Outer glow ring
            for r, a in [(14, 40), (10, 80), (6, 160)]:
                circ = pygame.Surface((r*2, r*2), pygame.SRCALPHA, 32)
                pygame.draw.circle(circ, (*color, a), (r, r), r)
                surface.blit(circ, (px - r, py - r))
            # Solid core
            pygame.draw.circle(surface, color, (px, py), 5)
            pygame.draw.circle(surface, (255, 255, 255), (px, py), 3)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _lerp_color(a: Tuple, b: Tuple, t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )
