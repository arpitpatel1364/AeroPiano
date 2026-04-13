#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   ██╗  ██╗ █████╗ ███╗   ██╗██████╗     ██████╗ ██╗ █████╗  ║
║   ██║  ██║██╔══██╗████╗  ██║██╔══██╗    ██╔══██╗██║██╔══██╗ ║
║   ███████║███████║██╔██╗ ██║██║  ██║    ██████╔╝██║███████║ ║
║   ██╔══██║██╔══██║██║╚██╗██║██║  ██║    ██╔═══╝ ██║██╔══██║ ║
║   ██║  ██║██║  ██║██║ ╚████║██████╔╝    ██║     ██║██║  ██║ ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝     ╚═╝     ╚═╝╚═╝  ╚═╝ ║
║                                                                  ║
║   Play Piano in the Air  •  Live Hand Tracking  •  2 Octaves  ║
╚══════════════════════════════════════════════════════════════════╝

Controls:
  Q / ESC      — Quit
  + / -        — Zoom in / out
  Arrow Up/Dn  — Move piano up / down
  R            — Reset zoom & position
  M            — Toggle mirror mode
  H            — Toggle hand skeleton
  V +/-        — Volume up / down
  S            — Toggle sustain
  SPACE        — Screenshot
"""

import sys
import os
import time
import math
import datetime
import threading

import cv2
import numpy as np
import pygame

from audio   import AudioEngine
from tracker import HandTracker, FINGERTIP_IDS
from piano   import PianoRenderer
from effects import ParticleSystem

# ═══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H   = 1280, 720
FPS_CAP        = 60
CAMERA_ID      = 0
CAM_W, CAM_H   = 1280, 720

# Glow palette for pressed keys
GLOW_COLORS = [
    (120, 170, 255),  # blue
    (180, 100, 255),  # purple
    (80,  230, 200),  # teal
    (255, 160,  80),  # amber
    (255,  90, 140),  # pink
]

# Pinch zoom thresholds
PINCH_CLOSE   = 42   # px — two-hand pinch to start zoom
ZOOM_SCALE    = 0.004


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_rounded_rect(surf: pygame.Surface, color, rect, radius: int,
                        alpha: int = 255, border_color=None, border_w: int = 1):
    """Draw a filled rounded rect with optional border, respecting alpha."""
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA, 32)
    pygame.draw.rect(s, (*color[:3], alpha), (0, 0, rect[2], rect[3]),
                     border_radius=radius)
    if border_color:
        pygame.draw.rect(s, (*border_color[:3], alpha),
                         (0, 0, rect[2], rect[3]),
                         width=border_w, border_radius=radius)
    surf.blit(s, (rect[0], rect[1]))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


# ═══════════════════════════════════════════════════════════════════════════════
#  LOADING SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

def run_loading_screen(screen: pygame.Surface, audio: AudioEngine,
                       all_freqs: list) -> None:
    """Show an animated loading bar while audio is pre-generated."""
    done = [0]
    total = len(all_freqs)

    def _progress(i, n):
        done[0] = i

    thread = audio.preload(all_freqs, on_progress=_progress)

    clock  = pygame.time.Clock()
    t0     = time.time()
    font_l = pygame.font.SysFont("Arial", 28, bold=True)
    font_s = pygame.font.SysFont("Arial", 15)

    while thread.is_alive() or done[0] < total:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        screen.fill((6, 6, 18))

        # Title
        title = font_l.render("AeroPiano", True, (180, 200, 255))
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, WIN_H // 2 - 80))

        sub = font_s.render("Synthesising piano sounds…", True, (120, 130, 160))
        screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2, WIN_H // 2 - 40))

        # Progress bar
        pct   = done[0] / total if total else 1.0
        bar_w = 420
        bx    = (WIN_W - bar_w) // 2
        by    = WIN_H // 2
        _draw_rounded_rect(screen, (25, 25, 50), (bx - 2, by - 2, bar_w + 4, 18), 8,
                            alpha=255, border_color=(60, 70, 120))
        filled_w = int(bar_w * pct)
        if filled_w > 0:
            _draw_rounded_rect(screen, (80, 130, 255), (bx, by, filled_w, 14), 7)

        pct_txt = font_s.render(f"{int(pct*100)}%", True, (160, 175, 220))
        screen.blit(pct_txt, (WIN_W // 2 - pct_txt.get_width() // 2, by + 22))

        # Animated dots
        dots = "●" * (int((time.time() - t0) * 2) % 4)
        dot_txt = font_s.render(dots, True, (80, 110, 200))
        screen.blit(dot_txt, (WIN_W // 2 - dot_txt.get_width() // 2, by + 42))

        pygame.display.flip()
        clock.tick(30)

        if pct >= 1.0 and not thread.is_alive():
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLAY UI (heads-up display)
# ═══════════════════════════════════════════════════════════════════════════════

class HUD:
    def __init__(self, screen_w: int, screen_h: int):
        self.sw, self.sh = screen_w, screen_h
        self.font_b  = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_r  = pygame.font.SysFont("Arial", 14)
        self.font_xl = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_sm = pygame.font.SysFont("Arial", 12)

        self.last_note = ""
        self.note_alpha = 0.0
        self.fps_history = []
        self.message = ""
        self.message_time = 0.0

        # Control Buttons (rect, label, action_name)
        self.buttons = [
            (pygame.Rect(20, self.sh // 2 - 100, 40, 40), "+", "zoom_in"),
            (pygame.Rect(20, self.sh // 2 - 50,  40, 40), "-", "zoom_out"),
            (pygame.Rect(20, self.sh // 2 + 30,  40, 40), "↑", "move_up"),
            (pygame.Rect(20, self.sh // 2 + 80,  40, 40), "↓", "move_down"),
            (pygame.Rect(20, self.sh // 2 + 130, 40, 40), "R", "reset"),
        ]
        self.hover_btn = None

    def set_note(self, name: str) -> None:
        self.last_note = name
        self.note_alpha = 1.0

    def set_message(self, msg: str) -> None:
        self.message = msg
        self.message_time = time.time()

    def check_buttons(self, px: int, py: int) -> str | None:
        for rect, label, name in self.buttons:
            if rect.collidepoint(px, py):
                return name
        return None

    def update(self, dt: float) -> None:
        self.note_alpha = max(0.0, self.note_alpha - dt * 1.8)

    def draw(self, surface: pygame.Surface, fps: float, zoom: float,
             volume: float, sustain: bool, hands: int,
             mirror: bool, show_skeleton: bool,
             particles_count: int) -> None:

        # ── Top bar (Glassmorphism) ──────────────────────────────────────────
        # Subtle gradient background for top bar
        for i in range(46):
            a = int(180 * (1 - i / 46) + 40)
            pygame.draw.line(surface, (15, 15, 35, a), (0, i), (self.sw, i))
        
        # Bottom border highlight
        pygame.draw.line(surface, (80, 120, 255, 60), (0, 46), (self.sw, 46))

        # Title with shadow
        title_bg = self.font_b.render("✋ AeroPiano", True, (20, 30, 60))
        surface.blit(title_bg, (19, 15))
        title = self.font_b.render("✋ AeroPiano", True, (180, 210, 255))
        surface.blit(title, (18, 14))

        # FPS
        fps_col = (100, 255, 140) if fps >= 50 else (255, 220, 80) if fps >= 30 else (255, 100, 100)
        fps_txt = self.font_r.render(f"FPS {fps:.0f}", True, fps_col)
        surface.blit(fps_txt, (self.sw - fps_txt.get_width() - 20, 16))

        # Stats row
        stats = [
            ("ZOOM",   f"{zoom:.2f}×"),
            ("VOL",    f"{int(volume * 100)}%"),
            ("SUSTAIN", "ON" if sustain else "OFF"),
            ("HANDS",  f"● {hands}" if hands > 0 else "None"),
        ]
        x_cursor = self.sw // 2 - 180
        for label, val in stats:
            lbl_s = self.font_sm.render(label, True, (110, 125, 170))
            val_s = self.font_r.render(val, True, (210, 230, 255))
            surface.blit(lbl_s, (x_cursor, 10))
            surface.blit(val_s, (x_cursor, 24))
            x_cursor += 100

        # ── Large note display (centre-bottom) ───────────────────────────────
        if self.note_alpha > 0.05:
            a_i = int(255 * self.note_alpha)
            note_str = self.last_note.replace("#", "♯")
            note_surf = self.font_xl.render(note_str, True, (200, 220, 255))
            note_surf.set_alpha(a_i)
            nx = self.sw // 2 - note_surf.get_width() // 2
            ny = self.sh - 110
            surface.blit(note_surf, (nx, ny))

        # ── Bottom help bar ───────────────────────────────────────────────────
        _draw_rounded_rect(surface, (8, 8, 22), (0, self.sh - 30, self.sw, 30), 0, alpha=180)
        help_items = [
            "Q/ESC Quit", "+/- Zoom", "↑↓ Move", "M Mirror",
            "H Skeleton", "V+/V- Volume", "S Sustain", "SPACE Screenshot",
        ]
        hx = 10
        for item in help_items:
            ht = self.font_sm.render(item, True, (80, 90, 130))
            surface.blit(ht, (hx, self.sh - 22))
            hx += ht.get_width() + 20

        # ── Popup message ─────────────────────────────────────────────────────
        if self.message:
            age = time.time() - self.message_time
            if age < 2.5:
                ma = int(255 * max(0, 1 - age / 2.5))
                msg_s = self.font_b.render(self.message, True, (230, 240, 255))
                msg_s.set_alpha(ma)
                mx = self.sw // 2 - msg_s.get_width() // 2
                surface.blit(msg_s, (mx, self.sh - 68))

        # ── Control Buttons ───────────────────────────────────────────────────
        for rect, label, name in self.buttons:
            is_hover = (self.hover_btn == name)
            # Glass button style
            bg_col = (80, 140, 255, 120) if is_hover else (30, 40, 80, 80)
            brd_col = (150, 200, 255, 180) if is_hover else (80, 100, 180, 100)
            
            _draw_rounded_rect(surface, bg_col, rect, 8, alpha=bg_col[3], 
                                border_color=brd_col, border_w=2)
            
            # Label
            lt = self.font_b.render(label, True, (230, 240, 255))
            surface.blit(lt, (rect.centerx - lt.get_width() // 2, 
                               rect.centery - lt.get_height() // 2))

        # ── Hand-count indicator dots ─────────────────────────────────────────
        for i in range(2):
            col = (80, 200, 120) if i < hands else (50, 55, 80)
            pygame.draw.circle(surface, col, (self.sw - 50 + i * 16, 14), 5)

        # ── Particles count (debug) ───────────────────────────────────────────
        # (disabled for clean look; uncomment to debug)
        # pc_txt = self.font_sm.render(f"P:{particles_count}", True, (80,80,80))
        # surface.blit(pc_txt, (4, self.sh - 44))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Pygame init ──────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WIN_W, WIN_H),
                                     pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("AeroPiano — Play Piano in the Air")
    try:
        pygame.display.set_icon(
            pygame.transform.scale(
                pygame.Surface((32, 32)), (32, 32)))
    except Exception:
        pass

    clock = pygame.time.Clock()

    # ── Sub-systems ──────────────────────────────────────────────────────────
    audio    = AudioEngine()
    piano    = PianoRenderer(WIN_W, WIN_H)
    piano.init_fonts()
    effects  = ParticleSystem()
    tracker  = HandTracker(max_hands=2)
    hud      = HUD(WIN_W, WIN_H)

    # Collect all key frequencies for preloading
    all_freqs = [k.freq for k in piano.keys]

    # ── Loading screen ───────────────────────────────────────────────────────
    run_loading_screen(screen, audio, all_freqs)

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("⚠  Camera not found. Running in no-camera mode.")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── State variables ───────────────────────────────────────────────────────
    mirror        = True
    show_skeleton = True
    sustain       = False
    key_state: dict[int, str | None] = {}   # finger_idx -> key.name or None
    pinch_start_dist = None
    pinch_start_zoom = 1.0
    prev_frame_time  = time.time()

    # Camera frame as pygame surface
    cam_surf: pygame.Surface | None = None

    # Background starfield for no-camera mode
    stars = [(np.random.randint(0, WIN_W),
              np.random.randint(0, WIN_H),
              np.random.uniform(0.3, 1.0)) for _ in range(180)]

    print("\n🎹  AeroPiano started! Press Q or ESC to quit.\n")

    # ════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ════════════════════════════════════════════════════════════════════════
    running = True
    while running:
        now = time.time()
        dt  = min(now - prev_frame_time, 0.05)
        prev_frame_time = now

        # ── Events ────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif ev.key == pygame.K_EQUALS or ev.key == pygame.K_PLUS:
                    piano.set_target_zoom(piano.target_zoom + 0.1)
                    hud.set_message(f"Zoom  {piano.target_zoom:.1f}×")

                elif ev.key == pygame.K_MINUS:
                    piano.set_target_zoom(piano.target_zoom - 0.1)
                    hud.set_message(f"Zoom  {piano.target_zoom:.1f}×")

                elif ev.key == pygame.K_UP:
                    piano.target_offset_y -= 30
                    hud.set_message("Piano ↑")

                elif ev.key == pygame.K_DOWN:
                    piano.target_offset_y += 30
                    hud.set_message("Piano ↓")

                elif ev.key == pygame.K_r:
                    piano.set_target_zoom(1.0)
                    piano.target_offset_y = 0
                    hud.set_message("Reset view")

                elif ev.key == pygame.K_m:
                    mirror = not mirror
                    hud.set_message(f"Mirror {'ON' if mirror else 'OFF'}")

                elif ev.key == pygame.K_h:
                    show_skeleton = not show_skeleton
                    hud.set_message(f"Skeleton {'ON' if show_skeleton else 'OFF'}")

                elif ev.key == pygame.K_s:
                    sustain = not sustain
                    hud.set_message(f"Sustain {'ON' if sustain else 'OFF'}")

                elif ev.key == pygame.K_v:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        audio.set_volume(audio.volume - 0.05)
                    else:
                        audio.set_volume(audio.volume + 0.05)
                    hud.set_message(f"Volume  {int(audio.volume * 100)}%")

                elif ev.key == pygame.K_SPACE:
                    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"screenshot_{ts}.png"
                    pygame.image.save(screen, fname)
                    hud.set_message(f"Saved {fname}")

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    btn = hud.check_buttons(*ev.pos)
                    if btn:
                        _handle_action(btn, piano, hud, audio)

        def _handle_action(action, piano, hud, audio):
            if action == "zoom_in":
                piano.set_target_zoom(piano.target_zoom + 0.1)
                hud.set_message(f"Zoom {piano.target_zoom:.1f}×")
            elif action == "zoom_out":
                piano.set_target_zoom(piano.target_zoom - 0.1)
                hud.set_message(f"Zoom {piano.target_zoom:.1f}×")
            elif action == "move_up":
                piano.target_offset_y -= 30
                hud.set_message("Piano ↑")
            elif action == "move_down":
                piano.target_offset_y += 30
                hud.set_message("Piano ↓")
            elif action == "reset":
                piano.set_target_zoom(1.0)
                piano.target_offset_y = 0
                hud.set_message("Reset view")

        # ── Camera frame ──────────────────────────────────────────────────
        hands_data = []
        raw_frame  = None

        if cap is not None:
            ret, frame = cap.read()
            if ret:
                raw_frame = frame
                if mirror:
                    frame = cv2.flip(frame, 1)

                # Hand tracking
                hands_data = tracker.process(frame)

                # Draw skeleton on frame (OpenCV)
                if show_skeleton and hands_data:
                    skeleton_frame = frame.copy()
                    for hd in hands_data:
                        # Draw thin, glowing connections
                        for conn in [(0,1),(1,2),(2,3),(3,4),
                                     (5,6),(6,7),(7,8),
                                     (9,10),(10,11),(11,12),
                                     (13,14),(14,15),(15,16),
                                     (17,18),(18,19),(19,20),
                                     (0,5),(5,9),(9,13),(13,17),(0,17)]:
                            a, b = hd.landmarks_px[conn[0]], hd.landmarks_px[conn[1]]
                            cv2.line(skeleton_frame, a, b, (140, 240, 200), 1, cv2.LINE_AA)
                        
                        # Draw small, sharp joint dots
                        for idx, (px, py) in enumerate(hd.landmarks_px):
                            col = (80, 255, 180) if idx not in [4,8,12,16,20] else (255, 120, 120)
                            cv2.circle(skeleton_frame, (px, py), 3, col, -1, cv2.LINE_AA)
                            cv2.circle(skeleton_frame, (px, py), 3, (255,255,255), 1, cv2.LINE_AA)
                    frame = skeleton_frame

                # Convert BGR → RGB → pygame Surface
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (WIN_W, WIN_H))
                cam_surf = pygame.surfarray.make_surface(np.rot90(rgb))

        # ── Render background ──────────────────────────────────────────────
        if cam_surf is not None:
            # High transparency for the 'Clean/Cyber' look
            cam_surf.set_alpha(70)
            screen.fill((6, 6, 18))
            screen.blit(cam_surf, (0, 0))
        else:
            # Animated starfield (no camera)
            screen.fill((5, 5, 16))
            t_star = time.time()
            for (sx, sy, spd) in stars:
                b = int(80 + 60 * math.sin(t_star * spd + sx * 0.1))
                pygame.draw.circle(screen, (b, b, b + 20), (sx, sy), 1)

        # ── Dark vignette overlay ──────────────────────────────────────────
        vig = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA, 32)
        # Simple radial darken at edges
        pygame.draw.rect(vig, (0, 0, 0, 0), (60, 40, WIN_W - 120, WIN_H - 80))
        # Borders
        for side, rect in [
            ((0,0,0,100), (0, 0, 60, WIN_H)),
            ((0,0,0,100), (WIN_W-60, 0, 60, WIN_H)),
            ((0,0,0,60),  (0, 0, WIN_W, 40)),
            ((0,0,0,60),  (0, WIN_H-40, WIN_W, 40)),
        ]:
            pygame.draw.rect(vig, side, rect)
        screen.blit(vig, (0, 0))

        # ── Process hand → piano interaction ──────────────────────────────
        active_fingers: list = []   # (px, py, finger_idx)
        current_active_finger_ids = set()
        hud.hover_btn = None

        if hands_data:
            # ── Two-hand pinch zoom ──────────────────────────────────────
            if len(hands_data) == 2:
                # Scale landmarks to window size
                fh, fw = raw_frame.shape[:2] if raw_frame is not None else (CAM_H, CAM_W)
                def to_win(pt): return (int(pt[0] * WIN_W / fw), int(pt[1] * WIN_H / fh))
                
                tip_a = to_win(hands_data[0].landmarks_px[8])
                tip_b = to_win(hands_data[1].landmarks_px[8])
                dist  = math.dist(tip_a, tip_b)
                if pinch_start_dist is None:
                    if dist < 180:
                        pinch_start_dist = dist
                        pinch_start_zoom = piano.zoom
                else:
                    delta = dist - pinch_start_dist
                    piano.set_target_zoom(pinch_start_zoom + delta * ZOOM_SCALE)
            else:
                pinch_start_dist = None

            # ── Per-finger hit detection ───────────────────────────────
            fh, fw = raw_frame.shape[:2] if raw_frame is not None else (CAM_H, CAM_W)
            
            for hi, hd in enumerate(hands_data):
                for fi, tip_id in enumerate(FINGERTIP_IDS):
                    global_fi = hi * 5 + fi   # 0–9 across both hands
                    current_active_finger_ids.add(global_fi)
                    
                    # Scale camera coords to window coords
                    raw_px, raw_py = hd.landmarks_px[tip_id]
                    px = int(raw_px * WIN_W / fw)
                    py = int(raw_py * WIN_H / fh)
                    
                    active_fingers.append((px, py, fi))
                    
                    # ── Button interaction ──
                    btn_hover = hud.check_buttons(px, py)
                    if btn_hover:
                        hud.hover_btn = btn_hover
                        # Finger tap/click on button (if tip is deep enough or just immediate)
                        # For now, let's treat any finger hover as a 'hold' that triggers slowly
                        # or just trigger on 'hit' if not already on it
                        if prev_key_name != btn_hover:
                            _handle_action(btn_hover, piano, hud, audio)
                            key_state[global_fi] = btn_hover
                        continue

                    hit_key = piano.get_hit_key(px, py)

                    prev_key_name = key_state.get(global_fi)

                    if hit_key is not None:
                        if prev_key_name != hit_key.name:
                            # Finger moved from one key (or none) to another
                            if prev_key_name is not None:
                                # Release the old one
                                for k in piano.keys:
                                    if k.name == prev_key_name:
                                        k.release(global_fi)
                            
                            # Hit the new one
                            if hit_key.hit(global_fi):
                                # NEW press
                                vel = 0.65 + 0.35 * (fi / 4.0)
                                audio.play(hit_key.freq, velocity=vel)
                                hud.set_note(hit_key.name)

                                # Particle burst
                                kx, ky, kw, kh = piano.key_rect(hit_key)
                                cx, cy = int(kx + kw / 2), int(ky + kh // 4)
                                color_idx = (ord(hit_key.name[0]) + hi) % len(GLOW_COLORS)
                                effects.emit_burst(cx, cy, GLOW_COLORS[color_idx], count=22)
                            
                            key_state[global_fi] = hit_key.name
                    else:
                        # No hit
                        if prev_key_name is not None:
                            for k in piano.keys:
                                if k.name == prev_key_name:
                                    k.release(global_fi)
                            key_state[global_fi] = None

        else:
            pinch_start_dist = None
        
        # Release keys for fingers that are no longer detected
        for fid in list(key_state.keys()):
            if fid not in current_active_finger_ids:
                old_key_name = key_state[fid]
                if old_key_name:
                    for k in piano.keys:
                        if k.name == old_key_name:
                            k.release(fid)
                del key_state[fid]

        # ── Update systems ────────────────────────────────────────────────
        piano.update(dt)
        effects.update(dt)
        hud.update(dt)

        # ── Draw piano ────────────────────────────────────────────────────
        piano.draw(screen, active_fingers)

        # ── Draw particles ────────────────────────────────────────────────
        effects.draw(screen)

        # ── Draw HUD ──────────────────────────────────────────────────────
        fps_now = clock.get_fps()
        hud.draw(
            surface=screen,
            fps=fps_now,
            zoom=piano.zoom,
            volume=audio.volume,
            sustain=sustain,
            hands=len(hands_data),
            mirror=mirror,
            show_skeleton=show_skeleton,
            particles_count=effects.count,
        )

        # ── Present ───────────────────────────────────────────────────────
        pygame.display.flip()
        clock.tick(FPS_CAP)

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("\n🎹  AeroPiano closed. Goodbye!\n")
    if cap is not None:
        cap.release()
    tracker.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
