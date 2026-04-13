"""
audio.py — Realistic Piano Sound Synthesis Engine
Uses per-harmonic exponential decay for authentic piano timbre.
"""

import numpy as np
import pygame
import math
import threading
from typing import Callable, List

SAMPLE_RATE = 44100


class AudioEngine:
    """
    Generates realistic piano tones using additive synthesis with
    per-harmonic exponential decay and inharmonicity modelling.
    """

    # (harmonic_number, relative_amplitude, decay_rate_per_second)
    HARMONIC_TABLE = [
        (1,  1.000, 1.8),
        (2,  0.600, 2.6),
        (3,  0.300, 3.8),
        (4,  0.180, 5.2),
        (5,  0.100, 7.0),
        (6,  0.060, 9.0),
        (7,  0.035, 11.5),
        (8,  0.020, 14.0),
        (9,  0.012, 17.0),
        (10, 0.007, 20.0),
    ]

    # Inharmonicity constant (piano strings are stiff → partials are sharp)
    INHARMONICITY_B = 0.00018

    def __init__(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(64)
        self._cache: dict = {}
        self._lock = threading.Lock()
        self.volume = 0.85

    # ──────────────────────────────────────────────────────────────────────────
    # Synthesis
    # ──────────────────────────────────────────────────────────────────────────

    def _synthesize(self, freq: float, duration: float = 3.0) -> pygame.mixer.Sound:
        n = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n, endpoint=False)

        wave = np.zeros(n, dtype=np.float64)

        for k, amp, decay in self.HARMONIC_TABLE:
            # Inharmonic partial: f_k = f * k * sqrt(1 + B*k^2)
            f_k = freq * k * math.sqrt(1.0 + self.INHARMONICITY_B * k * k)
            # Per-harmonic exponential decay
            envelope_k = amp * np.exp(-decay * t)
            # Add partial with slight random phase for a natural chorus effect
            phase = np.random.uniform(0.0, 2 * math.pi)
            wave += envelope_k * np.sin(2.0 * math.pi * f_k * t + phase)

        # Attack transient (~5 ms bow-shape)
        attack_n = int(0.005 * SAMPLE_RATE)
        if attack_n > 0:
            ramp = np.linspace(0.0, 1.0, attack_n)
            wave[:attack_n] *= ramp

        # Release fade at very end to avoid click
        fade_n = int(0.02 * SAMPLE_RATE)
        wave[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)

        # Soft saturation (warms the tone)
        wave = np.tanh(wave * 1.4) / np.tanh(1.4)

        # Normalise to ±0.8
        peak = np.max(np.abs(wave))
        if peak > 1e-9:
            wave /= peak
        wave *= 0.80

        # Stereo: frequency-based panning (low → left, high → right)
        pan_ratio = 0.5 + math.log2(max(freq, 100) / 261.63) / 16.0
        pan_ratio = max(0.15, min(0.85, pan_ratio))

        left_gain  = math.sqrt(1.0 - pan_ratio) * 1.41
        right_gain = math.sqrt(pan_ratio)       * 1.41

        left  = np.clip(wave * left_gain  * 32767, -32767, 32767).astype(np.int16)
        right = np.clip(wave * right_gain * 32767, -32767, 32767).astype(np.int16)

        stereo = np.column_stack([left, right])
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def preload(self, frequencies: List[float],
                on_progress: Callable[[int, int], None] | None = None) -> threading.Thread:
        """Pre-generate all piano sounds in a background thread."""
        def _worker():
            for i, freq in enumerate(frequencies):
                key = round(freq, 3)
                sound = self._synthesize(freq)
                with self._lock:
                    self._cache[key] = sound
                if on_progress:
                    on_progress(i + 1, len(frequencies))

        thread = threading.Thread(target=_worker, daemon=True, name="AudioPreload")
        thread.start()
        return thread

    def play(self, freq: float, velocity: float = 1.0) -> None:
        """Play a note immediately. velocity in [0, 1]."""
        key = round(freq, 3)
        with self._lock:
            sound = self._cache.get(key)

        if sound is None:
            # Synthesise on-the-fly if not yet cached
            sound = self._synthesize(freq)
            with self._lock:
                self._cache[key] = sound

        vol = max(0.0, min(1.0, self.volume * velocity))
        sound.set_volume(vol)
        sound.play()

    def set_volume(self, vol: float) -> None:
        self.volume = max(0.0, min(1.0, vol))
