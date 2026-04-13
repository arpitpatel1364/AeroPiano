"""
tracker.py — MediaPipe Hand Tracking Wrapper
Provides clean landmark access and optional skeleton drawing.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# MediaPipe fingertip landmark indices
FINGERTIP_IDS  = [4, 8, 12, 16, 20]    # Thumb … Pinky tips
KNUCKLE_IDS    = [3, 7, 11, 15, 19]    # Proximal joints (DIP)
MCP_IDS        = [2, 6, 10, 14, 18]    # Metacarpo-phalangeal joints
WRIST_ID       = 0
ALL_TIP_IDS    = FINGERTIP_IDS


@dataclass
class HandData:
    """Processed data for a single detected hand."""
    landmarks_px: List[Tuple[int, int]]   # (x, y) in pixel coords
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) normalised [0–1]
    handedness: str                        # 'Left' or 'Right'

    def tip(self, finger_id: int) -> Tuple[int, int]:
        """Return pixel (x, y) of fingertip. finger_id: 4=thumb,8=index,…"""
        return self.landmarks_px[finger_id]

    def pinch_distance(self) -> float:
        """Pixel distance between thumb tip (4) and index tip (8)."""
        t = np.array(self.landmarks_px[4])
        i = np.array(self.landmarks_px[8])
        return float(np.linalg.norm(t - i))

    def is_pinching(self, threshold: float = 45.0) -> bool:
        return self.pinch_distance() < threshold


class HandTracker:
    """Wraps MediaPipe Hands for easy per-frame use."""

    def __init__(self, max_hands: int = 2, detection_conf: float = 0.7,
                 tracking_conf: float = 0.55):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Run detection on a BGR frame.
        Returns a list of HandData (one per detected hand).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return []

        hands: List[HandData] = []
        handedness_list = results.multi_handedness or []

        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            px_pts = []
            norm_pts = []
            for lm in hand_lm.landmark:
                px_pts.append((int(lm.x * w), int(lm.y * h)))
                norm_pts.append((lm.x, lm.y, lm.z))

            side = "Right"
            if i < len(handedness_list):
                side = handedness_list[i].classification[0].label

            hands.append(HandData(
                landmarks_px=px_pts,
                landmarks_norm=norm_pts,
                handedness=side,
            ))

        return hands

    # ──────────────────────────────────────────────────────────────────────────

    def draw_skeleton(self, frame_bgr: np.ndarray,
                      results_mp,
                      color: Tuple[int, int, int] = (0, 255, 160),
                      thickness: int = 2) -> None:
        """Draw skeleton directly onto a BGR frame (for the OpenCV overlay)."""
        if results_mp.multi_hand_landmarks:
            for hand_lm in results_mp.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(color=color, thickness=thickness, circle_radius=4),
                    self._mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                )

    def process_raw(self, frame_bgr: np.ndarray):
        """Return raw MediaPipe results (for skeleton drawing)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        rgb.flags.writeable = True
        return res

    def close(self) -> None:
        self._hands.close()
