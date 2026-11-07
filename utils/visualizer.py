"""Visualization utilities for bounding boxes, track IDs, and HUD telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional
import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionBox:
    """Container for a single tracked detection."""

    xyxy: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    is_accident: bool = True


class AccidentVisualizer:
    """Renders academic-standard bounding boxes, track IDs, and real-time FPS telemetry.

    Color standard (BGR):
        - Injured Person (`person_accident`, class 0): Red (0, 0, 255)
        - Accident Motorcycle (`motorcycle_accident`, class 1): Yellow (0, 255, 255)
        - Normal tracked traffic (pre-accident monitoring): Green/Teal (120, 220, 80)
    """

    COLOR_PERSON_ACCIDENT: Tuple[int, int, int] = (0, 0, 255)       # Red in BGR
    COLOR_MOTORCYCLE_ACCIDENT: Tuple[int, int, int] = (0, 255, 255) # Yellow in BGR
    COLOR_NORMAL_TRAFFIC: Tuple[int, int, int] = (120, 220, 80)     # Soft Green in BGR
    COLOR_DEFAULT: Tuple[int, int, int] = (255, 191, 0)

    def __init__(self, class_map: Optional[Dict[int, str]] = None) -> None:
        """Initialize the visualizer with an optional class ID to name mapping."""
        self.class_map: Dict[int, str] = class_map or {
            0: "person_accident",
            1: "motorcycle_accident",
        }

    def get_class_color(
        self,
        class_id: int,
        class_name: str = "",
        is_accident: bool = True,
    ) -> Tuple[int, int, int]:
        """Return the academic standard BGR color for a given class."""
        if not is_accident:
            return self.COLOR_NORMAL_TRAFFIC
        normalized_name = class_name.lower()
        if class_id == 0 or "person" in normalized_name:
            return self.COLOR_PERSON_ACCIDENT
        if class_id == 1 or "motorcycle" in normalized_name:
            return self.COLOR_MOTORCYCLE_ACCIDENT
        return self.COLOR_DEFAULT

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[DetectionBox],
        inplace: bool = False,
    ) -> np.ndarray:
        """Draw bounding boxes, class labels, confidence scores, and ByteTrack IDs.

        Args:
            frame_bgr: Input BGR image.
            detections: Sequence of `DetectionBox` instances.
            inplace: If True, draw directly on `frame_bgr`; otherwise copy first.

        Returns:
            Annotated BGR image.
        """
        canvas = frame_bgr if inplace else frame_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 2

        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            color = self.get_class_color(det.class_id, det.class_name, det.is_accident)
            box_thickness = 3 if det.is_accident else 1

            # Draw primary bounding box
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                thickness=box_thickness,
                lineType=cv2.LINE_AA,
            )

            # Build label with Track ID and confidence
            track_prefix = f"ID:{det.track_id} " if det.track_id is not None else ""
            label = f"{track_prefix}{det.class_name} {det.confidence:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(0, y1 - text_h - baseline - 6)
            label_y2 = label_y1 + text_h + baseline + 6
            label_x2 = min(canvas.shape[1] - 1, x1 + text_w + 8)

            # Filled background pill for high contrast readability
            cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)

            # Use dark text on yellow/green backgrounds and white text on red backgrounds
            text_color = (
                (255, 255, 255)
                if color == self.COLOR_PERSON_ACCIDENT
                else (0, 0, 0)
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 4, label_y2 - baseline - 2),
                font,
                font_scale,
                text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def draw_hud(
        self,
        frame_bgr: np.ndarray,
        fps: float,
        mean_l: float,
        low_light_enhanced: bool,
        confirmed_accident: bool = False,
        confirmation_streak: int = 0,
        confirmation_target: int = 5,
        inplace: bool = True,
    ) -> np.ndarray:
        """Render top-left telemetry overlay (FPS, illumination, enhancement state, alert state)."""
        canvas = frame_bgr if inplace else frame_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent status panel
        overlay = canvas.copy()
        panel_w, panel_h = 350, 118
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        ll_status = "ON (LAB+Gamma+CLAHE)" if low_light_enhanced else "OFF (Normal Light)"
        ll_color = (0, 255, 255) if low_light_enhanced else (180, 255, 180)

        cv2.putText(
            canvas,
            f"FPS: {fps:5.1f}",
            (24, 38),
            font,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Mean L-Illum: {mean_l:5.1f} | LL: {ll_status}",
            (24, 65),
            font,
            0.48,
            ll_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Temporal Confirmation: {confirmation_streak}/{confirmation_target} frames",
            (24, 90),
            font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if confirmed_accident:
            status_text = "STATUS: ACCIDENT CONFIRMED"
            status_color = (0, 0, 255)
            # Draw prominent red border around the frame
            h, w = canvas.shape[:2]
            cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (0, 0, 255), 4)
        else:
            status_text = "STATUS: MONITORING"
            status_color = (200, 200, 200)

        cv2.putText(
            canvas,
            status_text,
            (24, 116),
            font,
            0.55,
            status_color,
            2,
            cv2.LINE_AA,
        )

        return canvas
