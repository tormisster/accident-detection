"""Adaptive low-light image enhancement module in CIE LAB color space."""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


class AdaptiveLowLightEnhancer:
    """Enhances low-light frames using adaptive gamma correction and CLAHE on the L-channel.

    Converting BGR to CIE LAB isolates luminance (L) from chrominance (a, b),
    preventing color distortion while restoring shadow details in nighttime or
    poorly lit traffic surveillance footage.
    """

    def __init__(
        self,
        low_light_threshold: float = 90.0,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        target_illumination: float = 128.0,
    ) -> None:
        """Initialize the AdaptiveLowLightEnhancer.

        Args:
            low_light_threshold: Mean L-channel value below which enhancement triggers (0-255).
            clahe_clip_limit: Contrast limiting threshold for CLAHE.
            clahe_tile_grid_size: Grid size (width, height) for CLAHE local histogram equalization.
            target_illumination: Reference L-channel brightness used to compute adaptive gamma.
        """
        self.low_light_threshold = float(low_light_threshold)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = tuple(clahe_tile_grid_size)
        self.target_illumination = float(target_illumination)

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Construct a 256-element uint8 Look-Up Table for fast gamma correction."""
        gamma = max(0.1, min(gamma, 5.0))
        indices = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip(np.power(indices, gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
        return lut

    def compute_adaptive_gamma(self, mean_l: float) -> float:
        """Compute adaptive gamma (< 1.0 brightens dark regions) from mean illumination."""
        eps = 1e-5
        norm_current = np.clip(mean_l / 255.0, eps, 1.0 - eps)
        norm_target = np.clip(self.target_illumination / 255.0, eps, 1.0 - eps)
        gamma = float(np.log(norm_target) / np.log(norm_current))
        return float(np.clip(gamma, 0.35, 1.0))

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Conditionally enhance a BGR frame if its mean L-channel illumination is below threshold.

        Args:
            frame_bgr: Input image in BGR format (H, W, 3), dtype uint8.

        Returns:
            Tuple containing:
                - Enhanced (or original) BGR frame.
                - Boolean flag indicating whether low-light enhancement was applied.
                - Computed mean illumination on the L-channel (0-255 scale).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame_bgr is empty or None.")

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        mean_l = float(np.mean(l_channel))
        if mean_l >= self.low_light_threshold:
            return frame_bgr, False, mean_l

        # Step 1: Adaptive gamma correction on the L-channel
        gamma = self.compute_adaptive_gamma(mean_l)
        lut = self._build_gamma_lut(gamma)
        l_gamma = cv2.LUT(l_channel, lut)

        # Step 2: CLAHE on the gamma-corrected L-channel
        l_enhanced = self._clahe.apply(l_gamma)

        # Step 3: Merge enhanced L with original chrominance channels and convert back to BGR
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced_bgr, True, mean_l
