"""CSV telemetry and event logger for academic evaluation and latency benchmarking."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np


class ResearchCSVLogger:
    """Logs per-frame runtime performance metrics and confirmed accident events to CSV files."""

    EVENT_COLUMNS: List[str] = [
        "timestamp",
        "frame_idx",
        "num_motorcycle_accident",
        "num_person_accident",
        "max_confidence",
        "mean_confidence",
        "track_ids",
        "mean_l_illumination",
        "low_light_enhanced",
        "fps",
        "total_latency_ms",
        "telegram_dispatched",
        "snapshot_path",
    ]

    METRIC_COLUMNS: List[str] = [
        "timestamp",
        "frame_idx",
        "mean_l_illumination",
        "low_light_enhanced",
        "preprocess_latency_ms",
        "inference_track_latency_ms",
        "total_latency_ms",
        "fps",
        "num_detections",
        "num_motorcycle_accident",
        "num_person_accident",
        "confirmed_accident",
    ]

    def __init__(
        self,
        events_csv_path: str | Path = "logs/accident_events.csv",
        metrics_csv_path: str | Path = "logs/runtime_metrics.csv",
    ) -> None:
        """Initialize the CSV logger and ensure output files have headers."""
        self.events_csv_path = Path(events_csv_path)
        self.metrics_csv_path = Path(metrics_csv_path)

        self.events_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_csv(self.events_csv_path, self.EVENT_COLUMNS)
        self._init_csv(self.metrics_csv_path, self.METRIC_COLUMNS)

        self._fps_history: List[float] = []
        self._latency_history_ms: List[float] = []

    @staticmethod
    def _init_csv(file_path: Path, columns: Sequence[str]) -> None:
        """Create CSV file with header row if it does not already exist."""
        if not file_path.exists() or file_path.stat().st_size == 0:
            with file_path.open(mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    def log_frame_metrics(
        self,
        frame_idx: int,
        mean_l_illumination: float,
        low_light_enhanced: bool,
        preprocess_latency_ms: float,
        inference_track_latency_ms: float,
        total_latency_ms: float,
        fps: float,
        num_detections: int,
        num_motorcycle_accident: int,
        num_person_accident: int,
        confirmed_accident: bool,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a single frame's performance and detection metrics to the metrics CSV."""
        if timestamp is None:
            timestamp = datetime.now().isoformat(timespec="milliseconds")

        self._fps_history.append(fps)
        self._latency_history_ms.append(total_latency_ms)

        row = [
            timestamp,
            frame_idx,
            f"{mean_l_illumination:.2f}",
            int(low_light_enhanced),
            f"{preprocess_latency_ms:.3f}",
            f"{inference_track_latency_ms:.3f}",
            f"{total_latency_ms:.3f}",
            f"{fps:.2f}",
            num_detections,
            num_motorcycle_accident,
            num_person_accident,
            int(confirmed_accident),
        ]
        with self.metrics_csv_path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_accident_event(
        self,
        frame_idx: int,
        num_motorcycle_accident: int,
        num_person_accident: int,
        max_confidence: float,
        mean_confidence: float,
        track_ids: Sequence[int],
        mean_l_illumination: float,
        low_light_enhanced: bool,
        fps: float,
        total_latency_ms: float,
        telegram_dispatched: bool,
        snapshot_path: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a confirmed accident event to the accident events CSV."""
        if timestamp is None:
            timestamp = datetime.now().isoformat(timespec="milliseconds")

        track_ids_str = ";".join(str(tid) for tid in track_ids) if track_ids else "none"
        row = [
            timestamp,
            frame_idx,
            num_motorcycle_accident,
            num_person_accident,
            f"{max_confidence:.4f}",
            f"{mean_confidence:.4f}",
            track_ids_str,
            f"{mean_l_illumination:.2f}",
            int(low_light_enhanced),
            f"{fps:.2f}",
            f"{total_latency_ms:.3f}",
            int(telegram_dispatched),
            snapshot_path,
        ]
        with self.events_csv_path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_summary_statistics(self) -> Dict[str, float]:
        """Compute aggregate runtime statistics for paper reporting."""
        if not self._fps_history:
            return {
                "mean_fps": 0.0,
                "std_fps": 0.0,
                "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }
        fps_arr = np.asarray(self._fps_history, dtype=np.float64)
        lat_arr = np.asarray(self._latency_history_ms, dtype=np.float64)
        return {
            "mean_fps": float(np.mean(fps_arr)),
            "std_fps": float(np.std(fps_arr)),
            "mean_latency_ms": float(np.mean(lat_arr)),
            "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        }
