"""Main real-time inference pipeline for Motorcycle Accident Detection and Notification.

Combines:
1. Adaptive Low-Light Enhancement (CIE LAB + Adaptive Gamma + CLAHE)
2. Ultralytics YOLO11 with ByteTrack multi-object tracking (`.track(persist=True, tracker="bytetrack.yaml")`)
3. Dual-mode Accident Recognition:
   - Direct Custom-Trained Model Mode (`0: person_accident`, `1: motorcycle_accident`)
   - COCO Pretrained Kinematic Collision & Fall Detection Mode (for `yolo11x.pt`)
4. Multi-frame temporal confirmation & cooldown state machine
5. Asynchronous Telegram Bot notification & academic CSV logging
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]

from notification.telegram_bot import TelegramNotifier
from preprocessing.low_light import AdaptiveLowLightEnhancer
from utils.logger import ResearchCSVLogger
from utils.visualizer import AccidentVisualizer, DetectionBox

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AccidentInference")


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TemporalAccidentConfirmator:
    """Multi-frame temporal confirmation and alert cooldown state machine.

    Reduces false-positive single-frame spikes by requiring accident detections
    to persist across `confirmation_frames` before declaring a confirmed event.
    """

    def __init__(
        self,
        confirmation_frames: int = 5,
        max_missing_frames: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.cooldown_seconds = float(cooldown_seconds)

        self.active_streak: int = 0
        self.missing_streak: int = 0
        self.track_hit_counts: Dict[int, int] = {}
        self.last_alert_timestamp: float = -float("inf")

    def update(
        self,
        accident_detections: List[DetectionBox],
        current_time: Optional[float] = None,
    ) -> Tuple[bool, bool, int]:
        """Update temporal state with current frame's accident detections."""
        if current_time is None:
            current_time = time.monotonic()

        if accident_detections:
            self.active_streak += 1
            self.missing_streak = 0
            for det in accident_detections:
                if det.track_id is not None:
                    self.track_hit_counts[det.track_id] = (
                        self.track_hit_counts.get(det.track_id, 0) + 1
                    )
        else:
            self.missing_streak += 1
            if self.missing_streak > self.max_missing_frames:
                self.active_streak = 0
                self.track_hit_counts.clear()

        max_track_streak = max(
            (
                self.track_hit_counts.get(det.track_id, 0)
                for det in accident_detections
                if det.track_id is not None
            ),
            default=0,
        )
        effective_streak = max(self.active_streak, max_track_streak)
        is_confirmed = bool(accident_detections) and (
            effective_streak >= self.confirmation_frames
        )

        should_trigger_alert = False
        if is_confirmed and (
            current_time - self.last_alert_timestamp >= self.cooldown_seconds
        ):
            should_trigger_alert = True
            self.last_alert_timestamp = current_time

        return is_confirmed, should_trigger_alert, effective_streak


class AccidentDetectionPipeline:
    """End-to-end real-time motorcycle accident detection and notification system."""

    COCO_VEHICLE_CLASSES: Set[int] = {2, 5, 7}  # car, bus, truck
    COCO_MOTORCYCLE_CLASS: int = 3
    COCO_PERSON_CLASS: int = 0

    def __init__(
        self,
        config: Dict[str, Any],
        enable_low_light_override: Optional[bool] = None,
        enable_tracker_override: Optional[bool] = None,
        enable_confirmation_override: Optional[bool] = None,
    ) -> None:
        self.config = config

        # 1. Model & Class Configuration
        model_cfg = config.get("model", {})
        self.weights_path: str = model_cfg.get("weights", "yolo11x.pt")
        self.conf_threshold: float = float(model_cfg.get("conf_threshold", 0.35))
        self.iou_threshold: float = float(model_cfg.get("iou_threshold", 0.45))
        self.tracker_config: str = model_cfg.get("tracker", "bytetrack.yaml")
        self.device: str = str(model_cfg.get("device", "cpu"))
        self.use_tracker: bool = (
            True if enable_tracker_override is None else bool(enable_tracker_override)
        )

        raw_classes = config.get(
            "classes", {0: "person_accident", 1: "motorcycle_accident"}
        )
        self.class_map: Dict[int, str] = {int(k): str(v) for k, v in raw_classes.items()}

        # 2. Preprocessing (Adaptive Low-Light Enhancer)
        prep_cfg = config.get("preprocessing", {})
        self.enable_low_light: bool = (
            bool(prep_cfg.get("enable_low_light_enhancement", True))
            if enable_low_light_override is None
            else bool(enable_low_light_override)
        )
        tile_grid = prep_cfg.get("clahe_tile_grid_size", [8, 8])
        self.enhancer = AdaptiveLowLightEnhancer(
            low_light_threshold=float(prep_cfg.get("low_light_threshold", 90.0)),
            clahe_clip_limit=float(prep_cfg.get("clahe_clip_limit", 2.5)),
            clahe_tile_grid_size=(int(tile_grid[0]), int(tile_grid[1])),
            target_illumination=float(prep_cfg.get("target_illumination", 128.0)),
        )

        # 3. Multi-frame Temporal Confirmation
        temp_cfg = config.get("temporal_confirmation", {})
        conf_frames = int(temp_cfg.get("confirmation_frames", 5))
        if enable_confirmation_override is False:
            conf_frames = 1
        self.confirmator = TemporalAccidentConfirmator(
            confirmation_frames=conf_frames,
            max_missing_frames=int(temp_cfg.get("max_missing_frames", 3)),
            cooldown_seconds=float(temp_cfg.get("cooldown_seconds", 60.0)),
        )

        # 4. Asynchronous Telegram Notification
        tg_cfg = config.get("telegram", {})
        self.notifier = TelegramNotifier(
            bot_token=str(tg_cfg.get("bot_token", "")),
            chat_id=str(tg_cfg.get("chat_id", "")),
            enabled=bool(tg_cfg.get("enabled", True)),
            timeout_seconds=float(tg_cfg.get("timeout_seconds", 10.0)),
        )

        # 5. Visualization & Logging
        log_cfg = config.get("logging", {})
        self.save_snapshots: bool = bool(log_cfg.get("save_snapshots", True))
        self.snapshots_dir = Path(log_cfg.get("snapshots_dir", "logs/snapshots"))
        if self.save_snapshots:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.csv_logger = ResearchCSVLogger(
            events_csv_path=log_cfg.get("events_csv", "logs/accident_events.csv"),
            metrics_csv_path=log_cfg.get("metrics_csv", "logs/runtime_metrics.csv"),
        )
        self.visualizer = AccidentVisualizer(class_map=self.class_map)

        logger.info("Loading YOLO model from '%s'...", self.weights_path)
        if YOLO is None:
            raise ImportError(
                "The 'ultralytics' package is required for YOLO inference. "
                "Install it with: pip install -r requirements.txt"
            )
        self.model = YOLO(self.weights_path)

        # Detect whether the weights are custom-trained on accident classes or COCO pretrained
        model_names = {
            int(k): str(v).lower() for k, v in getattr(self.model, "names", {}).items()
        }
        self.is_coco_pretrained = (
            model_names.get(0) == "person" and model_names.get(3) == "motorcycle"
        )
        self._accident_track_memory: Dict[int, int] = {}
        self._last_collision_frame: int = -1000

        if self.is_coco_pretrained:
            logger.info(
                "Detected COCO pretrained model ('%s'). Activating ByteTrack Collision & Fallen-Rider Kinematic Detector.",
                self.weights_path,
            )
        else:
            logger.info(
                "Detected custom-trained accident model ('%s'). Using direct class mapping.",
                self.weights_path,
            )

    @staticmethod
    def _overlap_over_min_area(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
        pad: int = 0,
    ) -> float:
        """Compute intersection area divided by the smaller box's area."""
        ax1, ay1, ax2, ay2 = box_a[0] - pad, box_a[1] - pad, box_a[2] + pad, box_a[3] + pad
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return float(inter) / float(min(area_a, area_b))

    def _parse_yolo_results(self, results: Any, frame_idx: int) -> List[DetectionBox]:
        """Extract `DetectionBox` objects (with `is_accident` flag) from YOLO output."""
        detections: List[DetectionBox] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy_np = boxes.xyxy.cpu().numpy().astype(int)
        conf_np = boxes.conf.cpu().numpy().astype(float)
        cls_np = boxes.cls.cpu().numpy().astype(int)
        track_ids_np: Optional[np.ndarray] = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        )

        if not self.is_coco_pretrained:
            for idx in range(len(xyxy_np)):
                cls_id = int(cls_np[idx])
                if cls_id not in self.class_map:
                    continue
                x1, y1, x2, y2 = xyxy_np[idx]
                track_id = int(track_ids_np[idx]) if track_ids_np is not None else None
                detections.append(
                    DetectionBox(
                        xyxy=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=cls_id,
                        class_name=self.class_map[cls_id],
                        confidence=float(conf_np[idx]),
                        track_id=track_id,
                        is_accident=True,
                    )
                )
            return detections

        # COCO Pretrained Mode: Separate vehicles (car/bus/truck), motorcycles, and persons
        vehicles: List[Tuple[Tuple[int, int, int, int], float, Optional[int], int]] = []
        motorcycles: List[Tuple[Tuple[int, int, int, int], float, Optional[int]]] = []
        persons: List[Tuple[Tuple[int, int, int, int], float, Optional[int]]] = []

        for idx in range(len(xyxy_np)):
            cls_id = int(cls_np[idx])
            conf = float(conf_np[idx])
            x1, y1, x2, y2 = (
                int(xyxy_np[idx][0]),
                int(xyxy_np[idx][1]),
                int(xyxy_np[idx][2]),
                int(xyxy_np[idx][3]),
            )
            tid = int(track_ids_np[idx]) if track_ids_np is not None else None
            box = (x1, y1, x2, y2)

            if cls_id in self.COCO_VEHICLE_CLASSES:
                vehicles.append((box, conf, tid, cls_id))
            elif cls_id == self.COCO_MOTORCYCLE_CLASS:
                motorcycles.append((box, conf, tid))
            elif cls_id == self.COCO_PERSON_CLASS:
                persons.append((box, conf, tid))

        # Check for active collision between a motor vehicle (car/truck/bus) and a fallen/thrown rider
        frame_has_vehicle_collision = False
        for p_box, p_conf, p_tid in persons:
            pw = max(1, p_box[2] - p_box[0])
            ph = max(1, p_box[3] - p_box[1])
            aspect_ratio = pw / float(ph)
            # A rider thrown onto/across a car or lying on the road has a non-upright aspect ratio (>= 0.65)
            collides_with_car = (aspect_ratio >= 0.65) and any(
                self._overlap_over_min_area(p_box, v_box, pad=15) >= 0.08
                for v_box, _, _, _ in vehicles
            )
            is_fallen_posture = aspect_ratio >= 0.85
            if collides_with_car or (
                is_fallen_posture and (frame_idx - self._last_collision_frame <= 400)
            ):
                frame_has_vehicle_collision = True
                if collides_with_car:
                    self._last_collision_frame = frame_idx
                if p_tid is not None:
                    self._accident_track_memory[p_tid] = frame_idx

        # Build DetectionBox list
        accident_person_boxes: List[Tuple[int, int, int, int]] = []
        for p_box, p_conf, p_tid in persons:
            pw = max(1, p_box[2] - p_box[0])
            ph = max(1, p_box[3] - p_box[1])
            aspect_ratio = pw / float(ph)
            in_memory = (
                p_tid is not None
                and (frame_idx - self._accident_track_memory.get(p_tid, -9999)) <= 120
            )
            collides_now = (aspect_ratio >= 0.65) and any(
                self._overlap_over_min_area(p_box, v_box, pad=15) >= 0.08
                for v_box, _, _, _ in vehicles
            )
            fallen_after_crash = (aspect_ratio >= 0.85) and (
                frame_has_vehicle_collision
                or (frame_idx - self._last_collision_frame <= 400)
            )
            is_acc = collides_now or fallen_after_crash or in_memory
            if is_acc:
                accident_person_boxes.append(p_box)
                if p_tid is not None:
                    self._accident_track_memory[p_tid] = frame_idx

            detections.append(
                DetectionBox(
                    xyxy=p_box,
                    class_id=0,
                    class_name="person_accident" if is_acc else "person",
                    confidence=p_conf,
                    track_id=p_tid,
                    is_accident=is_acc,
                )
            )

        for m_box, m_conf, m_tid in motorcycles:
            in_memory = (
                m_tid is not None
                and (frame_idx - self._accident_track_memory.get(m_tid, -9999)) <= 60
            )
            near_accident_person = any(
                self._overlap_over_min_area(m_box, ap_box, pad=45) >= 0.02
                for ap_box in accident_person_boxes
            )
            collides_during_crash = (
                (frame_idx - self._last_collision_frame <= 35)
                and any(
                    self._overlap_over_min_area(m_box, v_box, pad=30) >= 0.02
                    for v_box, _, _, _ in vehicles
                )
            )
            is_acc = near_accident_person or collides_during_crash or in_memory
            if is_acc and m_tid is not None:
                self._accident_track_memory[m_tid] = frame_idx

            detections.append(
                DetectionBox(
                    xyxy=m_box,
                    class_id=1,
                    class_name="motorcycle_accident" if is_acc else "motorcycle",
                    confidence=m_conf,
                    track_id=m_tid,
                    is_accident=is_acc,
                )
            )

        return detections

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        source_label: str = "Stream",
        video_timestamp_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the full pipeline on a single video frame."""
        t_start = time.perf_counter()

        # Step 1: Conditional Low-Light Enhancement
        t_prep_start = time.perf_counter()
        if self.enable_low_light:
            proc_frame, low_light_applied, mean_l = self.enhancer.process(frame_bgr)
        else:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            mean_l = float(np.mean(lab[:, :, 0]))
            proc_frame, low_light_applied = frame_bgr, False
        preprocess_latency_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # Step 2: YOLO Detection + ByteTrack Multi-Object Tracking
        t_inf_start = time.perf_counter()
        if self.is_coco_pretrained:
            target_classes = [0, 2, 3, 5, 7]
            effective_conf = min(self.conf_threshold, 0.30)
        else:
            target_classes = list(self.class_map.keys())
            effective_conf = self.conf_threshold

        if self.use_tracker:
            results = self.model.track(
                source=proc_frame,
                persist=True,
                tracker=self.tracker_config,
                conf=effective_conf,
                iou=self.iou_threshold,
                classes=target_classes,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=proc_frame,
                conf=effective_conf,
                iou=self.iou_threshold,
                classes=target_classes,
                verbose=False,
            )
        inference_latency_ms = (time.perf_counter() - t_inf_start) * 1000.0

        all_detections = self._parse_yolo_results(results, frame_idx=frame_idx)
        accident_detections = [d for d in all_detections if d.is_accident]

        # Count accident class occurrences
        num_persons = sum(1 for d in accident_detections if d.class_id == 0)
        num_motorcycles = sum(1 for d in accident_detections if d.class_id == 1)

        # Step 3: Multi-Frame Confirmation State Machine
        is_confirmed, should_alert, streak = self.confirmator.update(
            accident_detections,
            current_time=video_timestamp_sec,
        )

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(total_latency_ms, 1e-3)

        # Step 4: Annotate Frame (Red for person_accident, Yellow for motorcycle_accident)
        annotated = self.visualizer.draw_detections(
            proc_frame, all_detections, inplace=False
        )
        annotated = self.visualizer.draw_hud(
            annotated,
            fps=fps,
            mean_l=mean_l,
            low_light_enhanced=low_light_applied,
            confirmed_accident=is_confirmed,
            confirmation_streak=streak,
            confirmation_target=self.confirmator.confirmation_frames,
            inplace=True,
        )

        # Step 5: Log Metrics & Dispatch Asynchronous Alert if Triggered
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if video_timestamp_sec is not None:
            timestamp_str = f"{timestamp_str} (Video t={video_timestamp_sec:.1f}s)"

        self.csv_logger.log_frame_metrics(
            frame_idx=frame_idx,
            mean_l_illumination=mean_l,
            low_light_enhanced=low_light_applied,
            preprocess_latency_ms=preprocess_latency_ms,
            inference_track_latency_ms=inference_latency_ms,
            total_latency_ms=total_latency_ms,
            fps=fps,
            num_detections=len(accident_detections),
            num_motorcycle_accident=num_motorcycles,
            num_person_accident=num_persons,
            confirmed_accident=is_confirmed,
        )

        if should_alert:
            conf_values = [d.confidence for d in accident_detections]
            max_conf = max(conf_values) if conf_values else 0.0
            mean_conf = float(np.mean(conf_values)) if conf_values else 0.0
            track_ids = [
                d.track_id for d in accident_detections if d.track_id is not None
            ]

            snapshot_path_str = ""
            if self.save_snapshots:
                safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_file = (
                    self.snapshots_dir / f"accident_frame_{frame_idx}_{safe_ts}.jpg"
                )
                cv2.imwrite(str(snap_file), annotated)
                snapshot_path_str = str(snap_file)

            self.notifier.send_accident_alert(
                frame_bgr=annotated,
                num_motorcycles=num_motorcycles,
                num_persons=num_persons,
                max_confidence=max_conf,
                timestamp=timestamp_str,
                low_light_active=low_light_applied,
                location_tag=source_label,
            )

            self.csv_logger.log_accident_event(
                frame_idx=frame_idx,
                num_motorcycle_accident=num_motorcycles,
                num_person_accident=num_persons,
                max_confidence=max_conf,
                mean_confidence=mean_conf,
                track_ids=track_ids,
                mean_l_illumination=mean_l,
                low_light_enhanced=low_light_applied,
                fps=fps,
                total_latency_ms=total_latency_ms,
                telegram_dispatched=self.notifier.is_configured,
                snapshot_path=snapshot_path_str,
                timestamp=timestamp_str,
            )
            logger.warning(
                "🚨 CONFIRMED ACCIDENT at frame %d (t=%.2fs) | Motorcycles=%d, Persons=%d, MaxConf=%.3f | Snapshot=%s",
                frame_idx,
                video_timestamp_sec or 0.0,
                num_motorcycles,
                num_persons,
                max_conf,
                snapshot_path_str,
            )

        telemetry = {
            "frame_idx": frame_idx,
            "fps": fps,
            "preprocess_latency_ms": preprocess_latency_ms,
            "inference_latency_ms": inference_latency_ms,
            "total_latency_ms": total_latency_ms,
            "mean_l": mean_l,
            "low_light_applied": low_light_applied,
            "detections": accident_detections,
            "num_persons": num_persons,
            "num_motorcycles": num_motorcycles,
            "is_confirmed": is_confirmed,
            "should_alert": should_alert,
        }
        return annotated, telemetry

    def run_stream(
        self,
        source: str | int = 0,
        output_video_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> Dict[str, float]:
        """Run real-time inference on a video file, RTSP stream, or webcam index."""
        cap_source: str | int = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer: Optional[cv2.VideoWriter] = None
        if output_video_path:
            out_path = Path(output_video_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))

        frame_idx = start_frame
        logger.info("Starting inference loop on source='%s'...", source)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                video_ts = frame_idx / max(src_fps, 1.0)
                annotated, _ = self.process_frame(
                    frame_bgr=frame,
                    frame_idx=frame_idx,
                    source_label=str(source),
                    video_timestamp_sec=video_ts,
                )

                if writer is not None:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Motorcycle Accident Detection & ByteTrack", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User requested exit via 'q'.")
                        break

                if max_frames is not None and (frame_idx - start_frame) >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.notifier.flush(timeout=30.0)

        summary = self.csv_logger.get_summary_statistics()
        logger.info(
            "Inference finished (%d frames). Mean FPS: %.2f | Mean Latency: %.2f ms",
            frame_idx - start_frame,
            summary["mean_fps"],
            summary["mean_latency_ms"],
        )
        return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_video = "ภาพคลิปอุบัติเหตุ จากกล้องวงจรปิด สี่แยก บ่อขิง ม.7 ต.หน้าพระธาตุ อ.พนัสนิคม ชลบุรี.mp4"
    default_source = default_video if Path(default_video).exists() else "0"

    parser = argparse.ArgumentParser(
        description="Real-Time Motorcycle Accident Detection with Low-Light Enhancement & ByteTrack"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=default_source,
        help="Video file path, RTSP URL, or webcam index.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/output_accident_detection.mp4",
        help="Optional path to save annotated output MP4 video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without OpenCV GUI window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Optional starting frame index.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    config = load_config(args.config)
    pipeline = AccidentDetectionPipeline(config=config)
    pipeline.run_stream(
        source=args.source,
        output_video_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    main()
