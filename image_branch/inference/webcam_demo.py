"""
Live Webcam Deepfake Detection & Explainability Demo.

Features:
1. Real-time video display loop with semi-transparent Grad-CAM heatmap blending.
2. Frame skipping (inference run once every 10 frames) for smooth 30+ FPS playback.
3. Low confidence / bad frame handling (<40% confidence shows 'LOW CONFIDENCE - CANNOT ASSESS').
4. Key Handlers:
   - 'q': Quit cleanly
   - 's': Save 3-panel static report (Original + Grad-CAM + ELA) to reports/dataset_audit/webcam_captures/
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.inference.gradcam import load_trained_model, generate_gradcam
from image_branch.inference.ela import generate_ela
from image_branch.inference.report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="Live Webcam Deepfake Detection Demo")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--frame-stride", type=int, default=10, help="Inference frame stride (default: 10)")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for ELA (default: 90)")
    parser.add_argument("--confidence-threshold", type=float, default=0.40, help="Confidence threshold for verdict (default: 0.40)")
    parser.add_argument("--test-seconds", type=int, default=0, help="If > 0, run automated performance test for specified seconds and exit")
    return parser.parse_args()


def draw_countdown_overlay(frame_bgr: np.ndarray, seconds_left: int) -> np.ndarray:
    """
    Renders a prominent 3-2-1 countdown overlay in the center of the frame.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # Semi-transparent backdrop box in center
    box_w, box_h = 360, 240
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)

    # Accent border around countdown card
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 3, cv2.LINE_AA)

    # Header text
    cv2.putText(
        annotated,
        "GET READY TO POSE!",
        (cx - 125, y1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Countdown big number
    count_str = str(seconds_left)
    font_scale = 4.0
    thickness = 8
    (tw, th), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = cx - tw // 2
    ty = cy + th // 2 + 15

    # Outline (black stroke)
    cv2.putText(annotated, count_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 5, cv2.LINE_AA)
    # Foreground text (bright gold/yellow)
    cv2.putText(annotated, count_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255), thickness, cv2.LINE_AA)

    return annotated


def draw_hud_overlay(
    frame_bgr: np.ndarray,
    last_res: Optional[Dict[str, Any]],
    fps: float,
    inf_time_ms: float,
    conf_threshold: float = 0.40,
    save_msg_counter: int = 0
) -> np.ndarray:
    """
    Renders top banner, verdict badges, class prediction, and FPS metrics onto the live frame.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # 1. Top HUD semi-transparent background banner
    banner_height = 85
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # 2. Bottom status bar for FPS and latency
    status_bar_h = 30
    overlay_bot = annotated.copy()
    cv2.rectangle(overlay_bot, (0, h - status_bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay_bot, 0.85, annotated, 0.15, 0, annotated)

    # Status Bar Text
    status_text = f"Display FPS: {fps:.1f} | Inference Latency: {inf_time_ms:.1f} ms | Controls: ['q'] Quit  ['s'] Save Report"
    cv2.putText(annotated, status_text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if last_res is None:
        cv2.putText(annotated, "INITIALIZING DEEPFAKE DETECTOR...", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
        return annotated

    conf = last_res.get("confidence", 0.0)
    pred_class = last_res.get("prediction", "Unknown")
    binary_verdict = last_res.get("binary_verdict", "UNKNOWN")

    # Task 2: Handle Low Confidence / Missing Face gracefully
    if conf < conf_threshold:
        verdict_text = "LOW CONFIDENCE — CANNOT ASSESS"
        verdict_color = (0, 215, 255)  # Amber / Yellow
        sub_text = f"Top Candidate: {pred_class} ({conf*100:.1f}% < {conf_threshold*100:.0f}% threshold)"
    else:
        if binary_verdict == "MANIPULATED":
            verdict_text = "VERDICT: MANIPULATED (FAKE)"
            verdict_color = (40, 40, 240)  # Red
        else:
            verdict_text = "VERDICT: AUTHENTIC (REAL)"
            verdict_color = (50, 220, 50)  # Green

        sub_text = f"Class: {pred_class} | Confidence: {conf*100:.1f}%"

    # Draw Verdict Title
    cv2.putText(annotated, verdict_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2, cv2.LINE_AA)
    # Draw Subtitle
    cv2.putText(annotated, sub_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # Save Confirmation Toast Message
    if save_msg_counter > 0:
        toast_w, toast_h = 320, 45
        tx1, ty1 = (w - toast_w) // 2, h - 90
        cv2.rectangle(annotated, (tx1, ty1), (tx1 + toast_w, ty1 + toast_h), (0, 180, 0), -1)
        cv2.putText(annotated, "HIGH-RES REPORT SAVED!", (tx1 + 20, ty1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def run_webcam_loop(args):
    """
    Main webcam capture and display loop.
    """
    captures_dir = PROJECT_ROOT / "reports" / "dataset_audit" / "webcam_captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  ADL PROJECT — STEP 7: LIVE WEBCAM DEEPFAKE DETECTION DEMO")
    print("=" * 80)
    print(f"[Setup] Opening VideoCapture device ID: {args.camera_id}")
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"[Error] Could not open camera device {args.camera_id}. Attempting fallback...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No available camera device found!")

    # Explicitly set 1280x720 capture resolution
    requested_w, requested_h = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_h)

    granted_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    granted_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Setup] Requested Resolution: {requested_w}x{requested_h} | Granted Resolution: {granted_w}x{granted_h}")

    window_name = "ADL Project - Live Deepfake Detection & Explainability Demo"
    if args.test_seconds == 0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if granted_w > 0 and granted_h > 0:
            cv2.resizeWindow(window_name, granted_w, granted_h)

    # Pre-load model onto GPU/CPU
    model, device = load_trained_model()
    print(f"[Setup] Detector initialized on device: {device}")
    print(f"[Setup] Configuration: Frame Stride = {args.frame_stride} | Confidence Threshold = {args.confidence_threshold * 100:.0f}%")
    print("[Controls] Press 'q' to QUIT | Press 's' to START 3-sec countdown & SAVE report")
    print("-" * 80)

    frame_count = 0
    last_res: Optional[Dict[str, Any]] = None
    last_inference_ms = 0.0
    save_msg_counter = 0

    countdown_active = False
    countdown_start_time = 0.0

    fps_eval_count = 0
    t_start = time.time()
    display_fps = 30.0

    inference_times_ms = []
    first_frame_logged = False

    try:
        while True:
            t_frame_start = time.time()
            ret, frame_bgr = cap.read()

            if not ret or frame_bgr is None:
                print("[Warning] Blank or unreadable frame received. Skipping...")
                continue

            h, w = frame_bgr.shape[:2]
            if not first_frame_logged:
                print(f"[Setup] Confirmed Live Frame Matrix Dimensions: {w}x{h}")
                first_frame_logged = True

            frame_count += 1
            fps_eval_count += 1

            # Convert BGR frame to RGB numpy array for model input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Perform model inference once every N frames
            if frame_count % args.frame_stride == 1 or last_res is None:
                t_inf_0 = time.time()
                try:
                    last_res = generate_gradcam(frame_rgb, model=model, device=device)
                    last_inference_ms = (time.time() - t_inf_0) * 1000.0
                    inference_times_ms.append(last_inference_ms)
                except Exception as exc:
                    print(f"[Warning] Inference failed on frame {frame_count}: {exc}")

            # Prepare Display Frame with Grad-CAM Semi-Transparent Overlay
            if last_res is not None and last_res.get("confidence", 0.0) >= args.confidence_threshold:
                overlay_rgb = last_res["overlay_image"]
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                overlay_resized = cv2.resize(overlay_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
                blended_frame = cv2.addWeighted(frame_bgr, 0.65, overlay_resized, 0.35, 0)
            else:
                blended_frame = frame_bgr.copy()

            # Render HUD Overlay (Verdict, Class, Confidence, FPS)
            hud_frame = draw_hud_overlay(
                blended_frame,
                last_res,
                fps=display_fps,
                inf_time_ms=last_inference_ms,
                conf_threshold=args.confidence_threshold,
                save_msg_counter=save_msg_counter
            )

            if save_msg_counter > 0:
                save_msg_counter -= 1

            # Handle 3-second Countdown Overlay & Capture
            if countdown_active:
                elapsed = time.time() - countdown_start_time
                remaining = 3 - int(elapsed)
                if remaining > 0:
                    hud_frame = draw_countdown_overlay(hud_frame, seconds_left=remaining)
                else:
                    countdown_active = False
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_png_path = captures_dir / f"webcam_report_{timestamp}.png"
                    print(f"\n[Countdown Complete] Capturing frame at resolution {w}x{h} & saving report: {report_png_path}")
                    generate_report(frame_rgb, output_path=report_png_path, save_fig=True)
                    save_msg_counter = 45  # Show toast message for ~1.5 sec

            # Calculate FPS every 15 frames
            if fps_eval_count >= 15:
                t_now = time.time()
                display_fps = fps_eval_count / max(t_now - t_start, 1e-4)
                fps_eval_count = 0
                t_start = t_now

            # Show live window if not running automated test
            if args.test_seconds == 0:
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF

                # Handler for 'q': Quit cleanly
                if key == ord('q'):
                    print("\n['q' Pressed] Exiting live webcam loop cleanly...")
                    break

                # Handler for 's': Trigger 3-second countdown
                elif key == ord('s'):
                    if not countdown_active:
                        print("\n['s' Pressed] Starting 3-second capture countdown...")
                        countdown_active = True
                        countdown_start_time = time.time()

            # Automated test mode condition
            if args.test_seconds > 0:
                if frame_count == 10 and not countdown_active:
                    print("\n[Automated Test] Triggering 3-second capture countdown...")
                    countdown_active = True
                    countdown_start_time = time.time()

                if frame_count >= (args.test_seconds * 30):
                    print(f"\n[Test Complete] Reached target test duration of {args.test_seconds} seconds.")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    avg_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    print("\n" + "=" * 80)
    print("  WEBCAM DEMO PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  Total Frames Processed:    {frame_count:,}")
    print(f"  Display Frame Rate:        {display_fps:.1f} FPS")
    print(f"  Total Inferences Run:      {len(inference_times_ms):,}")
    print(f"  Avg Inference Latency:     {avg_inf_ms:.2f} ms per forward pass")
    print(f"  Output Captures Directory: {captures_dir}")
    print("=" * 80)

    return {
        "total_frames": frame_count,
        "display_fps": display_fps,
        "avg_inference_latency_ms": avg_inf_ms,
        "total_inferences": len(inference_times_ms)
    }


if __name__ == "__main__":
    args = parse_args()
    run_webcam_loop(args)
