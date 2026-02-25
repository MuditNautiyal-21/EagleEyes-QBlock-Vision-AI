# app/runner_visualizer.py
"""
Real-time visualizer for EagleEyes_PROD.

- Watches data/incoming/ for new images.
- Runs QBlockEngine on each image.
- Displays the image in an OpenCV window with annotations:
    - Final status (OK/NG) and failure reasons.
    - De-duplicated bounding boxes for 'big_q_block'.
- Press 'q' to quit the visualizer.
- This script now ALSO performs the actions of runner_folder_watcher.py:
    - Logs results to a CSV file.
    - Moves processed images to Good_Images or No_Good_Images.
"""

import time
import csv
import datetime as dt
from pathlib import Path
import cv2
import numpy as np
import shutil

from app.qblock_engine import QBlockEngine

# --- CONFIG ---
INCOMING_DIR = Path("data/incoming")
GOOD_DIR = Path("data/Good_Images")
NG_DIR = Path("data/No_Good_Images")
RESULTS_DIR = Path("data/results")
RESULTS_CSV = RESULTS_DIR / "visualizer_results.csv"

POLL_INTERVAL_SEC = 0.5  # How often to scan the folder
WINDOW_NAME = "Eagle Eyes - Real-Time Visualizer"

def ensure_dirs():
    """Make sure all necessary directories exist."""
    for d in [INCOMING_DIR, GOOD_DIR, NG_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def init_csv_if_needed():
    """Creates the CSV log file with headers if it doesn't exist."""
    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp", "image_name", "status", "big_count",
                    "raw_big_count", "count_ok", "vis_big_ok", "dens_big_ok",
                    "relpos_ok", "failed_checks"
                ]
            )

def append_result_to_csv(result: dict):
    """Appends a single evaluation result to the CSV log."""
    ts = dt.datetime.now().isoformat(timespec="seconds")
    failed = ";".join(result.get("failed_checks", [])) if result.get("failed_checks") else ""
    row = [
        ts, Path(result["image"]).name, result["status"],
        result.get("big_count", 0), result.get("raw_big_count", 0),
        int(result.get("count_ok", False)), int(result.get("vis_big_ok", False)),
        int(result.get("dens_big_ok", False)), int(result.get("relpos_ok", False)),
        failed
    ]
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def draw_on_image(img: np.ndarray, result: dict, engine: QBlockEngine):
    """Draws the engine's results onto the image."""
    status = result.get("status", "NG")
    big_count = result.get("big_count", 0)
    failed_checks = result.get("failed_checks", [])

    # --- Draw Bounding Boxes (de-duplicated only) ---
    xyxys = result.get("xyxys")
    confs = result.get("confs")
    clses = result.get("clses") # Get the class array
    dedup_indices = result.get("dedup_indices", []) # Use empty list as default

    # Get the class index for 'big_q_block' dynamically from the engine's rules
    try:
        big_q_block_cls_index = engine.classes.index("big_q_block")
    except (ValueError, AttributeError):
        big_q_block_cls_index = 0 # Fallback to 0 if not found
    
    # --- Draw ALL raw 'big_q_block' detections first (in a muted color) ---
    if all(v is not None for v in [xyxys, confs, clses]):
        for i in range(len(xyxys)):
            if int(clses[i]) != big_q_block_cls_index:
                continue
            
            # Draw a less prominent box for all raw detections
            x1, y1, x2, y2 = map(int, xyxys[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 0), 1) # Blue, thin box

    # --- Draw the FINAL, de-duplicated 'big_q_block' boxes on top (in bright green) ---
    if all(v is not None for v in [xyxys, confs, clses]):
        # Create a set for quick lookup of de-duplicated indices
        dedup_set = set(dedup_indices if dedup_indices is not None else [])
        for i in dedup_set:
            # This check is redundant if dedup_indices only contains big_q_blocks, but good for safety
            if int(clses[i]) != big_q_block_cls_index:
                continue
            
            x1, y1, x2, y2 = map(int, xyxys[i])
            conf = confs[i]

            # Draw bright green box for final, validated detections
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with confidence for the final boxes
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 180, 0), -1)
            cv2.putText(img, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def create_dashboard(annotated_img: np.ndarray, result: dict, elapsed_ms: float) -> np.ndarray:
    """Creates a dashboard view with the image on top and a metrics table below."""
    # --- Adaptive Layout Setup ---
    # Base width for scaling calculations. All positions and sizes are relative to this.
    dash_width = 1920 
    img_area_height = 1080
    table_area_height = 320
    dash_height = img_area_height + table_area_height # Total canvas height
    dashboard = np.full((dash_height, dash_width, 3), (24, 24, 24), dtype=np.uint8)

    # Define column positions as fractions of width
    col1_label = int(dash_width * 0.03)
    col1_value = int(dash_width * 0.20)
    col2_label = int(dash_width * 0.55)
    col2_value = int(dash_width * 0.70)

    # --- Top Half: Annotated Image ---
    h, w = annotated_img.shape[:2]
    scale = min(dash_width / w, img_area_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(annotated_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste image into the top-center
    x_offset = (dash_width - new_w) // 2
    y_offset = (img_area_height - new_h) // 2
    dashboard[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    # --- Bottom Half: Metrics Table ---
    table_y_start = img_area_height

    # Helper to truncate long text
    def truncate_text(text: str, max_width: int, font, scale, thickness):
        (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if text_w <= max_width:
            return text
        
        ellipsis = "..."
        (ellipsis_w, _), _ = cv2.getTextSize(ellipsis, font, scale, thickness)
        
        # Find how many characters can fit
        for i in range(len(text), 0, -1):
            shortened_text = text[:i] + ellipsis
            (short_w, _), _ = cv2.getTextSize(shortened_text, font, scale, thickness)
            if short_w <= max_width:
                return shortened_text
        return ellipsis

    # Helper for drawing table rows
    def draw_row(y, label, value, value_color=(255, 255, 255), font_scale=1.0):
        cv2.putText(dashboard, label, (col1_label, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(dashboard, str(value), (col1_value, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, value_color, 2, cv2.LINE_AA)

    # --- Populate Table ---
    row_y = [table_y_start + int(h) for h in np.linspace(60, table_area_height - 40, 5)]

    # Row 1: Image Name
    img_name = Path(result["image"]).name
    max_name_width = col2_label - col1_value - 20 # Max width for the image name
    truncated_name = truncate_text(img_name, max_name_width, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    draw_row(row_y[0], "Image Name:", truncated_name)
    
    # Row 2: Final Status
    status = result.get("status", "NG")
    status_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
    draw_row(row_y[1], "Final Status:", status, status_color, font_scale=1.2)

    # Row 3: Processing Time
    draw_row(row_y[2], "Processing Time:", f"{elapsed_ms:.0f} ms")

    # Row 4: Q-Block Count
    big_count = result.get("big_count", 0)
    raw_big_count = result.get("raw_big_count", 0)
    draw_row(row_y[3], "Q-Block Count:", f"{big_count} (from {raw_big_count} raw)")

    # Row 5: Median Confidence
    median_conf_val = "N/A"
    if result.get("confs") is not None and result.get("dedup_indices") is not None:
        dedup_confs = [result["confs"][i] for i in result["dedup_indices"]]
        if dedup_confs:
            median_conf_val = f"{np.median(dedup_confs):.4f}"
    draw_row(row_y[4], "Median Confidence:", median_conf_val)

    # --- Right Side: Accuracy/Checks Table ---
    cv2.putText(dashboard, "Validation Checks", (col2_label, row_y[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(dashboard, (col2_label - 10, row_y[0] + 15), (dash_width - 30, row_y[0] + 15), (80, 80, 80), 2)

    def draw_check(y, name, is_ok, font_scale=0.9):
        status_text = "PASS" if is_ok else "FAIL"
        color = (0, 255, 0) if is_ok else (0, 0, 255)
        cv2.putText(dashboard, f"{name}:", (col2_label, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(dashboard, status_text, (col2_value, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

    draw_check(row_y[1], "Count", result.get("count_ok", False))
    draw_check(row_y[2], "Visibility", result.get("vis_big_ok", False))
    draw_check(row_y[3], "Density", result.get("dens_big_ok", False))
    draw_check(row_y[4], "Layout", result.get("relpos_ok", False))

    return dashboard


def run_visualizer():
    """Main loop to watch folder and display results."""
    ensure_dirs()
    init_csv_if_needed()

    print("[Visualizer] Starting EagleEyes_PROD visualizer...")
    engine = QBlockEngine()
    print(f"[Visualizer] Engine loaded. Watching: {INCOMING_DIR.resolve()}")
    print("[Visualizer] Press 'q' in the display window to quit.")

    # --- Create a resizable window ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 840) # Start with a reasonable default size

    # --- Create an initial placeholder window ---
    placeholder_dash = np.full((1400, 1920, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(placeholder_dash, "Waiting for image in data/incoming...",
                (500, 700), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, placeholder_dash)
    # ---

    while True:
        # Find all available images
        image_paths = list(INCOMING_DIR.glob("*.*"))
        
        if not image_paths:
            # If no images, pause briefly to prevent high CPU usage
            time.sleep(POLL_INTERVAL_SEC)
            # Check for quit key during idle time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Process all available images without a long sleep
        for img_path in image_paths:
            quit_pressed = False
            if not img_path.is_file(): continue

            print(f"\n[+] Processing new image: {img_path.name}", end="", flush=True)
            start_time = time.monotonic()

            # 1. Evaluate, draw, log, and move
            result = engine.evaluate_image(img_path)
            image = cv2.imread(str(img_path))
            draw_on_image(image, result, engine)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            dashboard = create_dashboard(image, result, elapsed_ms)
            cv2.imshow(WINDOW_NAME, dashboard)
            append_result_to_csv(result)
            dst_dir = GOOD_DIR if result["status"] == "OK" else NG_DIR
            shutil.move(str(img_path), dst_dir / img_path.name)
            
            print(f" -> {result['status']} in {elapsed_ms:.0f} ms. Moved to {dst_dir.name}")

            # Allow window to refresh and check for quit key for each image
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_pressed = True
                break
        
        if quit_pressed:
            break # Exit the main while loop

    cv2.destroyAllWindows()
    print("[Visualizer] Shutting down.")

if __name__ == "__main__":
    run_visualizer()