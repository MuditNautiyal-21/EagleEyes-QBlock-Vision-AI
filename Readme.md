<p align="center">
  <h1 align="center">🦅 Eagle Eyes — Vision AI for Print Defect Detection</h1>
  <p align="center">
    YOLOv8-powered quality inspection for casino payout ticket printing
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/ONNX_Runtime-Inference-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX"/>
  <img src="https://img.shields.io/badge/Accuracy-100%25-00C853?style=for-the-badge" alt="Accuracy"/>
</p>

---

## What & Why

Casino payout ticket printers produce **~30 million tickets per day** across the industry. A **4% defect rate** means over a million potentially misread tickets daily — leading to revenue disputes, regulatory risk, and customer complaints.

**Eagle Eyes** is an automated visual inspection system that detects **Q-Block alignment markers** on printed tickets and validates print quality through a multi-gate decision pipeline. It replaces manual spot-checking with deterministic, auditable, real-time inspection.

---

## Architecture

```mermaid
flowchart LR
    A["📷 Image Input"] --> B["🔍 YOLO Primary Pass"]
    B --> C{"Count\nMatches?"}
    C -- Yes --> E["🧹 Conservative Dedup"]
    C -- No --> D["🔄 Recovery Pass\n(Low conf + TTA)"]
    D --> E
    E --> F["4-Gate Validation"]
    F --> G["✅ OK / ❌ NG Decision"]
    G --> H["📄 CSV Log +\n📁 Image Routing"]

    subgraph F["4-Gate Validation"]
        direction TB
        F1["Count Gate"]
        F2["Visibility Gate"]
        F3["Density Gate"]
        F4["Layout Gate"]
    end

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#0f3460,color:#fff
    style D fill:#16213e,stroke:#e94560,color:#fff,stroke-dasharray: 5 5
    style E fill:#0f3460,stroke:#533483,color:#fff
    style G fill:#1a1a2e,stroke:#00C853,color:#fff
    style H fill:#1a1a2e,stroke:#0f3460,color:#fff
    style F1 fill:#533483,stroke:#e94560,color:#fff
    style F2 fill:#533483,stroke:#e94560,color:#fff
    style F3 fill:#533483,stroke:#e94560,color:#fff
    style F4 fill:#533483,stroke:#e94560,color:#fff
```

> See [ARCHITECTURE.md](ARCHITECTURE.md) for a full technical deep-dive on each stage.

---

## Results

Validated on **2,177 images** from an extended production dataset:

| Metric | Value |
|---|---|
| **Overall Accuracy** | **100%** |
| OK Images (True Positives) | 54 / 54 |
| NG Images (True Negatives) | 2,123 / 2,123 |
| False Positives | 0 |
| False Negatives | 0 |
| Avg. Inference Time | ~83 ms/image (GPU) |

---

## Demo

https://github.com/user-attachments/assets/placeholder

> *Click above to watch the real-time inspection demo. If the video doesn't render, [download it here](GIF/demo.mp4).*

## Dashboard

The real-time OpenCV dashboard shows live annotations, gate results, and decision status:

<p align="center">
  <img src="GIF/dashboard_ok.png" width="48%" alt="Dashboard — OK decision"/>
  &nbsp;
  <img src="GIF/dashboard_ng.png" width="48%" alt="Dashboard — NG decision"/>
</p>

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | Ultralytics YOLOv8 (ONNX export) |
| Image Processing | OpenCV 4.8+ |
| Deep Learning | PyTorch 2.0+ |
| Inference Runtime | ONNX Runtime |
| Scientific Compute | NumPy |
| Language | Python 3.10+ |

---

## Installation

```bash
git clone https://github.com/MuditNautiyal-21/EagleEyes-QBlock-Vision-AI.git
cd EagleEyes-QBlock-Vision-AI
pip install -r requirements.txt
```

> **Note:** Model weights (`.onnx`) are excluded from the repository due to size. Place the trained model at the path specified in `config/rules.json` → `meta.weights_path`.

## Usage

```bash
python app/runner_visualizer.py
```

1. Drop ticket images into `data/incoming/`
2. The dashboard opens automatically — images are evaluated in real time
3. Results are logged to `data/results/visualizer_results.csv`
4. Images are routed to `data/Good_Images/` (OK) or `data/No_Good_Images/` (NG)
5. Press **`q`** to exit

---

## Configuration

All operational thresholds live in [`config/rules.json`](config/rules.json). See [`config/rules_template.json`](config/rules_template.json) for an annotated template with placeholder values.

| Section | Controls |
|---|---|
| `count_inference` | Primary & recovery pass parameters (confidence, IoU, TTA, image size) |
| `count_rules` | Expected marker counts and tolerances per layout |
| `visibility_thresholds` | Confidence and area-fraction bounds |
| `density_thresholds` | Grayscale mean and dark-pixel ratio ranges |
| `relative_position` | Spatial constraint parameters |

**Tuning workflow:** Change one threshold group → run a validation pack of known OK/NG images → review the CSV log for `failed_checks` distribution.

---

## Project Structure

```
EagleEyes_PROD/
├── app/
│   ├── qblock_engine.py        # Core engine — inference, dedup, 4-gate validation
│   ├── runner_visualizer.py    # Production runner + OpenCV dashboard
│   └── fx_router.py            # Output router (PLC-ready, optional)
├── config/
│   ├── rules.json              # All production thresholds
│   └── rules_template.json     # Annotated config template
├── models/                     # YOLO model weights (gitignored)
├── data/
│   ├── incoming/               # Drop images here
│   ├── Good_Images/            # OK results routed here
│   ├── No_Good_Images/         # NG results routed here
│   └── results/                # CSV audit logs
├── ARCHITECTURE.md             # Technical deep-dive
├── requirements.txt
└── .gitignore
```

---

## Known Limitations

- **YOLO hallucinations** — False positives can increase when the recovery pass is triggered at very low confidence thresholds
- **Ink density sensitivity** — Changes in printer exposure or lighting conditions can shift grayscale/dark-ratio values outside calibrated ranges
- **Single ticket layout** — Currently tuned for specific Q-Block layouts (14 or 21 markers); new layouts require re-calibration

---

## Attribution

Built by **Team Affida** as a capstone project for the **MS Data Science** program at the **University at Buffalo**, in collaboration with **Nissha Medical Technologies**.

> [!IMPORTANT]
> This project was developed as part of a university capstone collaboration with Nissha Medical Technologies. Source code is shared for portfolio and educational purposes. For usage permissions, please contact the author.
