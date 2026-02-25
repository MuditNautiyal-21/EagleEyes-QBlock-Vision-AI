# Architecture Deep-Dive

> Technical reference for the Eagle Eyes defect detection pipeline.
> For setup and usage, see the [README](Readme.md).

---

## Pipeline Overview

Eagle Eyes processes ticket images through a sequential pipeline designed for **zero false positives** — the system prefers to reject a borderline-OK image rather than pass a defective one.

```
Image → YOLO Inference → Conservative Dedup → 4-Gate Validation → OK/NG → CSV + Routing
```

Each stage is described below.

---

## 1. Hybrid Inference Strategy

Detection runs in two phases to balance **speed** and **recall**.

### Primary Pass

- Standard YOLO inference at high resolution (1280px)
- Moderate confidence threshold (e.g., 0.05) — keeps most real detections while filtering noise
- NMS IoU at 0.5, no Test-Time Augmentation
- **If the deduplicated count matches an expected layout (14 or 21), skip recovery**

### Recovery Pass

Triggered **only** when the primary count doesn't match any expected layout:

- Same resolution, but confidence threshold drops to near-zero (0.001)
- TTA enabled (multi-scale + flip augmentation) to catch faded or partial markers
- NMS IoU tightened to 0.45 to reduce duplicate boxes from augmentation
- **Merges** recovery detections with primary detections before dedup

**Why hybrid?** Running TTA on every image would triple inference time. Since ~90% of images pass on the primary pass, the recovery path is rarely triggered in production.

---

## 2. Conservative Dedup

After inference (or merge), overlapping boxes are deduplicated with **strict criteria** to avoid accidentally removing real, closely-spaced markers.

A detection is suppressed **only if ALL conditions are met**:

| Condition | Threshold | Rationale |
|---|---|---|
| IoU with a kept box | ≥ 0.85 | Must be nearly the same box, not just overlapping |
| Area ratio | 0.80 – 1.25 | Sizes must be similar (prevents merging big + small artifacts) |

**Algorithm:**
1. Sort all detections by confidence (descending)
2. For each detection, check against all already-kept boxes
3. Suppress only if both IoU and area-ratio conditions are met
4. Keep the higher-confidence detection

This is intentionally more conservative than standard NMS. In standard NMS, an IoU of ~0.5 triggers suppression; here, 0.85 ensures only true duplicates are removed.

---

## 3. Four Validation Gates

After dedup, the remaining detections are validated through four independent gates. **All four must pass for an OK decision.** Any failure produces an NG result with a `failed_checks` list for auditability.

### Gate 1: Count

**What it checks:** Does the deduplicated big-block count match an allowed layout?

- Expected counts are configured in `count_rules.big_expected_any` (e.g., `[14, 21]`)
- Each layout can have its own tolerance (e.g., 21-block layout allows ±1, 14-block allows ±0)

**Why it matters:** Missing or extra markers indicate print registration failures, skipped print heads, or paper feed errors.

### Gate 2: Visibility

**What it checks:** Are all detected markers confident and properly sized?

For each detected big-block:
- **Confidence** must fall within `[conf_min, conf_max]`
- **Area fraction** (box area ÷ image area) must fall within `[area_min, area_max]`

A configurable tolerance allows a small number of blocks to fall outside bounds (e.g., up to 3 in a 21-block layout) to accommodate edge markers that may be partially cut off.

**Why it matters:** Low-confidence or undersized detections suggest faded ink, partial occlusion, or model hallucinations — all correlate with print defects.

### Gate 3: Density

**What it checks:** Is the ink coverage within calibrated norms?

For each detected block, a grayscale crop is extracted and two features are computed:
- **Mean gray value** (0=black, 255=white) — measures overall ink intensity
- **Dark pixel ratio** (fraction below Otsu threshold) — measures ink fill coverage

The gate uses **median statistics** across all blocks (not per-block) and compares against calibrated ranges.

**Why it matters:** Under-inking produces markers that scan correctly but fail downstream optical readers. Over-inking causes bleed that merges adjacent features.

### Gate 4: Layout (Relative Position)

**What it checks:** Are the markers in the correct spatial arrangement?

- Computes **median pairwise distance** between big-block centers, normalized by image diagonal
- Uses **SVD-based orientation** estimation to detect rotation or skew
- Values are compared against calibrated ranges derived from known-good images

**Why it matters:** A correct count with correct ink density can still fail if markers are misaligned — indicating paper jam, roller slip, or registration drift.

---

## 4. Output & Logging

Every evaluation produces:

| Output | Description |
|---|---|
| `status` | `"OK"` or `"NG"` |
| `big_count` | Deduplicated marker count |
| `failed_checks` | List of gate names that failed (empty for OK) |
| `count_ok`, `vis_big_ok`, `dens_big_ok`, `relpos_ok` | Per-gate pass/fail booleans |
| CSV row | Appended to `data/results/visualizer_results.csv` |
| Image routing | Moved to `data/Good_Images/` or `data/No_Good_Images/` |

The optional `FXRouter` (`app/fx_router.py`) provides a standardized output layer with PLC-ready signal mapping (`green`/`red`/`yellow`) for future factory integration.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Big-block only gating | Small blocks have higher detection variance; gating on big blocks alone achieves 100% accuracy |
| Conservative dedup (IoU ≥ 0.85) | Standard NMS is too aggressive for closely-spaced markers on dense layouts |
| Recovery pass on-demand | Keeps average inference time low while catching edge cases |
| Median-based density | Robust to individual outlier blocks (e.g., partially cut edge markers) |
| All thresholds in JSON | No hardcoded values — enables tuning without code changes |
