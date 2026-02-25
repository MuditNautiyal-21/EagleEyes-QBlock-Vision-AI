# qblock_engine.py

"""
This file act as control brain for the the VISION AI project - EAGLE EYES.
- Uses only big_q_block (class 0) for gating.
- Uses threshold from rules.json (count, visibility, density, layout)
- Runs primary + recovery inferences for a single image and returns rich dict.
- NEW: hybrid strategy
    * raw_class_counts and raw_big_count are kept for analysis
    * a conservative de-dup step merges only near-identical overlapping boxes
      to get a stable big_q_block count for gating.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# --- Paths ---
RULES_JSON = Path("config/rules.json")

# Only Big Q-block is used for gating (class index from rules["meta"]["classes"])
CLS_BIG_DEFAULT = 0


# ---------------- basic utils ----------------
def gpu_available() -> bool:
    return torch.cuda.is_available()


def pick_device(prefer_gpu: bool = True):
    if prefer_gpu and gpu_available():
        return 0  # CUDA:0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_rules(rules_path: Path = RULES_JSON) -> Dict[str, Any]:
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_area_frac(xyxy, H: int, W: int) -> float:
    x1, y1, x2, y2 = xyxy
    return float(max(0, (x2 - x1)) * max(0, (y2 - y1))) / float(H * W)


def crop_gray(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


def ink_features(gray):
    if gray is None or gray.size == 0:
        return None, None
    mean_gray = float(np.mean(gray))
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = float((gray < t).mean())
    return mean_gray, dark_ratio


def within(v, lo, hi):
    # allow None to pass (we already guard where needed)
    return (v is None) or (lo <= v <= hi)


def at_least(v, lo):
    return (v is None) or (v >= lo)


def range_from(rules: Dict[str, Any], path: List[str], fallback: Tuple[float, float]):
    node = rules
    for k in path:
        if k not in node:
            return fallback
        node = node[k]
    if isinstance(node, dict) and "min" in node and "max" in node:
        return float(node["min"]), float(node["max"])
    return fallback


def min_from(rules: Dict[str, Any], path: List[str], fallback: float):
    node = rules
    for k in path:
        if k not in node:
            return fallback
        node = node[k]
    try:
        return float(node)
    except Exception:
        return fallback


def centers(idx: np.ndarray, xyxys: np.ndarray) -> np.ndarray:
    """Always returns shape (N,2), even when empty."""
    if idx is None or len(idx) == 0:
        return np.zeros((0, 2), dtype=float)
    pts = []
    for i in idx:
        x1, y1, x2, y2 = xyxys[i]
        pts.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.array(pts, dtype=float)


def diag(H: int, W: int) -> float:
    return float(np.sqrt(H * H + W * W))


def box_iou_xyxy(a, b) -> float:
    """IoU for [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = aa + bb - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


# ---------------- QBlockEngine ----------------
class QBlockEngine:
    """
    Minimal production engine:
    - loads YOLO model + rules
    - runs primary + recovery pass on ONE image
    - returns OK / NG + which checks failed
    """

    def __init__(self, rules_path: Path = RULES_JSON, prefer_gpu: bool = True):
        # Load rules
        self.rules = load_rules(rules_path)
        meta = self.rules["meta"]

        self.weights_path = Path(meta["weights_path"])
        self.classes = meta.get("classes", ["big_q_block", "small_q_block"])

        # class index for big_q_block
        if "big_q_block" in self.classes:
            self.cls_big = int(self.classes.index("big_q_block"))
        else:
            self.cls_big = CLS_BIG_DEFAULT

        # --- expected counts (strict 14 or 21) ---
        cr = self.rules.get("count_rules", {})
        exp_big_single = int(cr.get("big_expected", 21))
        exp_big_any = cr.get("big_expected_any", [exp_big_single])
        exp_big_any = sorted({int(x) for x in exp_big_any})  # e.g. [14, 21]
        self.allowed_big_counts = set(exp_big_any)
        self.min_big_for_recovery = min(exp_big_any)

        # --- inference params ---
        p = self.rules["count_inference"]["primary"]
        r = self.rules["count_inference"]["recovery"]
        self.p_imgsz = int(p["imgsz"])
        self.p_conf = float(p["conf"])
        self.p_iou = float(p["iou"])
        self.p_tta = bool(p["tta"])

        self.r_imgsz = int(r["imgsz"])
        self.r_conf = float(r["conf"])
        self.r_iou = float(r.get("iou", 0.45))
        self.r_tta = bool(r["tta"])

        # --- visibility thresholds (big only) ---
        vt = self.rules.get("visibility_thresholds", {})
        self.v_conf_big_min = min_from(
            self.rules, ["visibility_thresholds", "conf_min", "big_q_block"], 0.5
        )
        self.v_area_big_min = min_from(
            self.rules, ["visibility_thresholds", "area_min", "big_q_block"], 0.0
        )

        # layout-aware visibility tolerance for big_q_block
        tol_node_big = vt.get("tolerance", {}).get("big_q_block", 0)
        if isinstance(tol_node_big, dict):
            self.v_tol_big_default = int(tol_node_big.get("base", tol_node_big.get("default", 0)))
            self.v_tol_big_for21 = int(tol_node_big.get("for_21", self.v_tol_big_default))
        else:
            self.v_tol_big_default = int(tol_node_big)
            self.v_tol_big_for21 = self.v_tol_big_default

        v_conf_big_rng = vt.get("ranges", {}).get("conf", {}).get("big_q_block", {})
        v_area_big_rng = vt.get("ranges", {}).get("area_frac", {}).get("big_q_block", {})
        self.v_conf_big_max = float(v_conf_big_rng.get("max", 1.0))
        self.v_area_big_max = float(v_area_big_rng.get("max", float("inf")))

        # --- density thresholds (big only) ---
        self.d_gray_big = range_from(
            self.rules, ["density_thresholds", "ranges", "mean_gray", "big_q_block"], (0.0, 255.0)
        )
        self.d_dark_big = range_from(
            self.rules, ["density_thresholds", "ranges", "dark_ratio", "big_q_block"], (0.0, 1.0)
        )

        # --- relative position (big-big + orientation) ---
        self.rp_bb = range_from(
            self.rules, ["relative_position", "ranges", "center_dist_frac", "big_big"], (0.0, 1.0)
        )
        self.rp_angle = range_from(
            self.rules, ["relative_position", "ranges", "orientation_deg"], (-180.0, 180.0)
        )

        # --- device + model ---
        self.device = pick_device(prefer_gpu=prefer_gpu)
        self.use_onnx = str(self.weights_path).lower().endswith(".onnx")

        if self.use_onnx and not gpu_available():
            # ONNX on CPU is still OK; just slower.
            self.device = "cpu"

        self.model = YOLO(self.weights_path)

    # ---------- internal helpers ----------

    def _class_visibility_ok(
        self,
        sel_idx,
        confs,
        xyxys,
        H,
        W,
        conf_min,
        conf_max,
        area_min,
        area_max,
        vis_tol: int = 0,
    ) -> bool:
        if not len(sel_idx):
            return False
        ok_count = 0
        for i in sel_idx:
            c = float(confs[i])
            a = bbox_area_frac(xyxys[i], H, W)
            if at_least(c, conf_min) and c <= conf_max and at_least(a, area_min) and a <= area_max:
                ok_count += 1
        required_ok = max(1, len(sel_idx) - max(0, vis_tol))
        return ok_count >= required_ok

    def _class_density_ok(self, sel_idx, img, xyxys, gray_range, dark_range) -> bool:
        if not len(sel_idx):
            return False
        gvals, dvals = [], []
        for i in sel_idx:
            g = crop_gray(img, xyxys[i])
            mg, dr = ink_features(g)
            if mg is None:
                continue
            gvals.append(mg)
            dvals.append(dr)
        if not gvals or not dvals:
            return False
        return within(np.median(gvals), *gray_range) and within(np.median(dvals), *dark_range)

    def _relpos_ok_for(self, sel_big_idx, xyxys, H, W) -> bool:
        """
        Big-to-big relative position only:
        - median pairwise distance of big centers (normalized by diag)
        - alignment/orientation via SVD
        """
        Cb = centers(sel_big_idx, xyxys)

        def med_pairwise(cs: np.ndarray):
            n = cs.shape[0]
            if n < 2:
                return None
            d = []
            for i in range(n):
                for j in range(i + 1, n):
                    d.append(np.linalg.norm(cs[i] - cs[j]))
            return np.median(d) / diag(H, W)

        bb_med = med_pairwise(Cb)

        ang_ok = True
        if Cb.size > 0 and Cb.shape[0] >= 2:
            U, S, Vt = np.linalg.svd(Cb - Cb.mean(axis=0, keepdims=True))
            vx, vy = Vt[0, 0], Vt[0, 1]
            ang = np.degrees(np.arctan2(vy, vx))
            ang_ok = within(ang, *self.rp_angle)

        return within(bb_med, *self.rp_bb) and ang_ok

    def _dedup_indices(
        self,
        idx_all: np.ndarray,
        xyxys: np.ndarray,
        confs: np.ndarray,
        iou_thr: float = 0.85,
        area_ratio_lo: float = 0.8,
        area_ratio_hi: float = 1.25,
    ) -> np.ndarray:
        """
        Conservative de-dup:
        - sort by confidence desc
        - a box is dropped only if:
            * IoU with an already-kept box is VERY high
            * and area is similar (within area_ratio_lo..area_ratio_hi)
        This should collapse "double boxes on same Q-block" but
        keep genuinely separate Q-blocks apart.
        """
        if idx_all is None or len(idx_all) <= 1:
            return idx_all

        order = sorted(idx_all.tolist(), key=lambda i: float(confs[i]), reverse=True)
        kept = []

        for i in order:
            bi = xyxys[i]
            ai = max(1e-6, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
            keep_me = True
            for j in kept:
                bj = xyxys[j]
                aj = max(1e-6, float((bj[2] - bj[0]) * (bj[3] - bj[1])))
                ratio = ai / aj if aj > 0 else 1.0
                # areas should be reasonably similar
                if ratio < area_ratio_lo or ratio > area_ratio_hi:
                    continue
                iou = box_iou_xyxy(bi, bj)
                if iou >= iou_thr:
                    keep_me = False
                    break
            if keep_me:
                kept.append(i)

        return np.array(kept, dtype=int)

    def _run_pass(self, image_path: Path, mode: str = "primary"):
        if mode == "primary":
            imgsz, conf, iou, tta = self.p_imgsz, self.p_conf, self.p_iou, self.p_tta
        else:
            imgsz, conf, iou, tta = self.r_imgsz, self.r_conf, self.r_iou, self.r_tta

        res_list = self.model.predict(
            source=str(image_path),
            imgsz=int(imgsz),
            device=self.device,
            conf=float(conf),
            iou=float(iou),
            stream=False,
            verbose=False,
            save=False,
            half=False,
            augment=bool(tta),
            classes=[self.cls_big],  # only big_q_block
            workers=0,
            batch=1,
        )
        return res_list[0]

    # ---------- public API ----------
    def evaluate_image(self, image_path: Path, debug: bool = False) -> Dict[str, Any]:
        """
        Evaluate ONE ticket image.
        Returns a dict with:
          - status: "OK" or "NG"
          - big_count (deduped, used for gating)
          - raw_big_count (raw YOLO count)
          - count_ok, vis_big_ok, dens_big_ok, relpos_ok
          - failed_checks: list of strings
          - raw_class_counts: {class_id: count}
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return {
                "image": str(image_path),
                "status": "NG",
                "big_count": 0,
                "raw_big_count": 0,
                "count_ok": False,
                "vis_big_ok": False,
                "dens_big_ok": False,
                "relpos_ok": False,
                "failed_checks": ["image_read"],
                "raw_class_counts": {},
            }

        H, W = img.shape[:2]

        # --- primary pass ---
        res = self._run_pass(image_path, mode="primary")
        boxes = getattr(res, "boxes", None)

        # --- fallback: recovery if no boxes ---
        if boxes is None or boxes.xyxy is None or boxes.cls is None or boxes.conf is None or len(boxes) == 0:
            res = self._run_pass(image_path, mode="recovery")
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                return {
                    "image": str(image_path),
                    "status": "NG",
                    "big_count": 0,
                    "raw_big_count": 0,
                    "count_ok": False,
                    "vis_big_ok": False,
                    "dens_big_ok": False,
                    "relpos_ok": False,
                    "failed_checks": ["no_detections"],
                    "raw_class_counts": {},
                }

        xyxys = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        # raw class stats (for analysis / future predictive stuff)
        if clses.size > 0:
            uniq, cnts = np.unique(clses, return_counts=True)
            raw_class_counts = {int(k): int(v) for k, v in zip(uniq, cnts)}
        else:
            raw_class_counts = {}

        # big-only indices (class 0 or whatever big index is)
        big_idx_all = np.where(clses == self.cls_big)[0]
        raw_big_count = int(len(big_idx_all))

        # extra recovery if clearly short on big blocks
        if raw_big_count < self.min_big_for_recovery:
            res2 = self._run_pass(image_path, mode="recovery")
            b2 = getattr(res2, "boxes", None)
            if b2 and b2.xyxy is not None and len(b2) > 0:
                xyxys = b2.xyxy.cpu().numpy()
                clses = b2.cls.cpu().numpy().astype(int)
                confs = b2.conf.cpu().numpy()
                big_idx_all = np.where(clses == self.cls_big)[0]
                raw_big_count = int(len(big_idx_all))
                if clses.size > 0:
                    uniq, cnts = np.unique(clses, return_counts=True)
                    raw_class_counts = {int(k): int(v) for k, v in zip(uniq, cnts)}
                else:
                    raw_class_counts = {}

        # --- HYBRID STEP: conservative de-dup for big_q_block ---
        if raw_big_count > 0:
            big_idx_uniq = self._dedup_indices(
                big_idx_all,
                xyxys,
                confs,
                iou_thr=0.85,        # very high IoU -> almost same place
                area_ratio_lo=0.8,   # similar area
                area_ratio_hi=1.25,
            )
        else:
            big_idx_uniq = big_idx_all

        big_count = int(len(big_idx_uniq))

        # --- STRICT COUNT RULE: only allow 14 or 21 big blocks ---
        if big_count not in self.allowed_big_counts:
            count_ok = False
            vis_big_ok = False
            dens_big_ok = False
            relpos_ok = False
            failed_checks = ["count_mismatch"]
        else:
            # use deduped big detections for all checks
            sel_big = big_idx_uniq

            count_ok = True  # by definition of allowed_big_counts
            vis_tol_big_use = self.v_tol_big_for21 if big_count >= 21 else self.v_tol_big_default

            vis_big_ok = self._class_visibility_ok(
                sel_big,
                confs,
                xyxys,
                H,
                W,
                self.v_conf_big_min,
                self.v_conf_big_max,
                self.v_area_big_min,
                self.v_area_big_max,
                vis_tol_big_use,
            )
            dens_big_ok = self._class_density_ok(sel_big, img, xyxys, self.d_gray_big, self.d_dark_big)
            relpos_ok = self._relpos_ok_for(sel_big, xyxys, H, W)

            failed_checks = []
            if not count_ok:
                failed_checks.append("count_ok")
            if not vis_big_ok:
                failed_checks.append("vis_big_ok")
            if not dens_big_ok:
                failed_checks.append("dens_big_ok")
            if not relpos_ok:
                failed_checks.append("relpos_ok")

        status = "OK" if (count_ok and vis_big_ok and dens_big_ok and relpos_ok) else "NG"

        result = {
            "image": str(image_path),
            "status": status,
            "big_count": big_count,              # deduped (used for gating)
            "raw_big_count": raw_big_count,      # raw YOLO count
            "count_ok": bool(count_ok),
            "vis_big_ok": bool(vis_big_ok),
            "dens_big_ok": bool(dens_big_ok),
            "relpos_ok": bool(relpos_ok),
            "failed_checks": failed_checks,
            "raw_class_counts": raw_class_counts,
            # --- ADDED FOR VISUALIZATION ---
            "xyxys": xyxys,
            "confs": confs,
            "clses": clses,
            "dedup_indices": big_idx_uniq.tolist() if big_idx_uniq is not None else [],
        }

        if debug:
            print(
                f"[QBlockEngine] {image_path.name} → {status} | "
                f"big(dedup)={big_count}, raw_big={raw_big_count}, "
                f"count_ok={count_ok}, vis={vis_big_ok}, "
                f"dens={dens_big_ok}, relpos={relpos_ok}, "
                f"failed={failed_checks}, raw_classes={raw_class_counts}"
            )

        return result


# CLI Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.qblock_engine <image_path>")
        sys.exit(1)

    engine = QBlockEngine()
    out = engine.evaluate_image(Path(sys.argv[1]), debug=True)
    print(out)
