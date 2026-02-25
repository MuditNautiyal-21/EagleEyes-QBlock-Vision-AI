# app/fx_router.py
"""
fx_router.py
- Converts QBlockEngine outputs into unified actions.
- Handles: OK / NG mapping, CSV logging, future PLC hooks.
- Very lightweight so production system stays stable.
"""

from pathlib import Path
import csv
import datetime
from typing import Dict, Any

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

CSV_LOG = LOG_DIR / "results_log.csv"


class FXRouter:
    def __init__(self):
        # ensure CSV exists with headers
        if not CSV_LOG.exists():
            with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "image",
                    "status",
                    "big_count",
                    "failed_checks"
                ])

    # ------------------------------------------------------------------
    # PUBLIC: route_result
    # ------------------------------------------------------------------
    def route_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes result from QBlockEngine.evaluate_image()
        Returns a unified output dict for live console + PLC hook.
        """

        status = result.get("status", "NG")
        img = result.get("image", "")
        big_count = result.get("big_count", 0)
        fails = result.get("failed_checks", [])

        # --- console format for live mode ---
        console_str = (
            f"[LIVE] {Path(img).name} → {status} "
            f"| big={big_count} | fails={fails}"
        )

        # --- write to CSV log ---
        self._write_csv(result)

        # --- prepare output for PLC layer (future hook) ---
        mapped_signal = self._map_signal(status)

        return {
            "console": console_str,
            "signal": mapped_signal,  # { "green":1 , "red":0 , "yellow":0 }
            "raw": result
        }

    # ------------------------------------------------------------------
    def _map_signal(self, status: str) -> Dict[str, int]:
        """
        PLC mapping (future-ready).
        Only OK and NG required now.
        Yellow reserved for WARNING states in future.
        """
        if status == "OK":
            return {"green": 1, "red": 0, "yellow": 0}

        return {"green": 0, "red": 1, "yellow": 0}

    # ------------------------------------------------------------------
    def _write_csv(self, result: Dict[str, Any]):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts,
                result.get("image", ""),
                result.get("status", ""),
                result.get("big_count", ""),
                "|".join(result.get("failed_checks", [])),
            ])
