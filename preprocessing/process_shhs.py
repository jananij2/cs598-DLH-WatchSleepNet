"""Stage 2 preprocessing for the Sleep Heart Health Study (SHHS).

Reads the compressed per-subject NPZ files produced by ``extract_shhs.py``
(Stage 1), runs R-peak detection via biosppy, computes a per-sample IBI time
series, downsamples to 25 Hz, and writes the final ``.npz`` cache files
expected by the WatchSleepNet ``SSDataset`` loader.

This script is designed to run on a remote or GPU-capable machine after the
extracted NPZs have been transferred from the local EDF source.

Output NPZ keys (same format as preprocess_shhs.py)
----------------------------------------------------
data   : float32 ndarray — IBI signal at 25 Hz
stages : int32 ndarray   — per-sample sleep stage at 25 Hz
fs     : int64 scalar    — 25
ahi    : float32 scalar  — Apnea-Hypopnea Index

Usage::

    python process_shhs.py \\
        --signal_dir /remote/SHHS_extracted \\
        --dst_dir /remote/SHHS_MESA_IBI \\
        [--workers N]
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import biosppy.signals.ecg
import numpy as np
from scipy.signal import resample_poly

__all__ = ["process_shhs"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match preprocess_shhs.py)
# ---------------------------------------------------------------------------

_TARGET_FS: int = 25
_IBI_OUTLIER_THRESHOLD: float = 2.0  # seconds; HR < 30 BPM treated as artifact


# ---------------------------------------------------------------------------
# IBI computation
# ---------------------------------------------------------------------------


def _compute_ibi(signal: np.ndarray, fs: float) -> np.ndarray:
    """Compute per-sample IBI from ECG via biosppy R-peak detection.

    Args:
        signal: 1-D float64 ECG array at ``fs`` Hz.
        fs:     Sampling frequency in Hz.

    Returns:
        float64 ndarray of shape ``(len(signal),)`` with per-sample IBI in
        seconds.  Intervals ≥ ``_IBI_OUTLIER_THRESHOLD`` are zeroed out.
    """
    out = biosppy.signals.ecg.ecg(signal, sampling_rate=fs, show=False)
    r_peaks = out["rpeaks"]
    ibi_values = np.diff(r_peaks) / fs

    ibi = np.zeros(signal.shape, dtype=np.float64)
    for i in range(len(r_peaks) - 1):
        ibi[r_peaks[i] : r_peaks[i + 1]] = ibi_values[i]
    ibi[ibi >= _IBI_OUTLIER_THRESHOLD] = 0.0
    return ibi


def _process_subject(
    signal: np.ndarray, fs: int, stages: np.ndarray, ahi: float
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """Run R-peak detection, IBI computation, and downsampling for one subject.

    Args:
        signal: Raw ECG array at ``fs`` Hz.
        fs:     Sampling rate of ``signal``.
        stages: Per-sample sleep stage array at ``fs`` Hz.
        ahi:    Apnea-Hypopnea Index.

    Returns:
        ``(ibi_ds, _TARGET_FS, stages_ds, ahi)`` at 25 Hz.
    """
    ibi = _compute_ibi(signal.astype(np.float64), fs)

    if fs % _TARGET_FS == 0:
        factor = fs // _TARGET_FS
        ibi_ds    = ibi[::factor]
        stages_ds = stages[::factor].astype(np.float64)
    else:
        ibi_ds    = resample_poly(ibi, _TARGET_FS, fs)
        stages_ds = resample_poly(stages.astype(np.float64), _TARGET_FS, fs)

    return ibi_ds, _TARGET_FS, stages_ds, ahi


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def _process_one(args: Tuple) -> Tuple[str, str]:
    """Worker function executed in a subprocess.

    Args:
        args: ``(subject_key, signal_path, dst_dir)``

    Returns:
        ``(subject_key, outcome)`` — ``"ok"``, ``"skip"``, or ``"warn"``.
    """
    subject_key, signal_path, dst_dir = args
    out_path = Path(dst_dir) / f"{subject_key}.npz"

    if out_path.exists():
        return subject_key, "skip"

    try:
        npz = np.load(signal_path)
        signal = npz["signal"]
        stages = npz["stages"]
        fs     = int(npz["fs"])
        ahi    = float(npz["ahi"])

        ibi_ds, fs_out, stages_ds, ahi_val = _process_subject(signal, fs, stages, ahi)

        np.savez(
            str(out_path),
            data=ibi_ds.astype(np.float32),
            stages=stages_ds.astype(np.int32),
            fs=np.int64(fs_out),
            ahi=np.float32(ahi_val),
        )
        return subject_key, "ok"
    except Exception as exc:
        logger.warning("[%s] Failed: %s", subject_key, exc)
        return subject_key, "warn"


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------


def process_shhs(
    signal_dir: "str | Path",
    dst_dir: "str | Path",
    workers: Optional[int] = None,
) -> Dict[str, str]:
    """Run IBI computation on extracted SHHS NPZs and write final cache files.

    Args:
        signal_dir: Directory of NPZ files produced by ``extract_shhs.py``.
        dst_dir:    Output directory for final ``.npz`` files.  Created if absent.
        workers:    Number of parallel worker processes.  Defaults to
                    ``cpu_count() // 2``.

    Returns:
        Dict mapping ``subject_key → outcome`` (``"ok"``, ``"skip"``, ``"warn"``).
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    signal_dir = Path(signal_dir)
    dst_dir    = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    signal_files = sorted(signal_dir.glob("shhs*.npz"))
    if not signal_files:
        raise FileNotFoundError(f"No shhs*.npz files found in {signal_dir}")

    work_items: List[Tuple] = [
        (f.stem, str(f), str(dst_dir)) for f in signal_files
    ]

    already_done = [item for item in work_items if (dst_dir / f"{item[0]}.npz").exists()]
    work_items   = [item for item in work_items if not (dst_dir / f"{item[0]}.npz").exists()]
    if already_done:
        logger.info("Skipping %d already-processed subject(s) (npz exists).", len(already_done))
    logger.info("Subjects to process: %d", len(work_items))

    n_workers = workers if workers is not None else max(1, cpu_count() // 2)
    logger.info("Using %d worker processes.", n_workers)

    results: Dict[str, str] = {}
    try:
        with Pool(n_workers) as pool:
            iterator = pool.imap_unordered(_process_one, work_items)
            if tqdm:
                iterator = tqdm(iterator, total=len(work_items), desc="SHHS process")
            for subject_key, outcome in iterator:
                results[subject_key] = outcome
                if outcome == "ok":
                    logger.info("[%s] ok", subject_key)
                elif outcome == "skip":
                    logger.debug("[%s] skipped (already exists)", subject_key)
    except KeyboardInterrupt:
        ok = sum(1 for v in results.values() if v == "ok")
        skipped = sum(1 for v in results.values() if v == "skip")
        print(
            f"\nInterrupted after {ok} completed, {skipped} skipped. Re-run to resume.",
            flush=True,
        )
        raise

    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v == "skip")
    warned = sum(1 for v in results.values() if v == "warn")
    logger.info("Finished. ok=%d, skipped=%d, warnings=%d", ok, skipped, warned)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: compute SHHS IBI from extracted NPZs and write final cache files."
    )
    parser.add_argument("--signal_dir", required=True,
                        help="Directory of NPZ files from extract_shhs.py.")
    parser.add_argument("--dst_dir", required=True,
                        help="Output directory for final .npz files.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: cpu_count // 2).")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    results = process_shhs(
        signal_dir=args.signal_dir,
        dst_dir=args.dst_dir,
        workers=args.workers,
    )
    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v == "skip")
    warned = sum(1 for v in results.values() if v == "warn")
    print(f"\nDone. ok={ok}, skipped={skipped}, warnings={warned}")
