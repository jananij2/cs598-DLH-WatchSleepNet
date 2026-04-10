"""Stage 1 preprocessing for the Sleep Heart Health Study (SHHS).

Reads each subject's EDF, extracts only the ECG channel, parses sleep stage
annotations, and writes a compressed ``.npz`` per subject.  No peak detection
is performed — output is meant to be transferred to a remote/GPU machine where
``process_shhs.py`` (Stage 2) computes IBI.

Multi-channel EDFs (6–8 channels) are reduced to a single ECG channel, cutting
per-subject size from ~35 MB EDF to ~8 MB compressed NPZ (~80 % reduction).

Output NPZ keys
---------------
signal  : float32 ndarray, shape (n_samples,) at ``fs`` Hz
stages  : int8 ndarray,    shape (n_samples,) — per-sample sleep stage
          (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
fs      : int64 scalar — sampling rate of ``signal`` and ``stages``
ahi     : float32 scalar — Apnea-Hypopnea Index

Usage::

    python extract_shhs.py \\
        --src_dir /Volumes/MastersData/shhs \\
        --dst_dir /Volumes/MastersData/SHHS_extracted \\
        --harmonized_csv /Volumes/MastersData/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv

Add ``--target_fs 100`` to downsample before saving (~4 MB/subject instead of ~8 MB).
Both biosppy and process_shhs.py accept arbitrary sampling rates.

Where ``src_dir`` contains::

    polysomnography/edfs/shhs1/*.edf
    polysomnography/edfs/shhs2/*.edf
    polysomnography/annotations-events-profusion/shhs1/*-profusion.xml
    polysomnography/annotations-events-profusion/shhs2/*-profusion.xml
"""

from __future__ import annotations

import argparse
import logging
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

__all__ = ["extract_shhs"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VISITS: List[str] = ["shhs1", "shhs2"]
_ECG_CHANNEL: str = "ECG"
_EXCLUDE_CHANNELS: List[str] = ["SaO2", "H.R.", "SOUND", "AIRFLOW", "POSITION", "LIGHT"]
_EXCLUDED_SUBJECTS: frozenset = frozenset({"shhs1-204822"})
_MAX_EPOCHS: int = 1100  # ~9.17 hours; matches WatchSleepNet truncation

_ID_COL: str = "nsrrid"
_AHI_COL: str = "nsrr_ahi_hp3r_aasm15"


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def _extract_subject(
    edf_path: Path, ann_path: Path, ahi: float, target_fs: Optional[int]
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """Extract ECG + stage labels from one SHHS EDF + XML pair.

    Args:
        edf_path:  Path to the subject's EDF file.
        ann_path:  Path to the paired ``*-profusion.xml`` annotation file.
        ahi:       Apnea-Hypopnea Index for this subject.
        target_fs: If given, resample to this rate before returning.
                   ``None`` keeps the native EDF rate (typically 125 Hz).

    Returns:
        ``(signal, fs, stages, ahi)`` ready to write to NPZ.
    """
    raw = mne.io.read_raw_edf(
        str(edf_path), verbose=0, exclude=_EXCLUDE_CHANNELS, infer_types=False
    )
    raw.pick(picks=[_ECG_CHANNEL])
    native_fs = raw.info["sfreq"]
    signal = raw.get_data().flatten()

    # Parse and expand sleep stage annotations
    tree = ET.parse(str(ann_path))
    root = tree.getroot()
    epoch_stages = np.array(
        [s.text for s in root[4].findall("SleepStage")], dtype=np.int8
    )
    epoch_stages[epoch_stages == 4] = 3  # N4 → N3
    epoch_stages[epoch_stages == 5] = 4  # REM: 5 → 4
    samples_per_epoch = int(30 * native_fs)
    stages = np.repeat(epoch_stages, samples_per_epoch)

    # Truncate to max epochs and align lengths
    max_samples = _MAX_EPOCHS * samples_per_epoch
    signal = signal[:max_samples]
    stages = stages[:max_samples]
    min_len = min(len(signal), len(stages))
    signal = signal[:min_len]
    stages = stages[:min_len]

    if target_fs is not None and int(native_fs) != target_fs:
        signal = resample_poly(signal, target_fs, int(native_fs)).astype(np.float32)
        stages = resample_poly(
            stages.astype(np.float64), target_fs, int(native_fs)
        ).astype(np.int8)
        out_fs = target_fs
    else:
        signal = signal.astype(np.float32)
        out_fs = int(native_fs)

    return signal, out_fs, stages, ahi


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def _extract_one(args: Tuple) -> Tuple[str, str]:
    """Worker function executed in a subprocess.

    Args:
        args: ``(subject_key, edf_path, ann_path, ahi, dst_dir, target_fs)``

    Returns:
        ``(subject_key, outcome)`` where outcome is ``"ok"``, ``"skip"``, or ``"warn"``.
    """
    subject_key, edf_path, ann_path, ahi, dst_dir, target_fs = args
    out_path = Path(dst_dir) / f"{subject_key}.npz"

    if out_path.exists():
        return subject_key, "skip"

    try:
        signal, fs, stages, ahi_val = _extract_subject(
            Path(edf_path), Path(ann_path), ahi, target_fs
        )
        np.savez_compressed(
            str(out_path),
            signal=signal,
            stages=stages,
            fs=np.int64(fs),
            ahi=np.float32(ahi_val),
        )
        return subject_key, "ok"
    except Exception as exc:
        logger.warning("[%s] Failed: %s", subject_key, exc)
        return subject_key, "warn"


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------


def extract_shhs(
    src_dir: "str | Path",
    dst_dir: "str | Path",
    harmonized_csv: "str | Path",
    target_fs: Optional[int] = None,
    workers: Optional[int] = None,
) -> Dict[str, str]:
    """Extract ECG signals from SHHS EDFs into compressed per-subject NPZ files.

    Processes both SHHS1 and SHHS2 visits.  Already-existing output files are
    skipped (idempotent re-runs).

    Args:
        src_dir:        Root SHHS directory containing ``polysomnography/``.
        dst_dir:        Output directory for ``.npz`` files.  Created if absent.
        harmonized_csv: Path to ``shhs-harmonized-dataset-0.21.0.csv``.
        target_fs:      Optional output sampling rate (e.g. 100 Hz).  ``None``
                        keeps the native EDF rate.
        workers:        Number of parallel worker processes.  Defaults to
                        ``cpu_count() // 2``.

    Returns:
        Dict mapping ``subject_key → outcome`` (``"ok"``, ``"skip"``, ``"warn"``).
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading harmonized CSV: %s", harmonized_csv)
    info_df = pd.read_csv(harmonized_csv, dtype={_ID_COL: str})
    if "visitnumber" in info_df.columns:
        info_df = info_df[info_df["visitnumber"] == 1]
    info_df[_ID_COL] = info_df[_ID_COL].str.lstrip("0")
    ahi_lookup: Dict[str, float] = dict(
        zip(info_df[_ID_COL], info_df[_AHI_COL].astype(float))
    )

    work_items: List[Tuple] = []
    n_missing_ahi: int = 0
    for visit in _VISITS:
        edf_dir = src_dir / "polysomnography" / "edfs" / visit
        ann_dir = src_dir / "polysomnography" / "annotations-events-profusion" / visit

        if not edf_dir.exists():
            warnings.warn(f"{edf_dir} not found — skipping {visit}.", stacklevel=2)
            continue

        for edf_path in sorted(edf_dir.glob("*.edf")):
            sid = edf_path.stem.split("-")[1]
            subject_key = f"{visit}-{sid}"

            if subject_key in _EXCLUDED_SUBJECTS:
                logger.debug("Skipping excluded subject: %s", subject_key)
                continue

            ann_path = ann_dir / f"{edf_path.stem}-profusion.xml"
            if not ann_path.exists():
                logger.debug("Missing annotation for %s, skipping.", subject_key)
                continue

            ahi = ahi_lookup.get(sid.lstrip("0"), None)
            if ahi is None or (isinstance(ahi, float) and np.isnan(ahi)):
                logger.debug("[%s] AHI not found in CSV; skipping.", subject_key)
                n_missing_ahi += 1
                continue

            work_items.append(
                (subject_key, str(edf_path), str(ann_path), float(ahi), str(dst_dir), target_fs)
            )

    if n_missing_ahi:
        logger.warning(
            "%d subject(s) had no AHI in CSV and were skipped. "
            "Run with --log_level DEBUG to see individual subjects.",
            n_missing_ahi,
        )

    already_done = [item for item in work_items if (dst_dir / f"{item[0]}.npz").exists()]
    work_items   = [item for item in work_items if not (dst_dir / f"{item[0]}.npz").exists()]
    if already_done:
        logger.info("Skipping %d already-extracted subject(s) (npz exists).", len(already_done))
    logger.info("Subjects to extract: %d", len(work_items))

    n_workers = workers if workers is not None else max(1, cpu_count() // 2)
    logger.info("Using %d worker processes.", n_workers)

    results: Dict[str, str] = {}
    try:
        with Pool(n_workers) as pool:
            iterator = pool.imap_unordered(_extract_one, work_items)
            if tqdm:
                iterator = tqdm(iterator, total=len(work_items), desc="SHHS extract")
            for subject_key, outcome in iterator:
                results[subject_key] = outcome
                if outcome == "ok":
                    logger.info("[%s] extracted", subject_key)
                elif outcome == "skip":
                    logger.debug("[%s] skipped (already exists)", subject_key)
    except KeyboardInterrupt:
        ok = sum(1 for v in results.values() if v == "ok")
        skipped = sum(1 for v in results.values() if v == "skip")
        print(
            f"\nInterrupted after {ok} extracted, {skipped} skipped. Re-run to resume.",
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
        description="Stage 1: extract SHHS ECG + annotations from EDFs to compressed NPZ files."
    )
    parser.add_argument("--src_dir", required=True,
                        help="Root SHHS directory (parent of polysomnography/).")
    parser.add_argument("--dst_dir", required=True,
                        help="Output directory for extracted .npz files.")
    parser.add_argument("--harmonized_csv", required=True,
                        help="Path to shhs-harmonized-dataset-0.21.0.csv.")
    parser.add_argument("--target_fs", type=int, default=None,
                        help="Resample to this rate before saving (e.g. 100). Default: native rate.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: cpu_count // 2).")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    results = extract_shhs(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        harmonized_csv=args.harmonized_csv,
        target_fs=args.target_fs,
        workers=args.workers,
    )
    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v == "skip")
    warned = sum(1 for v in results.values() if v == "warn")
    print(f"\nDone. ok={ok}, skipped={skipped}, warnings={warned}")
