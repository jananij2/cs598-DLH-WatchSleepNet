"""Offline preprocessing script for the Multi-Ethnic Study of Atherosclerosis (MESA).

Processes the MESA Sleep ancillary study end-to-end: reads each subject's EDF
file, extracts the finger-tip PPG (Pleth) channel, detects peaks via
neurokit2, computes a per-sample IBI time series, downsamples to 25 Hz, and
writes a per-subject ``.npz`` cache file compatible with the WatchSleepNet
``SSDataset`` loader.

Reference implementation:
    https://github.com/willKeWang/WatchSleepNet_public/tree/main/dataset_preparation/public_datasets

Usage::

    python preprocess_mesa.py \\
        --src_dir /Volumes/MastersData/mesa \\
        --dst_dir /Volumes/MastersData/SHHS_MESA_IBI \\
        --harmonized_csv /Volumes/MastersData/mesa/datasets/mesa-sleep-harmonized-dataset-0.8.0.csv

Where ``src_dir`` contains::

    polysomnography/edfs/mesa-sleep-XXXXX.edf
    polysomnography/annotations-events-profusion/mesa-sleep-XXXXX-profusion.xml

Output NPZ keys: data (IBI float64 at 25 Hz), stages (float64), fs (int), ahi (float)

Output is written to the same directory as SHHS IBI output so both datasets
can be loaded together by the model's ``shhs_mesa_ibi`` dataset config.
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
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

__all__ = [
    "read_mesa_subject",
    "preprocess_mesa",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PPG_CHANNEL: str = "Pleth"
_EXCLUDE_CHANNELS: List[str] = ["EEG1", "EEG2", "Snore", "Thor", "Abdo", "Leg", "Therm", "Pos"]

_TARGET_FS: int = 25  # Hz — output sampling rate
_NATIVE_FS: int = 256  # Hz — MESA finger PPG sampling rate
_IBI_OUTLIER_THRESHOLD: float = 2.0  # seconds; HR < 30 BPM treated as artifact
_MAX_EPOCHS: int = 1100  # ~9.17 hours; matches WatchSleepNet truncation

# Harmonized CSV columns
_ID_COL: str = "nsrrid"
_AHI_COL: str = "nsrr_ahi_hp3u"


# ---------------------------------------------------------------------------
# Signal reading
# ---------------------------------------------------------------------------


def _read_edf_ppg(
    edf_path: Path, ann_path: Path
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Read PPG signal and sleep stage labels from one MESA EDF + XML pair.

    Args:
        edf_path: Path to the ``mesa-sleep-XXXXX.edf`` file.
        ann_path: Path to the paired ``mesa-sleep-XXXXX-profusion.xml`` file.

    Returns:
        A tuple ``(signal, fs, stages)`` where:

        - ``signal``: float64 ndarray of shape ``(n_samples,)`` containing
          the raw finger PPG at its native sampling rate (256 Hz).
        - ``fs``: native sampling frequency in Hz.
        - ``stages``: int8 ndarray of shape ``(n_samples,)`` with per-sample
          sleep stage labels expanded from 30-second epochs.
          Encoding: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
    """
    raw = mne.io.read_raw_edf(
        str(edf_path), verbose=0, exclude=_EXCLUDE_CHANNELS, infer_types=False
    )
    raw.pick(picks=[_PPG_CHANNEL])
    fs = raw.info["sfreq"]
    signal = raw.get_data().flatten()

    tree = ET.parse(str(ann_path))
    root = tree.getroot()
    stages = np.array(
        [s.text for s in root[4].findall("SleepStage")], dtype=np.int8
    )

    # Merge N3/N4: stage 4 → 3; shift REM: stage 5 → 4
    stages[stages == 4] = 3
    stages[stages == 5] = 4

    # Expand from per-epoch to per-sample
    samples_per_epoch = int(30 * fs)
    stages = np.repeat(stages, samples_per_epoch)

    # Truncate to max epochs
    max_samples = _MAX_EPOCHS * samples_per_epoch
    signal = signal[:max_samples]
    stages = stages[:max_samples]

    # Align lengths
    min_len = min(len(signal), len(stages))
    return signal[:min_len], fs, stages[:min_len]


# ---------------------------------------------------------------------------
# IBI computation
# ---------------------------------------------------------------------------


def _compute_ibi_ppg(signal: np.ndarray, fs: float) -> np.ndarray:
    """Compute per-sample IBI from PPG via neurokit2 peak detection.

    Args:
        signal: 1-D float64 PPG (Pleth) array at ``fs`` Hz.
        fs: Sampling frequency in Hz.

    Returns:
        float64 ndarray of shape ``(len(signal),)`` with per-sample IBI in
        seconds. Intervals ≥ :data:`_IBI_OUTLIER_THRESHOLD` are zeroed out.
    """
    signals, info = nk.ppg_process(signal, sampling_rate=fs)
    peaks = info["PPG_Peaks"]
    ibi_values = np.diff(peaks) / fs

    ibi = np.zeros(signal.shape, dtype=np.float64)
    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]
    ibi[ibi >= _IBI_OUTLIER_THRESHOLD] = 0.0
    return ibi


# ---------------------------------------------------------------------------
# Per-subject pipeline
# ---------------------------------------------------------------------------


def read_mesa_subject(
    edf_path: Path, ann_path: Path, ahi: float
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """End-to-end pipeline for a single MESA subject.

    Reads PPG from EDF, extracts IBI, downsamples to 25 Hz, and returns
    arrays ready to be written to an NPZ cache file.

    Note:
        MESA's native PPG rate of 256 Hz is not an integer multiple of 25 Hz,
        so polyphase resampling is used rather than simple integer slicing.

    Args:
        edf_path: Path to the subject's ``mesa-sleep-XXXXX.edf`` file.
        ann_path: Path to the subject's ``-profusion.xml`` annotation file.
        ahi: Apnea-Hypopnea Index for this subject (from harmonized CSV).

    Returns:
        A tuple ``(ibi_ds, fs_out, stages_ds, ahi)`` where:

        - ``ibi_ds``: float64 IBI array at :data:`_TARGET_FS` Hz.
        - ``fs_out``: :data:`_TARGET_FS` (25).
        - ``stages_ds``: float64 stage array at :data:`_TARGET_FS` Hz.
        - ``ahi``: the passed-through AHI value.
    """
    signal, fs, stages = _read_edf_ppg(edf_path, ann_path)
    ibi = _compute_ibi_ppg(signal, fs)

    original_fs = int(fs)
    if original_fs % _TARGET_FS == 0:
        factor = original_fs // _TARGET_FS
        ibi_ds = ibi[::factor]
        stages_ds = stages[::factor].astype(np.float64)
    else:
        # 256 Hz → 25 Hz requires polyphase resampling
        ibi_ds = resample_poly(ibi, _TARGET_FS, original_fs)
        stages_ds = resample_poly(stages.astype(np.float64), _TARGET_FS, original_fs)

    return ibi_ds, _TARGET_FS, stages_ds, ahi


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def _process_one(args: Tuple) -> Tuple[str, str]:
    """Worker function executed in a subprocess.

    Args:
        args: Tuple of ``(subject_key, edf_path, ann_path, ahi, dst_dir)``.

    Returns:
        ``(subject_key, outcome)`` where outcome is ``"ok"`` or ``"warn"``.
    """
    subject_key, edf_path, ann_path, ahi, dst_dir = args
    out_path = Path(dst_dir) / f"{subject_key}.npz"

    if out_path.exists():
        return subject_key, "skip"

    try:
        ibi_ds, fs_out, stages_ds, ahi_val = read_mesa_subject(
            Path(edf_path), Path(ann_path), ahi
        )
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
# Top-level preprocessing function
# ---------------------------------------------------------------------------


def preprocess_mesa(
    src_dir: "str | Path",
    dst_dir: "str | Path",
    harmonized_csv: "str | Path",
    workers: Optional[int] = None,
) -> Dict[str, str]:
    """Preprocess all MESA subjects and write per-subject ``.npz`` cache files.

    All MESA EDF and annotation files are expected in flat directories (no
    per-visit subdirectories). Subjects whose raw EDF files cannot be processed
    by neurokit2 are skipped with a warning (matches the 2,055-subject subset
    reported in the paper). Already-existing output files are skipped
    (idempotent re-runs).

    Args:
        src_dir: Root MESA directory containing ``polysomnography/``.
        dst_dir: Output directory for ``.npz`` files.  Created if absent.
        harmonized_csv: Path to the MESA harmonized CSV (e.g. ``mesa-sleep-harmonized-dataset-0.8.0.csv``).
        workers: Number of parallel worker processes.  Defaults to
            ``cpu_count() // 2``.

    Returns:
        A dict mapping ``subject_key → outcome`` where outcome is one of:

        - ``"ok"``: subject processed and ``.npz`` written.
        - ``"skip"``: ``.npz`` already existed; subject skipped.
        - ``"warn"``: subject failed; warning emitted.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    harmonized_csv = Path(harmonized_csv)
    if not harmonized_csv.exists():
        # Auto-discover any version of the harmonized CSV in the same directory
        candidates = sorted(harmonized_csv.parent.glob("mesa-sleep-harmonized-dataset-*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"Harmonized CSV not found: {harmonized_csv}\n"
                "Download it from NSRR: nsrr download mesa/datasets --file=\"mesa-sleep-harmonized-dataset-*.csv\""
            )
        harmonized_csv = candidates[-1]  # use latest version
        logger.warning("Specified CSV not found; using auto-discovered: %s", harmonized_csv)

    logger.info("Loading harmonized CSV: %s", harmonized_csv)
    info_df = pd.read_csv(harmonized_csv, dtype={_ID_COL: str})
    info_df[_ID_COL] = info_df[_ID_COL].str.lstrip("0")
    ahi_lookup: Dict[str, float] = dict(
        zip(info_df[_ID_COL], info_df[_AHI_COL].astype(float))
    )

    edf_dir = src_dir / "polysomnography" / "edfs"
    ann_dir = src_dir / "polysomnography" / "annotations-events-profusion"

    if not edf_dir.exists():
        raise FileNotFoundError(f"EDF directory not found: {edf_dir}")

    work_items: List[Tuple] = []
    n_missing_ahi: int = 0
    for edf_path in sorted(edf_dir.glob("*.edf")):
        # "mesa-sleep-1234.edf" → sid = "1234"
        sid = edf_path.stem.split("-")[-1].lstrip("0")
        subject_key = f"mesa-{sid}"

        ann_path = ann_dir / f"{edf_path.stem}-profusion.xml"
        if not ann_path.exists():
            logger.debug("Missing annotation for %s, skipping.", subject_key)
            continue

        ahi = ahi_lookup.get(sid, None)
        if ahi is None or (isinstance(ahi, float) and np.isnan(ahi)):
            logger.debug("[%s] AHI not found in CSV; skipping.", subject_key)
            n_missing_ahi += 1
            continue
        ahi = float(ahi)

        work_items.append(
            (subject_key, str(edf_path), str(ann_path), ahi, str(dst_dir))
        )

    if n_missing_ahi:
        logger.warning(
            "%d subject(s) had no AHI in CSV and were skipped (matches original pipeline). "
            "Run with --log_level DEBUG to see individual subjects.",
            n_missing_ahi,
        )

    # Pre-filter already-processed subjects so the pool only receives new work
    # and the progress bar reflects remaining work accurately.
    already_done = [item for item in work_items if (dst_dir / f"{item[0]}.npz").exists()]
    work_items   = [item for item in work_items if not (dst_dir / f"{item[0]}.npz").exists()]
    if already_done:
        logger.info(
            "Skipping %d already-processed subject(s) (npz exists).", len(already_done)
        )
    logger.info("Total subjects to process: %d", len(work_items))

    n_workers = workers if workers is not None else max(1, cpu_count() // 2)
    logger.info("Using %d worker processes.", n_workers)

    results: Dict[str, str] = {}
    try:
        with Pool(n_workers) as pool:
            iterator = pool.imap_unordered(_process_one, work_items)
            if tqdm:
                iterator = tqdm(iterator, total=len(work_items), desc="MESA")
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
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MESA dataset to per-subject IBI NPZ cache files."
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Root MESA directory (parent of polysomnography/).",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        help="Output directory for .npz files.",
    )
    parser.add_argument(
        "--harmonized_csv",
        required=True,
        help="Path to mesa-sleep-harmonized-dataset-*.csv (e.g. 0.8.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: cpu_count // 2).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    results = preprocess_mesa(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        harmonized_csv=args.harmonized_csv,
        workers=args.workers,
    )
    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v == "skip")
    warned = sum(1 for v in results.values() if v == "warn")
    print(f"\nDone. ok={ok}, skipped={skipped}, warnings={warned}")
