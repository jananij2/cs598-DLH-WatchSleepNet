"""Offline preprocessing script for the DREAMT v2.1.0 wearable sleep dataset.

Run this script once to convert raw PSG-aligned CSV files into per-subject ``.npz``
cache files that can be loaded by :class:`pyhealth.datasets.DREAMTDataset`.

Reference implementation:
    https://github.com/willKeWang/WatchSleepNet_public/tree/main/dataset_preparation/dreamt

Usage::

    python preprocess_dreamt.py \\
        --src_dir /path/to/dreamt_2.1.0/data_100Hz \\
        --dst_dir /path/to/dreamt_npz \\
        --participant_info /path/to/dreamt_2.1.0/participant_info.csv

The produced ``.npz`` files are compatible with both
:class:`pyhealth.datasets.DREAMTDataset` and the original WatchSleepNet
``SSDataset`` loader.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "calculate_ibi_segment",
    "read_dreamt_data",
    "preprocess_dreamt",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage label constants
# ---------------------------------------------------------------------------
_STAGE_STR_TO_INT: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "Missing": -1,
}

_MISSING_LABEL: int = -1
_WAKE_LABEL: int = 0
_NON_WAKE_LABELS: Tuple[int, ...] = (1, 2, 3, 4)

# Physiology: IBI >= 2.0 s → heart rate < 30 BPM → treat as artifact
_IBI_OUTLIER_THRESHOLD: float = 2.0

# Output sampling rate after stride-4 downsampling from 100 Hz
_OUTPUT_FS: int = 25
_EPOCH_DURATION_S: int = 30
_SAMPLES_PER_EPOCH: int = _OUTPUT_FS * _EPOCH_DURATION_S  # 750


# ---------------------------------------------------------------------------
# Core IBI extraction helper
# ---------------------------------------------------------------------------


def calculate_ibi_segment(
    ppg_signal: np.ndarray,
    fs: int,
    use_package: str = "neurokit",
) -> Tuple[np.ndarray, int]:
    """Compute a per-sample IBI time series from a PPG epoch.

    Detects systolic peaks in ``ppg_signal`` via NeuroKit2's ``ppg_process``,
    computes the Inter-Beat Interval (IBI) in seconds between consecutive peaks,
    and fills a per-sample IBI array at the original sampling rate ``fs``.
    Values at or above :data:`_IBI_OUTLIER_THRESHOLD` (2.0 s) are replaced with
    zero.  Epochs where fewer than 2 peaks are detected are returned as all-zero
    arrays.

    Args:
        ppg_signal: 1-D float array of raw PPG (BVP) samples at ``fs`` Hz.
        fs: Sampling rate of ``ppg_signal`` in Hz (typically 100).
        use_package: Peak-detection backend.  Only ``"neurokit"`` is currently
            supported.

    Returns:
        A tuple ``(ibi, empty_segment_count)`` where:

        - ``ibi``: float32 ndarray of shape ``(len(ppg_signal),)`` containing
          the per-sample IBI series.
        - ``empty_segment_count``: 1 if this epoch produced no usable IBI
          (too few peaks), 0 otherwise.

    Raises:
        ValueError: If ``use_package`` is not ``"neurokit"``.
    """
    if use_package != "neurokit":
        raise ValueError(
            f"Unsupported peak-detection backend: '{use_package}'. "
            "Only 'neurokit' is supported."
        )

    import neurokit2 as nk

    empty_segment_count = 0
    ibi = np.zeros(len(ppg_signal), dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _signals, info = nk.ppg_process(ppg_signal, sampling_rate=fs)
        except Exception:
            return ibi, 1

        for w in caught:
            if "Too few peaks detected" in str(w.message):
                return ibi, 1

    peaks = info.get("PPG_Peaks", np.array([], dtype=int))

    if len(peaks) < 2:
        return ibi, 1

    ibi_values = np.diff(peaks) / float(fs)

    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]

    ibi[ibi >= _IBI_OUTLIER_THRESHOLD] = 0.0

    return ibi, empty_segment_count


# ---------------------------------------------------------------------------
# Per-subject data reader and epoch segmenter
# ---------------------------------------------------------------------------


def read_dreamt_data(
    file_path: "str | Path",
    select_chs: list = None,
    fixed_fs: int = 100,
    epoch_duration: int = 30,
    use_package: str = "neurokit",
    downsample: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Load one DREAMT PSG CSV, extract IBI epochs, and return aligned arrays.

    Steps performed (matches WatchSleepNet ``dreamt_extract_ibi_se.py``):

    1. Read ``BVP`` and ``Sleep_Stage`` columns from ``file_path``.
    2. Map stage strings to integers; exclude ``"Missing"`` samples.
    3. Trim the recording to span the first–last non-wake sample.
    4. Segment into non-overlapping 30-second epochs; discard transition epochs
       (mixed stage labels within one epoch).
    5. For each clean epoch, call :func:`calculate_ibi_segment`.
    6. Optionally downsample from ``fixed_fs`` to 25 Hz (stride 4).

    Args:
        file_path: Path to a ``*_PSG_df_updated.csv`` file from DREAMT
            ``data_100Hz/``.
        select_chs: Column names to use as the PPG signal.  Defaults to
            ``["BVP"]``.
        fixed_fs: Sampling rate of the input CSV in Hz.  Defaults to 100.
        epoch_duration: Epoch length in seconds.  Defaults to 30.
        use_package: Peak-detection backend forwarded to
            :func:`calculate_ibi_segment`.
        downsample: If ``True``, apply stride-4 subsampling to reduce the
            output from ``fixed_fs`` to 25 Hz.

    Returns:
        A tuple ``(ibi_1d, fs_out, stages_1d)`` where:

        - ``ibi_1d``: float32 ndarray of shape ``(N,)`` with the concatenated
          IBI series.  ``N`` is always a multiple of
          ``(fs_out * epoch_duration)``.
        - ``fs_out``: Output sampling rate (25 if ``downsample=True``, else
          ``fixed_fs``).
        - ``stages_1d``: int32 ndarray of shape ``(N,)`` with one label per
          sample (same label repeated across each 30-second epoch).

    Raises:
        ValueError: If no non-wake epochs are found in the recording.
    """
    if select_chs is None:
        select_chs = ["BVP"]

    data = pd.read_csv(file_path)

    ppg_signal = data[select_chs[0]].to_numpy(dtype=np.float32)
    stage_strings = data["Sleep_Stage"].to_numpy()

    mapped_stages = np.array(
        [_STAGE_STR_TO_INT.get(s, _MISSING_LABEL) for s in stage_strings],
        dtype=np.int32,
    )

    # Remove Missing samples
    valid_mask = mapped_stages != _MISSING_LABEL
    ppg_signal = ppg_signal[valid_mask]
    mapped_stages = mapped_stages[valid_mask]

    # Trim to span of non-wake epochs
    samples_per_epoch_in = fixed_fs * epoch_duration
    non_wake_indices = np.where(np.isin(mapped_stages, list(_NON_WAKE_LABELS)))[0]

    if len(non_wake_indices) == 0:
        raise ValueError("No non-wake epochs found in recording.")

    start_epoch = non_wake_indices[0] // samples_per_epoch_in
    start_idx = max(int(start_epoch), 0)
    end_idx = min(int(non_wake_indices[-1]), len(ppg_signal))

    if start_idx < end_idx:
        ppg_signal = ppg_signal[start_idx:end_idx]
        mapped_stages = mapped_stages[start_idx : end_idx + 1]

    # Segment and extract IBI per epoch
    valid_ibi_epochs: list = []
    valid_stage_epochs: list = []
    total_empty = 0
    i = 0

    while i + samples_per_epoch_in <= len(ppg_signal):
        epoch_ppg = ppg_signal[i : i + samples_per_epoch_in]
        epoch_stages = mapped_stages[i : i + samples_per_epoch_in]

        if np.all(epoch_stages == epoch_stages[0]):
            ibi_epoch, empty = calculate_ibi_segment(epoch_ppg, fixed_fs, use_package)
            total_empty += empty
            valid_ibi_epochs.append(ibi_epoch)
            valid_stage_epochs.append(
                np.full(samples_per_epoch_in, epoch_stages[0], dtype=np.int32)
            )
            i += samples_per_epoch_in
        else:
            i += 1  # slide by one sample to find next aligned boundary

    if not valid_ibi_epochs:
        raise ValueError("No valid single-stage epochs found after segmentation.")

    ibi_1d = np.concatenate(valid_ibi_epochs).astype(np.float32)
    stages_1d = np.concatenate(valid_stage_epochs).astype(np.int32)

    logger.debug("Total empty segments: %d", total_empty)

    if downsample:
        ibi_1d = ibi_1d[::4]
        stages_1d = stages_1d[::4]
        fs_out = _OUTPUT_FS
    else:
        fs_out = fixed_fs

    return ibi_1d, fs_out, stages_1d


# ---------------------------------------------------------------------------
# Top-level preprocessing function
# ---------------------------------------------------------------------------


def preprocess_dreamt(
    src_dir: "str | Path",
    dst_dir: "str | Path",
    participant_info_path: "str | Path",
) -> Dict[str, str]:
    """Preprocess all DREAMT subjects and write per-subject ``.npz`` cache files.

    This function is intended to be run **once** before training.  It reads
    ``*_PSG_df_updated.csv`` files from ``src_dir``, extracts IBI via
    :func:`read_dreamt_data`, joins AHI from ``participant_info_path``, and
    saves one ``{SID}.npz`` per subject to ``dst_dir``.

    Already-existing output files are skipped (idempotent re-runs).

    Args:
        src_dir: Directory containing ``*_PSG_df_updated.csv`` files (the
            DREAMT ``data_100Hz/`` folder).
        dst_dir: Output directory for ``.npz`` files.  Created if absent.
        participant_info_path: Path to ``participant_info.csv`` containing at
            least the columns ``SID`` and ``AHI``.

    Returns:
        A dict mapping ``SID → outcome`` where outcome is one of:

        - ``"ok"``: subject processed and ``.npz`` written.
        - ``"skip"``: ``.npz`` already existed; subject skipped.
        - ``"warn"``: subject failed or had no valid epochs; warning emitted.

    Note:
        NeuroKit2 must be installed for IBI extraction.  Real DREAMT CSV files
        are required; this function is **not** suitable for unit-testing (use
        synthetic fixtures instead, per FR-012).
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    participant_info_path = Path(participant_info_path)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load participant metadata
    if participant_info_path.exists():
        info_df = pd.read_csv(participant_info_path)
        ahi_lookup: Dict[str, float] = dict(
            zip(info_df["SID"].astype(str), info_df["AHI"].astype(float))
        )
    else:
        warnings.warn(
            f"participant_info.csv not found at {participant_info_path}. "
            "AHI values will be NaN for all subjects.",
            stacklevel=2,
        )
        ahi_lookup = {}

    csv_files = sorted(src_dir.glob("*_PSG_df_updated.csv"))
    if not csv_files:
        warnings.warn(f"No *_PSG_df_updated.csv files found in {src_dir}.", stacklevel=2)
        return {}

    results: Dict[str, str] = {}
    iterator = tqdm(csv_files, desc="Preprocessing DREAMT") if tqdm else csv_files

    # Track the file currently being written so KeyboardInterrupt can clean it up.
    _current_out_path: Optional[Path] = None

    try:
        for csv_path in iterator:
            sid = csv_path.name.split("_")[0]
            out_path = dst_dir / f"{sid}.npz"

            if out_path.exists():
                results[sid] = "skip"
                continue

            try:
                ibi_1d, fs_out, stages_1d = read_dreamt_data(
                    file_path=csv_path,
                    select_chs=["BVP"],
                    fixed_fs=100,
                    epoch_duration=30,
                    use_package="neurokit",
                    downsample=True,
                )
            except Exception as exc:
                warnings.warn(
                    f"[{sid}] IBI extraction failed: {exc}. Skipping subject.",
                    stacklevel=2,
                )
                results[sid] = "warn"
                continue

            if len(ibi_1d) < _SAMPLES_PER_EPOCH:
                warnings.warn(
                    f"[{sid}] No valid epochs after processing. Skipping subject.",
                    stacklevel=2,
                )
                results[sid] = "warn"
                continue

            ahi = float(ahi_lookup.get(sid, float("nan")))
            if np.isnan(ahi):
                warnings.warn(
                    f"[{sid}] AHI not found in participant_info.csv; storing NaN.",
                    stacklevel=2,
                )

            _current_out_path = out_path
            np.savez(
                out_path,
                data=ibi_1d.astype(np.float32),
                stages=stages_1d.astype(np.int32),
                fs=np.int64(fs_out),
                ahi=np.float32(ahi),
            )
            _current_out_path = None
            results[sid] = "ok"
            logger.info(
                "[%s] Saved %d epochs → %s",
                sid,
                len(ibi_1d) // _SAMPLES_PER_EPOCH,
                out_path,
            )

    except KeyboardInterrupt:
        if _current_out_path is not None and _current_out_path.exists():
            _current_out_path.unlink()
            print(
                f"\nInterrupted — deleted partial file: {_current_out_path}",
                flush=True,
            )
        ok = sum(1 for v in results.values() if v == "ok")
        skipped = sum(1 for v in results.values() if v == "skip")
        print(
            f"Interrupted after {ok} subject(s) completed, "
            f"{skipped} skipped. Re-run to resume.",
            flush=True,
        )
        raise

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess DREAMT dataset to per-subject NPZ cache files."
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Path to DREAMT data_100Hz/ directory.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        help="Output directory for .npz files.",
    )
    parser.add_argument(
        "--participant_info",
        required=True,
        help="Path to participant_info.csv.",
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
    results = preprocess_dreamt(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        participant_info_path=args.participant_info,
    )
    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v == "skip")
    warned = sum(1 for v in results.values() if v == "warn")
    print(f"\nDone. ok={ok}, skipped={skipped}, warnings={warned}")
