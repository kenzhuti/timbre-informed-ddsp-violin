#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
downsample_violinists.py

Create a down-sampled copy of a violin-recording corpus while
 • keeping the original directory structure,
 • preserving the per-violinist proportions as closely as possible,
 • deleting IDs whose scaled quota would be zero,
 • selecting the files to keep at random.

Example
-------
python downsample_violinists.py \
       --ori_audio  /data/violin_corpus \
       --dst_audio  /data/violin_corpus_25to1 \
       --ratio      25 \
       --seed       42
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def compute_target_counts(id_counts: dict[str, int], ratio: float) -> dict[str, int]:
    """
    Determine how many recordings to keep for each violinist ID.

    Strategy
    --------
    1. baseline_i = floor(count_i * ratio)
    2. total_target = round(sum(count_i) * ratio)
    3. Distribute the remaining quota (step-2 – step-1 sum) by the
       fractional remainders (count_i * ratio – baseline_i), breaking ties
       with the larger original counts.

    IDs whose baseline is 0 *and* do not obtain an extra slot are dropped.
    """
    baseline = {k: math.floor(v * ratio) for k, v in id_counts.items()}
    total_target = int(round(sum(id_counts.values()) * ratio))

    remainders = {
        k: (id_counts[k] * ratio) - baseline[k]
        for k in id_counts
    }
    sorted_ids = sorted(
        id_counts,
        key=lambda k: (remainders[k], id_counts[k]),
        reverse=True,
    )

    target = baseline.copy()
    extra_needed = total_target - sum(baseline.values())
    for k in sorted_ids:
        if extra_needed == 0:
            break
        target[k] += 1
        extra_needed -= 1

    # Drop violinists with zero quota
    target = {k: v for k, v in target.items() if v > 0}
    return target


# def downsample_audio(
#     ori_audio: str | os.PathLike,
#     dst_audio: str | os.PathLike,
#     ratio: int = 1/25,
#     seed: int | None = None,
#     suffixes: tuple[str, ...] = (".mp3", ".wav", ".flac", ".npz"),
# ) -> None:
#     """
#     Copy a proportionally down-sampled subset of the corpus to *dst_audio*.
#
#     Parameters
#     ----------
#     ori_audio : str | Path
#         Root directory of the original audio corpus.
#     dst_audio : str | Path
#         Destination root (will be created if it does not exist).
#     ratio : int, default=25
#         Keep approximately 1 / *ratio* of the recordings.
#     seed : int | None
#         Random seed for reproducible sampling.
#     suffixes : tuple[str], default=(' .mp3', '.wav', '.flac')
#         File extensions regarded as audio files.
#     """
#     rng = random.Random(seed)
#     ori_audio, dst_audio = Path(ori_audio), Path(dst_audio)
#
#     # ------------------------------------------------------------------ #
#     # 1.  Enumerate all audio files and group by first-two-character ID   #
#     # ------------------------------------------------------------------ #
#     files_by_id: dict[str, list[Path]] = defaultdict(list)
#     for f in ori_audio.rglob("*"):
#         if f.suffix.lower() in suffixes and f.is_file():
#             violinist_id = f.name[:2]
#             files_by_id[violinist_id].append(f)
#
#     id_counts = {k: len(v) for k, v in files_by_id.items()}
#     if not id_counts:
#         raise RuntimeError("No audio files found under the given root.")
#
#     # ------------------------------------------------------- #
#     # 2.  Decide how many recordings to keep for each ID      #
#     # ------------------------------------------------------- #
#     target_counts = compute_target_counts(id_counts, ratio)
#
#     # ------------------------------------------------------- #
#     # 3.  Randomly sample and copy while keeping structure    #
#     # ------------------------------------------------------- #
#     dst_audio.mkdir(parents=True, exist_ok=True)
#     kept_total = 0
#     chosen_files = []  # List to store chosen file names
#
#     for vid, quota in target_counts.items():
#         chosen = rng.sample(files_by_id[vid], quota)
#         kept_total += quota
#
#         for src in chosen:
#             rel_path = src.relative_to(ori_audio)
#             dst_path = dst_audio / rel_path
#             dst_path.parent.mkdir(parents=True, exist_ok=True)
#             shutil.copy2(src, dst_path)
#             chosen_files.append(str(rel_path))  # Save relative path of chosen file
#
#     # ---------------------------- #
#     # 4.  Save chosen file names   #
#     # ---------------------------- #
#     txt_file_path = dst_audio / "chosen_files.txt"
#     with open(txt_file_path, "w") as txt_file:
#         txt_file.write("\n".join(chosen_files))
#
#     # ---------------------------- #
#     # 5.  Report a short summary   #
#     # ---------------------------- #
#     print(f"Done. Copied {kept_total} recordings from {len(target_counts)} IDs "
#           f"into “{dst_audio}”.  Ratio ≈ 1:{ratio}.")
#     print(f"Saved chosen file names to {txt_file_path}.")


# ---------------------------------------------------------------------- #
#  CLI                                                                   #
# ---------------------------------------------------------------------- #

def downsample_audio(
    ori_audio: str | os.PathLike,
    dst_audio: str | os.PathLike,
    ratio: float = 1/25,
    seed: int | None = None,
    suffixes: tuple[str, ...] = (".mp3", ".wav", ".flac", ".npz"),
) -> None:
    """
    Copy a proportionally down-sampled subset of the corpus to *dst_audio*.

    Parameters
    ----------
    ori_audio : str | Path
        Root directory of the original audio corpus.
    dst_audio : str | Path
        Destination root (will be created if it does not exist).
    ratio : float, default=1/25
        Proportion of recordings to keep (e.g., 1/25 for down-sampling to 1/25).
    seed : int | None
        Random seed for reproducible sampling.
    suffixes : tuple[str], default=(' .mp3', '.wav', '.flac')
        File extensions regarded as audio files.
    """
    rng = random.Random(seed)
    ori_audio, dst_audio = Path(ori_audio), Path(dst_audio)

    # ------------------------------------------------------------------ #
    # 1.  Enumerate all audio files and group by first-two-character ID   #
    # ------------------------------------------------------------------ #
    files_by_id: dict[str, list[Path]] = defaultdict(list)
    for f in ori_audio.rglob("*"):
        if f.suffix.lower() in suffixes and f.is_file():
            violinist_id = f.name[:2]
            files_by_id[violinist_id].append(f)

    id_counts = {k: len(v) for k, v in files_by_id.items()}
    if not id_counts:
        raise RuntimeError("No audio files found under the given root.")

    # ------------------------------------------------------- #
    # 2.  Decide how many recordings to keep for each ID      #
    # ------------------------------------------------------- #
    target_counts = compute_target_counts(id_counts, ratio)

    # ------------------------------------------------------- #
    # 3.  Randomly sample and copy while keeping structure    #
    # ------------------------------------------------------- #
    dst_audio.mkdir(parents=True, exist_ok=True)
    kept_total = 0
    chosen_files = []  # List to store chosen file names

    for vid, quota in target_counts.items():
        chosen = rng.sample(files_by_id[vid], quota)
        kept_total += quota

        for src in chosen:
            rel_path = src.relative_to(ori_audio)
            dst_path = dst_audio / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            chosen_files.append(str(rel_path))  # Save relative path of chosen file

    # ---------------------------- #
    # 4.  Save chosen file names   #
    # ---------------------------- #
    txt_file_path = dst_audio / "chosen_files.txt"
    with open(txt_file_path, "w") as txt_file:
        txt_file.write("\n".join(chosen_files))

    # ---------------------------- #
    # 5.  Report a short summary   #
    # ---------------------------- #
    print(f"Done. Copied {kept_total} recordings from {len(target_counts)} IDs "
          f"into “{dst_audio}”.  Ratio ≈ {ratio}.")
    print(f"Saved chosen file names to {txt_file_path}.")


def copy_difference_set(
    ori_audio: str | os.PathLike,
    train_audio: str | os.PathLike,
    test_audio: str | os.PathLike,
    suffixes: tuple[str, ...] = (".mp3", ".wav", ".flac", ".npz"),
) -> None:
    """
    Compute the difference set of audio files between `ori_audio` and `train_audio`,
    and copy the resulting files into `test_audio`.

    Parameters
    ----------
    ori_audio : str | Path
        Root directory of the original audio corpus.
    train_audio : str | Path
        Directory containing training audio files.
    test_audio : str | Path
        Destination directory for the difference set.
    suffixes : tuple[str], default=(".mp3", ".wav", ".flac", ".npz")
        File extensions regarded as audio files.
    """
    ori_audio, train_audio, test_audio = Path(ori_audio), Path(train_audio), Path(test_audio)

    # Collect all valid files in ori_audio and train_audio
    ori_files = {f.relative_to(ori_audio) for f in ori_audio.rglob("*") if f.suffix.lower() in suffixes and f.is_file()}
    train_files = {f.relative_to(train_audio) for f in train_audio.rglob("*") if f.suffix.lower() in suffixes and f.is_file()}

    # Compute the difference set
    difference_files = ori_files - train_files

    # Copy files in the difference set to test_audio
    test_audio.mkdir(parents=True, exist_ok=True)
    for rel_path in difference_files:
        src_path = ori_audio / rel_path
        dst_path = test_audio / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(difference_files)} files to `{test_audio}`.")


if __name__ == "__main__":
    # python downsample_violinists.py \
    #        --ori_audio  /path/to/ori_audio \
    #        --dst_audio  /path/to/output_subset \
    #        --ratio      25        # 25 : 1 缩放
    #        --seed       42        # 可选，保证可复现
    parser = argparse.ArgumentParser(description="Down-sample violin corpus.")
    parser.add_argument("--ori_audio", required=False, help="Original root dir")
    parser.add_argument("--dst_audio", required=False, help="Output root dir")
    parser.add_argument("--ratio", type=int, default=25, help="Down-sampling ratio (default 25)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # args.ori_audio = '../../ViolinEtudes-zenodo_2025/original'
    # args.dst_audio = '../../ViolinEtudes-zenodo_2025/original_downsample_25'
    # args.ratio = 25  # ViolinEtudes size: 925
    # args.seed = 42  # original seed: 42

    args.ori_audio = '../../ViolinEtudes-zenodo-downsample_2025/anal/crepe'
    args.dst_audio = '../../ViolinEtudes-zenodo-downsample_2025/anal/train'
    args.ratio = 0.8 # ViolinEtudes Downsample anal size: 37, 37 * 0.8 ≈ 30
    args.seed = 42 # original seed: 42

    # downsample_audio(
    #     ori_audio=args.ori_audio,
    #     dst_audio=args.dst_audio,
    #     ratio=args.ratio,
    #     seed=args.seed,
    # )

    copy_difference_set(
        ori_audio='../../ViolinEtudes-zenodo-downsample_2025/anal/crepe',
        train_audio='../../ViolinEtudes-zenodo-downsample_2025/anal/train',
        test_audio='../../ViolinEtudes-zenodo-downsample_2025/anal/test',
        suffixes=(".mp3", ".wav", ".flac", ".npz"),
    )
