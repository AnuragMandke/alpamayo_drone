"""
scripts/download_uzh_fpv.py — Download the UZH-FPV drone racing dataset
(snapdragon sequences with ground truth) and extract them for conversion
by scripts/convert_uzh_fpv.py.

Downloads ~12GB of zips; sequences already present under --out are skipped,
and zips are deleted after extraction unless --keep_zips is given.

Usage:
    python scripts/download_uzh_fpv.py --out data/raw/uzh_fpv
"""

import argparse
import sys
import time
import zipfile
import urllib.error
import urllib.request
from pathlib import Path

MAX_RETRIES = 4          # per sequence, for transient network errors
RETRY_BACKOFF_S = 5      # multiplied by attempt number

BASE_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3"

# All snapdragon sequences with ground truth (fpv.ifi.uzh.ch/datasets)
SEQUENCES = [
    "indoor_forward_3_snapdragon_with_gt",
    "indoor_forward_5_snapdragon_with_gt",
    "indoor_forward_6_snapdragon_with_gt",
    "indoor_forward_7_snapdragon_with_gt",
    "indoor_forward_9_snapdragon_with_gt",
    "indoor_forward_10_snapdragon_with_gt",
    "indoor_45_2_snapdragon_with_gt",
    "indoor_45_4_snapdragon_with_gt",
    "indoor_45_9_snapdragon_with_gt",
    "indoor_45_12_snapdragon_with_gt",
    "indoor_45_13_snapdragon_with_gt",
    "indoor_45_14_snapdragon_with_gt",
    "outdoor_forward_1_snapdragon_with_gt",
    "outdoor_forward_3_snapdragon_with_gt",
    "outdoor_forward_5_snapdragon_with_gt",
    "outdoor_45_1_snapdragon_with_gt",
]


def sequence_complete(seq_dir: Path) -> bool:
    return (seq_dir / "groundtruth.txt").exists() and (seq_dir / "img").is_dir()


def download(url: str, dest: Path):
    """
    Stream-download with coarse progress (one line every ~100MB).

    Writes to a .part file and only renames to `dest` after verifying the
    full Content-Length was received, so a dropped connection cannot leave a
    truncated file that looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        next_report = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    pct = 100 * done / total if total else 0
                    print(f"    {done / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)",
                          flush=True)
                    next_report += 100 << 20

    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise IOError(
            f"truncated download: got {done} of {total} bytes "
            "(connection dropped)"
        )
    tmp.rename(dest)


def valid_zip(path: Path) -> bool:
    """True if `path` exists and is a structurally valid (non-truncated) zip."""
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def extract(zip_path: Path, seq: str, out_root: Path):
    """Extract into out_root/<seq>/ regardless of the zip's internal layout."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # If every member already lives under "<seq>/", extract to the root;
        # otherwise extract into the sequence directory.
        if all(n.startswith(seq + "/") for n in names if n.strip("/")):
            zf.extractall(out_root)
        else:
            zf.extractall(out_root / seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/uzh_fpv")
    p.add_argument("--keep_zips", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    failures = []

    for i, seq in enumerate(SEQUENCES, 1):
        seq_dir = out_root / seq
        if sequence_complete(seq_dir):
            print(f"[{i:2d}/{len(SEQUENCES)}] {seq}: already present, skipping",
                  flush=True)
            continue

        url = f"{BASE_URL}/{seq}.zip"
        zip_path = out_root / "zips" / f"{seq}.zip"
        print(f"[{i:2d}/{len(SEQUENCES)}] {seq}", flush=True)
        try:
            # Re-download unless a fully valid zip is already cached. A
            # leftover truncated zip from an interrupted run is discarded.
            if zip_path.exists() and not valid_zip(zip_path):
                print("    cached zip is corrupt/truncated — re-downloading",
                      flush=True)
                zip_path.unlink()

            for attempt in range(1, MAX_RETRIES + 1):
                if valid_zip(zip_path):
                    break
                try:
                    print(f"    downloading (attempt {attempt}/{MAX_RETRIES})",
                          flush=True)
                    download(url, zip_path)
                    break
                except (urllib.error.URLError, IOError) as e:
                    print(f"    transient error: {e}", flush=True)
                    if attempt == MAX_RETRIES:
                        raise
                    time.sleep(RETRY_BACKOFF_S * attempt)

            print(f"    extracting -> {seq_dir}", flush=True)
            extract(zip_path, seq, out_root)
            if not sequence_complete(seq_dir):
                raise RuntimeError("extraction finished but sequence is incomplete")
            if not args.keep_zips:
                zip_path.unlink()
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            failures.append(seq)

    print(f"\nDone. {len(SEQUENCES) - len(failures)}/{len(SEQUENCES)} sequences ready "
          f"under {out_root}")
    if failures:
        print("Failed sequences (rerun to retry):")
        for seq in failures:
            print(f"  {seq}")
        sys.exit(1)


if __name__ == "__main__":
    main()
