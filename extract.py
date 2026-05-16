import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple


SUPPORTED_VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def collect_videos(video_root: Path, output_root: Path, output_ext: str) -> List[Tuple[str, str, str, float, int]]:
    """Collect videos from video_root/<subdir>/*.avi and map them to output_root/<subdir>/."""
    jobs: List[Tuple[str, str, str, float, int]] = []

    subdirs = sorted([p for p in video_root.iterdir() if p.is_dir()])
    for subdir in subdirs:
        video_files = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        save_dir = output_root / subdir.name
        for video_path in video_files:
            jobs.append((str(video_path), str(save_dir), output_ext, 0.0, 0))

    return jobs


def extract_with_ffmpeg(
    video_path_str: str,
    save_dir_str: str,
    output_ext: str,
    interval_sec: float,
    jpg_quality: int,
) -> Tuple[str, int, str]:
    """Extract one frame every interval_sec seconds using ffmpeg."""
    video_path = Path(video_path_str)
    save_dir = Path(save_dir_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_ext = output_ext.lower().lstrip(".")
    output_pattern = save_dir / f"{video_path.stem}_%06d.{output_ext}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_sec}",
        "-start_number",
        "0",
    ]

    # Lower q:v means higher quality. 2 is high quality and usually much faster/smaller than PNG.
    if output_ext in {"jpg", "jpeg"}:
        cmd += ["-q:v", str(jpg_quality)]

    cmd.append(str(output_pattern))

    before = set(save_dir.glob(f"{video_path.stem}_*.{output_ext}"))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "unknown ffmpeg error").strip()
        return str(video_path), 0, f"[ERROR] {err}"

    after = set(save_dir.glob(f"{video_path.stem}_*.{output_ext}"))
    saved_count = len(after - before)

    # If files were overwritten, after-before may be 0. Count matching outputs as fallback.
    if saved_count == 0:
        saved_count = len(after)

    return str(video_path), saved_count, "[OK]"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast frame extraction using ffmpeg. Reads video/normal/<subdir>/*.avi "
            "and saves one frame every N seconds to data/posco/train/<subdir>/."
        )
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("video/normal"),
        help="Input root. Expected structure: video/normal/01/*.avi, video/normal/02/*.avi, ...",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/posco/train"),
        help="Output root. Frames are saved to data/posco/train/<subdir>/",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=10.0,
        help="Extract one frame every this many seconds. Default: 10",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel ffmpeg processes. Recommended: 2~4. Default: 4",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format. JPG is faster and smaller than PNG. Default: jpg",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=2,
        help="JPG quality for ffmpeg -q:v. Lower is better quality. Recommended: 2~5. Default: 2",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not found in PATH. Install it first, for example: sudo apt install ffmpeg"
        )

    video_root = args.video_root
    output_root = args.output_root

    if not video_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {video_root}")

    subdirs = sorted([p for p in video_root.iterdir() if p.is_dir()])
    if not subdirs:
        print(f"[WARN] No subfolders found under {video_root}")
        return

    jobs: List[Tuple[str, str, str, float, int]] = []
    for subdir in subdirs:
        video_files = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        save_dir = output_root / subdir.name
        print(f"[INFO] Subfolder {subdir.name}: found {len(video_files)} video files")

        for video_path in video_files:
            jobs.append(
                (
                    str(video_path),
                    str(save_dir),
                    args.output_ext,
                    args.interval_sec,
                    args.jpg_quality,
                )
            )

    if not jobs:
        print(f"[WARN] No video files found under {video_root}")
        return

    num_workers = max(1, min(args.num_workers, len(jobs)))
    print(f"\n[INFO] Total videos: {len(jobs)}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Interval: {args.interval_sec} seconds")
    print(f"[INFO] Output format: {args.output_ext}")
    print(f"[INFO] Parallel workers: {num_workers}\n")

    total_frames = 0
    total_videos = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_with_ffmpeg, *job) for job in jobs]

        for future in as_completed(futures):
            video_path, saved_count, status = future.result()
            total_videos += 1
            total_frames += saved_count
            print(f"{status} {video_path} | saved={saved_count}")

    print("\n[DONE]")
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames saved: {total_frames}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
