import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


SUPPORTED_VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# job = (video_path, save_dir, output_ext, mode, interval_sec, jpg_quality, output_stem)
Job = Tuple[str, str, str, str, float, int, str]


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS


def make_safe_stem(video_path: Path, root: Path) -> str:
    """Make a unique filename stem from the path relative to root."""
    try:
        rel = video_path.relative_to(root)
    except ValueError:
        rel = video_path.name
    rel_no_suffix = Path(rel).with_suffix("")
    return "_".join(rel_no_suffix.parts)


def collect_direct_videos(video_root: Path) -> List[Path]:
    return sorted(p for p in video_root.iterdir() if is_video_file(p))


def collect_recursive_videos(video_root: Path) -> List[Path]:
    return sorted(p for p in video_root.rglob("*") if is_video_file(p))


def collect_train_jobs(
    video_root: Path,
    output_root: Path,
    output_ext: str,
    interval_sec: float,
    jpg_quality: int,
) -> List[Job]:
    """
    Train mode:
      video/normal/01/*.avi -> data/posco/train/01/*.jpg
      video/normal/02/*.avi -> data/posco/train/02/*.jpg
    """
    jobs: List[Job] = []
    subdirs = sorted(p for p in video_root.iterdir() if p.is_dir())

    for subdir in subdirs:
        video_files = collect_direct_videos(subdir)
        save_dir = output_root / subdir.name
        print(f"[INFO] Train subfolder {subdir.name}: found {len(video_files)} video files")

        for video_path in video_files:
            jobs.append(
                (
                    str(video_path),
                    str(save_dir),
                    output_ext,
                    "interval",
                    interval_sec,
                    jpg_quality,
                    video_path.stem,
                )
            )

    return jobs


def collect_test_jobs(
    normal_video_root: Path,
    abnormal_video_root: Path,
    output_root: Path,
    output_ext: str,
    jpg_quality: int,
) -> List[Job]:
    """
    Test mode:
      normal:   video/normal/01/*.avi, video/normal/02/*.avi, ... -> data/posco/test/normal/*.jpg
                Extract only one frame per video.
      abnormal: video/anomaly/*.avi -> data/posco/test/abnormal/*.jpg
                Extract all frames.
    """
    jobs: List[Job] = []

    normal_save_dir = output_root / "normal"
    abnormal_save_dir = output_root / "abnormal"

    normal_videos = collect_recursive_videos(normal_video_root)
    print(f"[INFO] Test normal videos: found {len(normal_videos)} video files under {normal_video_root}")
    for video_path in normal_videos:
        # Include subfolder names in output stem to avoid overwriting same video names from 01/02/etc.
        output_stem = make_safe_stem(video_path, normal_video_root)
        jobs.append(
            (
                str(video_path),
                str(normal_save_dir),
                output_ext,
                "one_frame",
                0.0,
                jpg_quality,
                output_stem,
            )
        )

    if abnormal_video_root.exists():
        # User described video/anomaly/*.avi. This also supports subfolders just in case.
        abnormal_videos = collect_recursive_videos(abnormal_video_root)
    else:
        abnormal_videos = []
    print(f"[INFO] Test abnormal videos: found {len(abnormal_videos)} video files under {abnormal_video_root}")
    for video_path in abnormal_videos:
        output_stem = make_safe_stem(video_path, abnormal_video_root)
        jobs.append(
            (
                str(video_path),
                str(abnormal_save_dir),
                output_ext,
                "all_frames",
                0.0,
                jpg_quality,
                output_stem,
            )
        )

    return jobs


def run_ffmpeg_job(
    video_path_str: str,
    save_dir_str: str,
    output_ext: str,
    mode: str,
    interval_sec: float,
    jpg_quality: int,
    output_stem: str,
) -> Tuple[str, int, str]:
    video_path = Path(video_path_str)
    save_dir = Path(save_dir_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_ext = output_ext.lower().lstrip(".")

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
    ]

    if mode == "interval":
        if interval_sec <= 0:
            return str(video_path), 0, "[ERROR] interval_sec must be positive for interval mode"
        output_pattern = save_dir / f"{output_stem}_%06d.{output_ext}"
        cmd += ["-vf", f"fps=1/{interval_sec}", "-start_number", "0"]
        count_glob = f"{output_stem}_*.{output_ext}"
    elif mode == "one_frame":
        # Save only the first decoded frame from each normal video.
        output_pattern = save_dir / f"{output_stem}.{output_ext}"
        cmd += ["-frames:v", "1"]
        count_glob = f"{output_stem}.{output_ext}"
    elif mode == "all_frames":
        output_pattern = save_dir / f"{output_stem}_%06d.{output_ext}"
        cmd += ["-start_number", "0"]
        count_glob = f"{output_stem}_*.{output_ext}"
    else:
        return str(video_path), 0, f"[ERROR] Unknown mode: {mode}"

    # Lower q:v means higher quality. 2 is high quality and usually much faster/smaller than PNG.
    if output_ext in {"jpg", "jpeg"}:
        cmd += ["-q:v", str(jpg_quality)]

    cmd.append(str(output_pattern))

    before = set(save_dir.glob(count_glob))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "unknown ffmpeg error").strip()
        return str(video_path), 0, f"[ERROR] {err}"

    after = set(save_dir.glob(count_glob))
    saved_count = len(after - before)

    # If files were overwritten, after-before may be 0. Count matching outputs as fallback.
    if saved_count == 0:
        saved_count = len(after)

    return str(video_path), saved_count, "[OK]"


def run_jobs(jobs: Sequence[Job], num_workers: int, output_root: Path, output_ext: str) -> None:
    if not jobs:
        print("[WARN] No video files found. Nothing to extract.")
        return

    num_workers = max(1, min(num_workers, len(jobs)))
    print(f"\n[INFO] Total videos: {len(jobs)}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Output format: {output_ext}")
    print(f"[INFO] Parallel workers: {num_workers}\n")

    total_frames = 0
    total_videos = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_ffmpeg_job, *job) for job in jobs]

        for future in as_completed(futures):
            video_path, saved_count, status = future.result()
            total_videos += 1
            total_frames += saved_count
            print(f"{status} {video_path} | saved={saved_count}")

    print("\n[DONE]")
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames saved: {total_frames}")
    print(f"Output root: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast POSCO frame extraction using ffmpeg. "
            "Use --mode train for data/posco/train or --mode test for data/posco/test."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="train: extract interval frames to data/posco/train/<subdir>. test: extract normal/abnormal test frames.",
    )

    # Train mode arguments
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("video/normal"),
        help="Train input root. Expected: video/normal/01/*.avi, video/normal/02/*.avi, ...",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/posco/train"),
        help="Train output root. Frames are saved to data/posco/train/<subdir>/ in train mode.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=10.0,
        help="Train mode: extract one frame every this many seconds. Default: 10",
    )

    # Test mode arguments
    parser.add_argument(
        "--normal-video-root",
        type=Path,
        default=Path("video/normal"),
        help="Test mode normal input root. Supports subfolders, e.g., video/normal/01/*.avi.",
    )
    parser.add_argument(
        "--abnormal-video-root",
        type=Path,
        default=Path("video/anomaly"),
        help="Test mode abnormal input root. Expected: video/anomaly/*.avi. Subfolders are also supported.",
    )
    parser.add_argument(
        "--test-output-root",
        type=Path,
        default=Path("data/posco/test"),
        help="Test output root. Saves to data/posco/test/normal and data/posco/test/abnormal.",
    )

    # Common arguments
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

    if args.mode == "train":
        if not args.video_root.exists():
            raise FileNotFoundError(f"Train input folder does not exist: {args.video_root}")
        subdirs = sorted(p for p in args.video_root.iterdir() if p.is_dir())
        if not subdirs:
            print(f"[WARN] No subfolders found under {args.video_root}")
            return

        jobs = collect_train_jobs(
            video_root=args.video_root,
            output_root=args.output_root,
            output_ext=args.output_ext,
            interval_sec=args.interval_sec,
            jpg_quality=args.jpg_quality,
        )
        print(f"[INFO] Train interval: {args.interval_sec} seconds")
        run_jobs(jobs, args.num_workers, args.output_root, args.output_ext)

    elif args.mode == "test":
        if not args.normal_video_root.exists():
            raise FileNotFoundError(f"Test normal input folder does not exist: {args.normal_video_root}")
        if not args.abnormal_video_root.exists():
            raise FileNotFoundError(f"Test abnormal input folder does not exist: {args.abnormal_video_root}")

        jobs = collect_test_jobs(
            normal_video_root=args.normal_video_root,
            abnormal_video_root=args.abnormal_video_root,
            output_root=args.test_output_root,
            output_ext=args.output_ext,
            jpg_quality=args.jpg_quality,
        )
        print("[INFO] Test normal: extract one frame per video")
        print("[INFO] Test abnormal: extract all frames per video")
        run_jobs(jobs, args.num_workers, args.test_output_root, args.output_ext)


if __name__ == "__main__":
    main()
