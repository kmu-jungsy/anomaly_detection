import argparse
from pathlib import Path

import cv2


def extract_one_frame_every_n_seconds(video_path: Path, save_dir: Path, interval_sec: float = 10.0) -> int:
    """Extract one frame every `interval_sec` seconds from one video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 0:
        print(f"[WARN] Invalid FPS for {video_path}. Skip this video.")
        cap.release()
        return 0

    frame_interval = max(1, int(round(fps * interval_sec)))
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # Example: video_name_000000.png, video_name_000300.png, ...
        output_path = save_dir / f"{video_path.stem}_{frame_idx:06d}.png"
        ok = cv2.imwrite(str(output_path), frame)

        if ok:
            saved_count += 1
        else:
            print(f"[WARN] Failed to save: {output_path}")

        frame_idx += frame_interval

    cap.release()
    print(
        f"[OK] {video_path} -> {save_dir} | "
        f"fps={fps:.2f}, interval={interval_sec}s, saved={saved_count}"
    )
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract one frame every N seconds from video/normal/<subdir>/*.avi "
        "to data/posco/train/<subdir>/*.png"
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
    args = parser.parse_args()

    video_root = args.video_root
    output_root = args.output_root

    if not video_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {video_root}")

    subdirs = sorted([p for p in video_root.iterdir() if p.is_dir()])

    if not subdirs:
        print(f"[WARN] No subfolders found under {video_root}")
        return

    total_videos = 0
    total_frames = 0

    for subdir in subdirs:
        avi_files = sorted(subdir.glob("*.avi"))
        save_dir = output_root / subdir.name

        print(f"\n[INFO] Subfolder {subdir.name}: found {len(avi_files)} avi files")

        for video_path in avi_files:
            total_videos += 1
            total_frames += extract_one_frame_every_n_seconds(
                video_path=video_path,
                save_dir=save_dir,
                interval_sec=args.interval_sec,
            )

    print("\n[DONE]")
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames saved: {total_frames}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
