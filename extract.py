import cv2
from pathlib import Path

# Root folder containing normal/ and abnormal/
video_dir = Path("./video")
output_dir = Path("./extracted_frames")

normal_output_dir = output_dir / "normal"
abnormal_output_dir = output_dir / "abnormal"

normal_output_dir.mkdir(parents=True, exist_ok=True)
abnormal_output_dir.mkdir(parents=True, exist_ok=True)

normal_dir = video_dir / "normal"
abnormal_dir = video_dir / "abnormal"

normal_avi_files = sorted(normal_dir.glob("*.avi"))
abnormal_avi_files = sorted(abnormal_dir.glob("*.avi"))

print(f"Found {len(normal_avi_files)} normal avi files")
print(f"Found {len(abnormal_avi_files)} abnormal avi files")


def extract_frames(video_path: Path, save_dir: Path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        output_path = save_dir / f"{video_path.stem}_{frame_idx:06d}.png"
        cv2.imwrite(str(output_path), frame)

        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames from {video_path.name} to {save_dir}")


# Extract first 100 frames from normal videos
for video_path in normal_avi_files:
    extract_frames(
        video_path=video_path,
        save_dir=normal_output_dir,
        max_frames=100,
    )

# Extract all frames from abnormal videos
for video_path in abnormal_avi_files:
    extract_frames(
        video_path=video_path,
        save_dir=abnormal_output_dir,
        max_frames=None,
    )
