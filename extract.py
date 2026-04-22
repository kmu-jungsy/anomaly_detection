import cv2
from pathlib import Path

# Folder containing avi files
video_dir = Path("./")  # current directory
output_dir = Path("./extracted_frames")
output_dir.mkdir(exist_ok=True)

avi_files = sorted(video_dir.glob("*.avi"))

print(f"Found {len(avi_files)} avi files")

for video_path in avi_files:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        continue

    # Make output folder for each video
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(exist_ok=True)

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output_path = video_output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(output_path), frame)

        frame_idx += 1

    cap.release()

    print(f"Saved {frame_idx} frames from {video_path.name} to {video_output_dir}")
