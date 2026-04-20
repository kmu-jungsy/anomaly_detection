import cv2
from pathlib import Path

# Folder containing avi files
video_dir = Path("./")   # current directory
output_dir = Path("./extracted_frames")
output_dir.mkdir(exist_ok=True)

avi_files = sorted(video_dir.glob("*.avi"))

print(f"Found {len(avi_files)} avi files")

for video_path in avi_files:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        continue

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        output_path = output_dir / f"{video_path.stem}.png"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to read frame from: {video_path}")

    cap.release()
