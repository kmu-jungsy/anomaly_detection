import cv2
from pathlib import Path


def extract_frames_from_avi(
    video_path,
    output_dir,
    every_n_frames=1,
    max_frames=None
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved_idx = 0

    video_name = video_path.stem

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            output_path = output_dir / f"{video_name}_{saved_idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_idx += 1

            if max_frames is not None and saved_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()

    print(f"Saved {saved_idx} frames from {video_path.name} to {output_dir}")


if __name__ == "__main__":
    video_path = "video/sample.avi"
    output_dir = "frames"

    extract_frames_from_avi(
        video_path=video_path,
        output_dir=output_dir,
        every_n_frames=1,   # 1 means save every frame
        max_frames=None     # None means save all frames
    )
