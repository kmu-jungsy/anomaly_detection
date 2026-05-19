from pathlib import Path
import shutil

# input folder
normal_dir = Path("data/posco/train/normal")

# output folder
output_dir = Path("data/posco/train/normal_90copies")
output_dir.mkdir(parents=True, exist_ok=True)

# normal1.jpg ~ normal15.jpg
for i in range(1, 16):
    src = normal_dir / f"normal{i}.jpg"

    if not src.exists():
        print(f"Missing file: {src}")
        continue

    for j in range(90):
        dst = output_dir / f"normal{i}_{j:06d}.jpg"
        shutil.copy2(src, dst)

    print(f"Copied {src.name} x 90")

print("Done.")
