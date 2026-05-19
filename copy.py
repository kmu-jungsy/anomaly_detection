from pathlib import Path
import shutil
import re

folder = Path("data/posco/train/01")

# Select by filename prefix
selected_prefix = "[CH001] 0220_0240"

# Match: [CH001] 0220_0240_000000.jpg
pattern = re.compile(r"^(.*_)(\d{6})(\.jpg)$")

# Select only images starting with selected_prefix
selected_images = sorted(folder.glob(f"{selected_prefix}_*.jpg"))

print("Selected images:", len(selected_images))

if len(selected_images) == 0:
    print("No images found. Check selected_prefix or folder path.")
    exit()

# Find max number among all jpg images with this filename pattern
all_images = sorted(folder.glob("*.jpg"))
numbers = []

for img_path in all_images:
    match = pattern.match(img_path.name)
    if match:
        numbers.append(int(match.group(2)))

if len(numbers) == 0:
    print("No numbered jpg images found.")
    exit()

start_number = max(numbers) + 1
print("New copied images will start from:", start_number)

for idx, img_path in enumerate(selected_images):
    match = pattern.match(img_path.name)

    if not match:
        print(f"Skip unmatched file: {img_path.name}")
        continue

    prefix = match.group(1)
    suffix = match.group(3)

    new_number = start_number + idx
    new_name = f"{prefix}{new_number:06d}{suffix}"
    out_path = folder / new_name

    if out_path.exists():
        print(f"Already exists, skip: {out_path.name}")
        continue

    shutil.copy2(img_path, out_path)
    print(f"Copied: {img_path.name} -> {new_name}")
