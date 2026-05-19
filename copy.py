from pathlib import Path
import shutil
import re

folder = Path("data/posco/train/01")

selected_prefix = "[CH001] 0220_0240"

pattern = re.compile(r"^(.*_)(\d{6})(\.jpg)$")

# Safer: find all jpg files, then filter by exact prefix text
selected_images = sorted([
    p for p in folder.glob("*.jpg")
    if p.name.startswith(selected_prefix + "_")
])

print("Folder exists:", folder.exists())
print("Folder path:", folder.resolve())
print("Selected images:", len(selected_images))

# Debug: show some jpg filenames
print("First 10 jpg files:")
for p in sorted(folder.glob("*.jpg"))[:10]:
    print(repr(p.name))

if len(selected_images) == 0:
    print("No images found. Check if filename has different spaces, brackets, or extension.")
    exit()

# Find max number only among selected prefix files
numbers = []
for img_path in selected_images:
    match = pattern.match(img_path.name)
    if match:
        numbers.append(int(match.group(2)))

start_number = max(numbers) + 1
print("New copied images will start from:", start_number)

# Copy only the original selected images
original_images = selected_images.copy()

for idx, img_path in enumerate(original_images):
    match = pattern.match(img_path.name)
    if not match:
        print(f"Skip unmatched file: {repr(img_path.name)}")
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
