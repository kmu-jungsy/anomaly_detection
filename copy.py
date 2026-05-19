from pathlib import Path
import shutil
import re

folder = Path("data/posco/train/01")

# Match only:
# [CH001] 0220_0240_000000.jpg
# [CH001] 0220_0240_000001.jpg
# ...
pattern = re.compile(r"^(\[CH001\] 0220_0240_)(\d{6})(\.jpg)$")

images = sorted(folder.glob("*.jpg"))

print("Folder exists:", folder.exists())
print("Found jpg images:", len(images))

offset = 90

for img_path in images:
    match = pattern.match(img_path.name)

    # Skip other jpg files
    if not match:
        print(f"Skip: {img_path.name}")
        continue

    prefix = match.group(1)
    number = int(match.group(2))
    suffix = match.group(3)

    # Copy only original 000000 ~ 000089
    if number < 0 or number > 89:
        continue

    new_number = number + offset
    new_name = f"{prefix}{new_number:06d}{suffix}"
    out_path = folder / new_name

    if out_path.exists():
        print(f"Already exists, skip: {out_path.name}")
        continue

    shutil.copy2(img_path, out_path)
    print(f"Copied: {img_path.name} -> {new_name}")
