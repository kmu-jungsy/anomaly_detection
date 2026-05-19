from pathlib import Path
import shutil
import re

folder = Path(".")  # current folder

# Match files like: [CH001] 0220_0240_000000.jpg
pattern = re.compile(r"^(.*_)(\d{6})(\.jpg)$")

images = sorted(folder.glob("[CH001] 0220_0240_*.jpg"))

# Copy each image and increase last 6 digits by 90
offset = 90

for img_path in images:
    match = pattern.match(img_path.name)
    if not match:
        continue

    prefix = match.group(1)      # [CH001] 0220_0240_
    number = int(match.group(2)) # 000000
    suffix = match.group(3)      # .jpg

    new_number = number + offset
    new_name = f"{prefix}{new_number:06d}{suffix}"
    out_path = folder / new_name

    shutil.copy2(img_path, out_path)
    print(f"Copied: {img_path.name} -> {new_name}")
