from pathlib import Path

train_root = Path("data/posco/train")
test_normal_root = Path("data/posco/test/normal")

folders = ["02", "04", "06", "08"]
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

missing = []

for folder in folders:
    train_dir = train_root / folder
    test_dir = test_normal_root / folder

    if not train_dir.exists():
        print(f"[Warning] Train folder missing: {train_dir}")
        continue

    if not test_dir.exists():
        print(f"[Warning] Test normal folder missing: {test_dir}")
        continue

    train_names = {
        p.name
        for p in train_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    }

    test_paths = [
        p
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    for test_path in test_paths:
        if test_path.name not in train_names:
            missing.append(test_path)

    print(f"{folder}: test={len(test_paths)}, missing_from_train={sum(1 for p in missing if p.parent.name == folder)}")

print("\n=== Missing files ===")
for p in missing:
    print(p)

print(f"\nTotal missing: {len(missing)}")
