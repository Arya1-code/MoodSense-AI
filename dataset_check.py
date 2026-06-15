from pathlib import Path

dataset_path = Path("datasets/raw/fer2013/train")

for emotion in dataset_path.iterdir():
    if emotion.is_dir():
        count = len(list(emotion.glob("*")))
        print(f"{emotion.name}: {count}")