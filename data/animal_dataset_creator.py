import os
import json
from bing_image_downloader import downloader
from PIL import Image
import imagehash

# Configuration
emotions          = ["Angry", "Sad", "Other", "Happy"]
query_template    = "{} animal"
limit_per_emotion = 100
dataset_dir       = "dataset"
hash_file         = "hashes.json"  # stores all seen hashes

def load_hashes():
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return set(json.load(f))
    return set()

def save_hashes(hashes):
    with open(hash_file, "w") as f:
        json.dump(list(hashes), f)

def compute_hash(path):
    try:
        with Image.open(path) as img:
            # phash is more robust than average hash; you can also try imagehash.dhash or whash
            return str(imagehash.phash(img))
    except Exception:
        return None

def dedupe_folder(folder, existing_hashes):
    """
    Walk through `folder`, compute each image's hash,
    delete if hash already seen, otherwise add it.
    """
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        h = compute_hash(path)
        if h is None:
            # couldn’t read it—just remove or skip
            os.remove(path)
            continue

        if h in existing_hashes:
            os.remove(path)
        else:
            existing_hashes.add(h)

def main():
    os.makedirs(dataset_dir, exist_ok=True)
    hashes = load_hashes()

    for emotion in emotions:
        query = query_template.format(emotion if emotion != "Other" else "")
        print(f"→ Downloading {limit_per_emotion} images for '{query}'")
        downloader.download(
            query,
            limit=limit_per_emotion,
            output_dir=dataset_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )

        src = os.path.join(dataset_dir, query)
        dest = os.path.join(dataset_dir, emotion)
        if os.path.exists(src):
            os.rename(src, dest)
            print(f"   Renamed to '{dest}'—now deduping…")
            dedupe_folder(dest, hashes)
            print(f"   Kept {len(os.listdir(dest))} unique images in '{emotion}'")
        else:
            print(f"⚠ Warning: expected '{src}' not found; skipping dedupe")

    save_hashes(hashes)
    print("✅ Done. Master hash list now contains", len(hashes), "items.")

if __name__ == "__main__":
    main()
