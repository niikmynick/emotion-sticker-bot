import os
import json
from bing_image_downloader import downloader
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

# Configuration
emotions          = ["Angry", "Sad", "Other", "Happy"]
query_template    = "{} animal"
limit_per_emotion = 100
dataset_dir       = "dataset"
hash_file         = "hashes.json"  # stores all seen hashes

# Shared state
hashes = set()
hash_lock = threading.Lock()

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
            return str(imagehash.phash(img))
    except Exception:
        return None

def dedupe_folder(folder):
    """
    Compute hashes in parallel, then remove dups under a lock.
    """
    # 1) gather all file paths
    file_paths = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn))
    ]
    # 2) compute hashes with a process pool
    with ProcessPoolExecutor() as pool:
        hashes_list = list(pool.map(compute_hash, file_paths))

    # 3) walk through results under the lock
    for path, h in zip(file_paths, hashes_list):
        if h is None:
            print("No hashes found for {}".format(path))
            os.remove(path)
            continue
        with hash_lock:
            if h in hashes:
                print(f"Duplicate found: {h} for {path}, removing")
                os.remove(path)
            else:
                print(f"Unique image: {h} for {path}, keeping")
                hashes.add(h)

def process_emotion(emotion):
    """
    Download and dedupe one emotion folder. Meant to be run in a thread.
    """
    query = query_template.format(emotion) if emotion != "Other" else "animal"
    print(f"→ Downloading {limit_per_emotion} images for '{query}'")
    downloader.download(
        query,
        limit=limit_per_emotion,
        output_dir=dataset_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=False
    )
    src = os.path.join(dataset_dir, query)
    dest = os.path.join(dataset_dir, emotion)
    if not os.path.exists(src):
        print(f"⚠ Warning: expected '{src}' not found; skipping")
        return emotion, 0

    os.rename(src, dest)
    print(f"   Renamed to '{dest}'—now deduping…")
    dedupe_folder(dest)
    kept = len(os.listdir(dest))
    print(f"   Kept {kept} unique images in '{emotion}'")
    return emotion, kept

def main():
    os.makedirs(dataset_dir, exist_ok=True)
    global hashes
    hashes = load_hashes()

    # Thread pool to parallelize download+dedupe per emotion
    with ThreadPoolExecutor(max_workers=min(4, len(emotions))) as executor:
        futures = [executor.submit(process_emotion, emo) for emo in emotions]
        for fut in as_completed(futures):
            emo, kept = fut.result()
            print(f"✔ Finished '{emo}', {kept} images kept")

    save_hashes(hashes)
    print("✅ All done! Total hashes stored:", len(hashes))

if __name__ == "__main__":
    # main()
    for emo in emotions:
        dedupe_folder(os.path.join(dataset_dir, emo))
