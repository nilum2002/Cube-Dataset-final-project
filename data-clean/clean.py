import os
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm
import shutil  # optional: for moving instead of deleting

# ================== CONFIG ==================
DATA_DIR = "/media/nilum/New_Volume/01.projects/Dataset-cubes/raw-data"          # your image folder
THRESHOLD = 12                       # Lower = stricter (fewer kept). Start with 5-10 for "little differences"
KEEP_DIR = "/media/nilum/New_Volume/01.projects/Dataset-cubes/cleaned"      # where unique images go
# ===========================================

os.makedirs(KEEP_DIR, exist_ok=True)


# Step 1: Compute perceptual hashes for all images
print("Computing perceptual hashes...")
image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
phash_list = []  # List of (filename, phash)
for filename in tqdm(sorted(image_files)):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB').resize((256, 256))
            phash = imagehash.phash(img)
            phash_list.append((filename, phash))
    except Exception as e:
        print(f"Error with {filename}: {e}")

# Step 2: Full pairwise comparison to remove similar images
print("Finding and removing similar images...")
kept = []
kept_hashes = []
for filename, phash in tqdm(phash_list):
    is_similar = False
    for kept_phash in kept_hashes:
        if phash - kept_phash <= THRESHOLD:
            is_similar = True
            # Uncomment below to see which images are considered similar
            # print(f"{filename} is similar to a kept image, skipping.")
            break
    if not is_similar:
        kept.append(filename)
        kept_hashes.append(phash)

# Step 3: Copy kept images to new folder
print(f"\nKeeping {len(kept)} unique images...")
for filename in tqdm(kept):
    src = os.path.join(DATA_DIR, filename)
    dst = os.path.join(KEEP_DIR, filename)
    shutil.copy(src, dst)

print("Cleaning done!")