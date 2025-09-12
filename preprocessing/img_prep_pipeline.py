
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

PATCH_SIZE  = 256
STRIDE      = 128
RAW_ROOT    = 'dataset/a_raw'
OUTPUT_ROOT = 'dataset/e_preprocessed_img'
PATCH_META  = os.path.join(OUTPUT_ROOT, 'patch_metadata.csv')

def z_score_normalize(t):
    m, s = t.mean(), t.std() + 1e-6
    return (t - m) / s

def extract_patches(img):
    arr = np.array(img)
    patches = []
    w, h = img.size
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = arr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patches.append(Image.fromarray(patch))
    return patches

def process_day(day_folder, patch_meta):
    img_dir = os.path.join(RAW_ROOT, day_folder, 'images', 'pictures')
    if not os.path.isdir(img_dir):
        return
    patch_out_dir = os.path.join(OUTPUT_ROOT, day_folder, 'patch')
    os.makedirs(patch_out_dir, exist_ok=True)
    MAX_PATCHES_PER_IMAGE = 50  # Change this value as needed
    import random
    for fname in sorted(os.listdir(img_dir)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path)
        patches = extract_patches(img)
        # Randomly sample up to MAX_PATCHES_PER_IMAGE
        if len(patches) > MAX_PATCHES_PER_IMAGE:
            sampled_indices = random.sample(range(len(patches)), MAX_PATCHES_PER_IMAGE)
            sampled_patches = [patches[i] for i in sampled_indices]
        else:
            sampled_patches = patches
        for idx, patch in enumerate(sampled_patches):
            patch = patch.convert('RGB')
            t = transforms.ToTensor()(patch)
            if t.shape[0] != 3:
                print(f"❌ Invalid patch shape {t.shape}, skipping: {name}_p{idx}_none.npy")
                continue
            if t.mean() < 0.05 or t.std() < 0.01:
                continue
            tr = z_score_normalize(t)
            npy_name = f"{name}_p{idx}_none.npy"
            arr = tr.numpy()
            np.save(os.path.join(patch_out_dir, npy_name), arr)
            print(f"Saved {npy_name} with shape: {arr.shape}")
            patch_meta.append({
                'day': day_folder,
                'image_filename': fname,
                'patch_idx': idx,
                'augmentations': 'none',
                'npy_path': os.path.join(day_folder, 'patch', npy_name)
            })


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    patch_meta = []
    for day_folder in sorted(os.listdir(RAW_ROOT)):
        if os.path.isdir(os.path.join(RAW_ROOT, day_folder)):
            process_day(day_folder, patch_meta)
    pd.DataFrame(patch_meta).to_csv(PATCH_META, index=False)

if __name__ == '__main__':
    main()
