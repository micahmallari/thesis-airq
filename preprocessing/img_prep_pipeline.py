import os, random, itertools
import numpy as np, pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

PATCH_SIZE  = 256
STRIDE      = 128
AUGMENT     = True
RAW_ROOT    = 'dataset/a_raw'
OUTPUT_ROOT = 'dataset/e_preprocessed_img'
PATCH_META  = os.path.join(OUTPUT_ROOT, 'patch_metadata.csv')

def z_score_normalize(t):
    m, s = t.mean(), t.std() + 1e-6
    return (t - m) / s

# Individual augmentations
def aug_rotate(img): return TF.rotate(img, random.uniform(-15, 15), fill=0)
def aug_brightness(img): return TF.adjust_brightness(img, random.uniform(0.8, 1.2))
def aug_contrast(img): return TF.adjust_contrast(img, random.uniform(0.8, 1.2))
def aug_hflip(img): return TF.hflip(img)

AUGMENTATIONS = {
    'rotate': aug_rotate,
    'brightness': aug_brightness,
    'contrast': aug_contrast,
    'hflip': aug_hflip
}

# Generate all non-empty combinations
AUG_COMBINATIONS = []
for r in range(1, len(AUGMENTATIONS)+1):
    AUG_COMBINATIONS.extend(itertools.combinations(AUGMENTATIONS.keys(), r))

def apply_augmentations(img, aug_list):
    for aug_name in aug_list:
        img = AUGMENTATIONS[aug_name](img)
    return img

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
    if not os.path.isdir(img_dir): return

    patch_out_dir = os.path.join(OUTPUT_ROOT, day_folder, 'patch')
    os.makedirs(patch_out_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in ('.png', '.jpg', '.jpeg'): continue

        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path)

        # PATCHING
        patches = extract_patches(img)
        for idx, patch in enumerate(patches):
            patch = patch.convert('RGB')  # 👈 Force 3 channels
            selected_combos = random.sample(AUG_COMBINATIONS, k=2)
            for aug_combo in selected_combos:
                aug_patch = apply_augmentations(patch.copy(), aug_combo)
                t = transforms.ToTensor()(aug_patch)  # shape should now be [3, 256, 256]

                # Sanity check
                if t.shape[0] != 3:
                    print(f"❌ Invalid patch shape {t.shape}, skipping: {npy_name}")
                    continue

                if t.mean() < 0.05 or t.std() < 0.01: continue
                tr = z_score_normalize(t)

                suffix = "_".join(aug_combo) if aug_combo else "none"
                npy_name = f"{name}_p{idx}_{suffix}.npy"
                arr = tr.numpy()

                np.save(os.path.join(patch_out_dir, npy_name), arr)
                print(f"Saved {npy_name} with shape: {arr.shape}")

                patch_meta.append({
                    'day': day_folder,
                    'image_filename': fname,
                    'patch_idx': idx,
                    'augmentations': suffix,
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