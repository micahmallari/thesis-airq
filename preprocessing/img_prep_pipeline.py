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
    """Extract patches with spatial indexing for attention pooling"""
    arr = np.array(img)
    patches = []
    patch_positions = []  # Added spatial position tracking
    w, h = img.size
    patch_idx = 0
    
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = arr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patches.append(Image.fromarray(patch))
            patch_positions.append({
                'patch_idx': patch_idx,
                'x': x,
                'y': y,
                'center_x': x + PATCH_SIZE // 2,
                'center_y': y + PATCH_SIZE // 2
            })
            patch_idx += 1
    
    return patches, patch_positions

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

        # PATCHING with spatial information
        patches, patch_positions = extract_patches(img)
        
        for patch, pos_info in zip(patches, patch_positions):
            patch = patch.convert('RGB')  # 👈 Force 3 channels
            
            selected_combos = random.sample(AUG_COMBINATIONS, k=2)
            
            for aug_combo in selected_combos:
                aug_patch = apply_augmentations(patch.copy(), aug_combo)
                t = transforms.ToTensor()(aug_patch)  # shape should now be [3, 256, 256]

                # Sanity check
                if t.shape[0] != 3:
                    print(f" Invalid patch shape {t.shape}, skipping")
                    continue

                if t.mean() < 0.05 or t.std() < 0.01: continue
                tr = z_score_normalize(t)

                suffix = "_".join(aug_combo) if aug_combo else "none"
                npy_name = f"{name}_p{pos_info['patch_idx']}_{suffix}.npy"
                arr = tr.numpy()

                np.save(os.path.join(patch_out_dir, npy_name), arr)
                print(f"Saved {npy_name} with shape: {arr.shape}")

                patch_meta.append({
                    'day': day_folder,
                    'image_filename': fname,
                    'patch_idx': pos_info['patch_idx'],  # Spatial index for attention pooling
                    'augmentations': suffix,
                    'npy_path': os.path.join(day_folder, 'patch', npy_name),
                    # Additional spatial information for advanced attention mechanisms
                    'patch_x': pos_info['x'],
                    'patch_y': pos_info['y'],
                    'patch_center_x': pos_info['center_x'],
                    'patch_center_y': pos_info['center_y']
                })


def main():
    """
    Enhanced image preprocessing pipeline with spatial encoding support.
    
    This pipeline creates patches with spatial indexing and augmentation type tracking
    to support attention pooling with positional and augmentation embeddings.
    """
    print(" Starting enhanced image preprocessing pipeline")
    print("   Features: Spatial indexing + Augmentation tracking for attention pooling")
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    patch_meta = []
    
    processed_days = 0
    total_patches = 0
    
    for day_folder in sorted(os.listdir(RAW_ROOT)):
        if os.path.isdir(os.path.join(RAW_ROOT, day_folder)):
            print(f"\n Processing {day_folder}...")
            initial_patches = len(patch_meta)
            
            process_day(day_folder, patch_meta)
            
            day_patches = len(patch_meta) - initial_patches
            total_patches += day_patches
            processed_days += 1
            
            print(f"    {day_folder}: {day_patches} patches created")
    
    # Save enhanced metadata
    df = pd.DataFrame(patch_meta)
    df.to_csv(PATCH_META, index=False)
    
    print(f"\n Pipeline completed!")
    print(f"   Days processed: {processed_days}")
    print(f"   Total patches: {total_patches}")
    print(f"   Metadata saved: {PATCH_META}")
    print(f"   Spatial encoding ready: ✅")
    print(f"   Augmentation tracking ready: ✅")
    
    # Display sample metadata
    if len(patch_meta) > 0:
        print(f"\n Sample metadata:")
        sample = patch_meta[0]
        for key, value in sample.items():
            print(f"   {key}: {value}")

if __name__ == '__main__':
    main()
