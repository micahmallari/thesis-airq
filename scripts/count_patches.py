import pandas as pd
import os

PATCH_META = 'dataset/e_preprocessed_img/patch_metadata.csv'

df = pd.read_csv(PATCH_META)
print("Number of patches:", len(df))
print("Estimated total size (GB):", len(df) * 768 / 1024 / 1024)
print(f"Unique images: {df['image_filename'].nunique()}")

print("\nPatches per day:")
for day, group in df.groupby('day'):
    print(f"  {day}: {len(group)} patches, {group['image_filename'].nunique()} images")