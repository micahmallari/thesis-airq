"""Complete CapsNet Training, Testing, and Hyperparameter Tuning
Combined all training-related functionality into one file"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import traceback
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import json
from datetime import datetime
import shutil
import time
import warnings


# Fix imports - use absolute imports
try:
    from models.capsnet_model import create_capsnet_feature_extractor
except ImportError:
    # Try alternative import path
    try:
        import models.capsnet_model as capsnet_model
        create_capsnet_feature_extractor = capsnet_model.create_capsnet_feature_extractor
    except ImportError:
        # Last resort - add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        from models.capsnet_model import create_capsnet_feature_extractor

# Try to import Optuna for hyperparameter tuning (optional)
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("✅ Optuna available for hyperparameter tuning")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install with: pip install optuna")
    print("   Hyperparameter tuning will be disabled.")

class OutputManager:
    """Manages organized output folder structure"""
    
    def __init__(self, base_dir="outputs", model_type="capsnet"):
        self.base_dir = base_dir
        self.model_type = model_type.lower()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main directory structure
        self.structure = {
            'models': f"{base_dir}/{model_type}/models",
            'features': f"{base_dir}/{model_type}/features", 
            'plots': f"{base_dir}/{model_type}/plots",
            'logs': f"{base_dir}/{model_type}/logs",
            'checkpoints': f"{base_dir}/{model_type}/checkpoints",
            'hyperparameters': f"{base_dir}/{model_type}/hyperparameters",
            'metadata': f"{base_dir}/{model_type}/metadata",
            'experiments': f"{base_dir}/{model_type}/experiments",
            'tuning': f"{base_dir}/{model_type}/tuning"
        }
        
        # Create all directories
        self.create_directories()
        
        print(f"📁 Output structure created for {model_type.upper()}")
        print(f"   Base directory: {base_dir}")
        print(f"   Timestamp: {self.timestamp}")
    
    def create_directories(self):
        """Create all necessary directories"""
        for folder_type, path in self.structure.items():
            os.makedirs(path, exist_ok=True)
            
        # Create subdirectories for better organization
        subdirs = {
            'models': ['best', 'checkpoints', 'final', 'tuned'],
            'features': ['train', 'val', 'test', 'extracted'],
            'plots': ['training', 'validation', 'features', 'analysis', 'tuning'],
            'logs': ['training', 'testing', 'tuning'],
            'checkpoints': ['trunk', 'epoch', 'best'],
            'experiments': ['runs', 'comparisons', 'ablations'],
            'tuning': ['studies', 'results', 'plots', 'best_models'],
            'hyperparameters': ['basic', 'advanced', 'studies']
        }
        
        for main_dir, sub_dirs in subdirs.items():
            for sub_dir in sub_dirs:
                full_path = os.path.join(self.structure[main_dir], sub_dir)
                os.makedirs(full_path, exist_ok=True)
    
    def get_path(self, folder_type, subfolder=None, filename=None, day_folder=None):
        """Get organized path for different types of outputs"""
        base_path = self.structure[folder_type]
        
        # Add day folder if specified
        if day_folder:
            base_path = os.path.join(base_path, day_folder)
            os.makedirs(base_path, exist_ok=True)
        
        # Add subfolder if specified
        if subfolder:
            base_path = os.path.join(base_path, subfolder)
            os.makedirs(base_path, exist_ok=True)
        
        # Add filename if specified
        if filename:
            # Add timestamp to filename if it doesn't already have one
            if self.timestamp not in filename:
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{self.timestamp}{ext}"
            base_path = os.path.join(base_path, filename)
        
        return base_path
    
    def save_experiment_info(self, day_folder, config):
        """Save experiment configuration and info"""
        experiment_info = {
            'timestamp': self.timestamp,
            'model_type': self.model_type,
            'day_folder': day_folder,
            'config': config,
            'output_structure': self.structure
        }
        
        info_path = self.get_path('experiments', 'runs', f'experiment_{day_folder}_{self.timestamp}.json')
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)
        
        print(f"📋 Experiment info saved: {info_path}")
        return info_path
    
    def create_summary_report(self, day_folder, results):
        """Create a summary report of the experiment"""
        report_path = self.get_path('experiments', 'runs', f'summary_{day_folder}_{self.timestamp}.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# {self.model_type.upper()} Experiment Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Day Folder:** {day_folder}\n")
            f.write(f"**Model Type:** {self.model_type.upper()}\n\n")
            
            f.write("## Results\n")
            for key, value in results.items():
                f.write(f"- **{key}:** {value}\n")
            
            f.write("\n## Output Files\n")
            for folder_type, path in self.structure.items():
                f.write(f"- **{folder_type.title()}:** `{path}`\n")
        
        print(f"📊 Summary report created: {report_path}")
        return report_path

def custom_collate_fn(batch):
    """Optimized custom collate function"""
    images = []
    pm25_values = []
    metadata_list = []
    
    for item in batch:
        try:
            img, pm25, metadata = item
            images.append(img)
            pm25_values.append(pm25)
            metadata_list.append(metadata)
        except Exception as e:
            print(f"Error in collate_fn for item: {e}")
            continue
    
    if not images:
        return torch.empty(0, 3, 256, 256), torch.empty(0, 1), []
    
    # Stack tensors
    images = torch.stack(images, 0)
    pm25_values = torch.stack(pm25_values, 0)
    
    return images, pm25_values, metadata_list

class AirQualityDataset(Dataset):
    """Air Quality Dataset that correctly follows the data flow"""
    
    def __init__(self, learning_df, patch_metadata_df, day_folder, split_type='learning', transform=None):
        self.learning_df = learning_df.reset_index(drop=True)
        self.patch_metadata_df = patch_metadata_df.reset_index(drop=True)
        self.day_folder = day_folder
        self.split_type = split_type
        self.transform = transform
        
        # Filter patch metadata for this specific day
        self.day_patches = self.patch_metadata_df[
            self.patch_metadata_df['day'] == day_folder
        ].reset_index(drop=True)
        
        print(f"📊 Dataset initialization for {day_folder} ({split_type}):")
        print(f"   Input learning data: {len(self.learning_df)} entries")
        print(f"   Available patches for day: {len(self.day_patches)} patches")
        
        # Create data mapping
        self.data_mapping = self._create_data_mapping()
        
        print(f"   Final dataset size: {len(self.data_mapping)} samples")
        if len(self.learning_df) > 0:
            print(f"   Expansion factor: {len(self.data_mapping) / len(self.learning_df):.1f}x")
    
    def _create_data_mapping(self):
        """Create mapping between learning data entries and patch files"""
        mapping = []
        images_with_patches = 0
        images_without_patches = 0
        
        print(f"   📊 Creating data mapping...")
        print(f"   Processing {len(self.learning_df)} learning entries...")
        
        # Pre-group patches by image filename for faster lookup
        patches_by_image = self.day_patches.groupby('image_filename')
        
        # Add progress bar for mapping creation
        progress_bar = tqdm(
            self.learning_df.iterrows(), 
            total=len(self.learning_df),
            desc="   Mapping data",
            leave=False,
            ncols=100
        )
        
        for idx, (_, learning_row) in enumerate(progress_bar):
            image_filename = learning_row['image_filename']
            
            # Find all patches for this image using pre-grouped data
            if image_filename in patches_by_image.groups:
                image_patches = patches_by_image.get_group(image_filename)
                images_with_patches += 1
                
                # Create one entry per patch (each patch represents one augmented version)
                for _, patch_row in image_patches.iterrows():
                    mapping.append({
                        # From learning data
                        'image_filename': image_filename,
                        'timestamp': learning_row['timestamp'],
                        'pm2.5': float(learning_row['pm2.5']),
                        'pm10': float(learning_row.get('pm10', 0)),
                        'temperature': float(learning_row.get('temperature', 0)),
                        'humidity': float(learning_row.get('humidity', 0)),
                        'location': learning_row.get('location', 'unknown'),
                        
                        # From patch metadata
                        'patch_idx': int(patch_row['patch_idx']),
                        'augmentations': patch_row['augmentations'],
                        'npy_path': patch_row['npy_path']
                    })
            else:
                images_without_patches += 1
            
            # Update progress bar with current stats every 10 items
            if (idx + 1) % 10 == 0:
                progress_bar.set_postfix({
                    'mapped': len(mapping),
                    'with_patches': images_with_patches,
                    'without': images_without_patches,
                    'avg_patches_per_img': f"{len(mapping)/max(images_with_patches, 1):.1f}"
                })
        
        progress_bar.close()
        
        print(f"   ✅ Mapping completed:")
        print(f"      Total samples: {len(mapping)}")
        print(f"      Images with patches: {images_with_patches}")
        print(f"      Images without patches: {images_without_patches}")
        if images_with_patches > 0:
            avg_patches = len(mapping) / images_with_patches
            print(f"      Average patches per image: {avg_patches:.1f}")
            print(f"      This means each image has ~{avg_patches:.0f} augmented versions")
        
        return mapping
    
    def __len__(self):
        return len(self.data_mapping)
    
    def __getitem__(self, idx):
        data_entry = self.data_mapping[idx]
        
        # Load preprocessed .npy file
        npy_path = data_entry['npy_path']
        if not os.path.isabs(npy_path):
            npy_path = os.path.join('dataset', 'e_preprocessed_img', npy_path)
        
        # Load the preprocessed image
        try:
            image_data = np.load(npy_path)
            
            # Ensure correct format: (C, H, W)
            if len(image_data.shape) == 3:
                if image_data.shape[0] == 3:  # Already CHW
                    image = torch.FloatTensor(image_data)
                elif image_data.shape[2] == 3:  # HWC -> CHW
                    image = torch.FloatTensor(image_data).permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected image shape: {image_data.shape}")
            else:
                raise ValueError(f"Expected 3D image, got shape: {image_data.shape}")
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            # Ensure correct size (256x256)
            if image.shape[1] != 256 or image.shape[2] != 256:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)
        
        except Exception as e:
            print(f"❌ Error loading {npy_path}: {e}")
            # Return dummy data as fallback
            image = torch.zeros(3, 256, 256, dtype=torch.float32)
        
        # Get PM2.5 target
        pm25_val = data_entry['pm2.5']
        if pd.isna(pm25_val) or pm25_val <= 0:
            pm25_val = 1.0  # Fallback value
        
        pm25 = torch.FloatTensor([pm25_val])
        
        return image, pm25, data_entry

import gc
import psutil

class TrunkAirQualityDataset(Dataset):
    """Memory-efficient dataset that loads data in trunks"""
    
    def __init__(self, learning_df, patch_metadata_df, day_folder, trunk_size=10000, split_type='learning', transform=None, max_patches_per_image=50):
        self.learning_df = learning_df.reset_index(drop=True)
        self.patch_metadata_df = patch_metadata_df.reset_index(drop=True)
        self.day_folder = day_folder
        self.trunk_size = trunk_size
        self.split_type = split_type
        self.transform = transform
        self.max_patches_per_image = max_patches_per_image  # NEW: Limit patches per image
        
        # Filter patch metadata for this specific day
        self.day_patches = self.patch_metadata_df[
            self.patch_metadata_df['day'] == day_folder
        ].reset_index(drop=True)
        
        print(f"📊 TrunkDataset initialization for {day_folder} ({split_type}):")
        print(f"   Input learning data: {len(self.learning_df)} entries")
        print(f"   Available patches for day: {len(self.day_patches)} patches")
        print(f"   Max patches per image: {max_patches_per_image}")
        
        # Pre-group patches by image filename for faster lookup
        self.patches_by_image = self.day_patches.groupby('image_filename')
        
        # Calculate total samples with sampling
        self.total_samples = self._calculate_total_samples_with_sampling()
        self.num_trunks = max(1, (self.total_samples + trunk_size - 1) // trunk_size)
        
        print(f"   Total samples (after sampling): {self.total_samples:,}")
        print(f"   Trunk size: {trunk_size:,}")
        print(f"   Number of trunks: {self.num_trunks}")
        print(f"   Estimated training time: {self.num_trunks * 0.5:.1f} hours (at 30min/trunk)")
        
        # Current trunk data
        self.current_trunk = 0
        self.current_trunk_data = []
        if self.total_samples > 0:
            self._load_trunk(0)

    def _calculate_total_samples_with_sampling(self):
        """Calculate total samples with intelligent sampling"""
        total = 0
        for _, learning_row in self.learning_df.iterrows():
            image_filename = learning_row['image_filename']
            if image_filename in self.patches_by_image.groups:
                image_patches = self.patches_by_image.get_group(image_filename)
                # Limit patches per image to reduce redundancy
                patches_to_use = min(len(image_patches), self.max_patches_per_image)
                total += patches_to_use
        return total

    def _load_trunk(self, trunk_idx):
        """Load a specific trunk of data with sampling"""
        if self.total_samples == 0:
            self.current_trunk_data = []
            return
            
        print(f"📦 Loading trunk {trunk_idx + 1}/{self.num_trunks}...")
        
        start_idx = trunk_idx * self.trunk_size
        end_idx = min(start_idx + self.trunk_size, self.total_samples)
        
        self.current_trunk_data = []
        current_sample_idx = 0
        
        # Create mapping for this trunk with sampling
        for _, learning_row in self.learning_df.iterrows():
            image_filename = learning_row['image_filename']
            
            if image_filename in self.patches_by_image.groups:
                image_patches = self.patches_by_image.get_group(image_filename)
                
                # Sample patches intelligently - take diverse augmentations
                if len(image_patches) > self.max_patches_per_image:
                    # Sample evenly across different augmentation types
                    sampled_patches = image_patches.sample(n=self.max_patches_per_image, random_state=42)
                else:
                    sampled_patches = image_patches
                
                for _, patch_row in sampled_patches.iterrows():
                    if start_idx <= current_sample_idx < end_idx:
                        self.current_trunk_data.append({
                            # From learning data
                            'image_filename': image_filename,
                            'timestamp': learning_row['timestamp'],
                            'pm2.5': float(learning_row['pm2.5']),
                            'pm10': float(learning_row.get('pm10', 0)),
                            'temperature': float(learning_row.get('temperature', 0)),
                            'humidity': float(learning_row.get('humidity', 0)),
                            'location': learning_row.get('location', 'unknown'),
                            
                            # From patch metadata
                            'patch_idx': int(patch_row['patch_idx']),
                            'augmentations': patch_row['augmentations'],
                            'npy_path': patch_row['npy_path']
                        })
                    
                    current_sample_idx += 1
                    
                    if current_sample_idx >= end_idx:
                        break
            
            if current_sample_idx >= end_idx:
                break
        
        self.current_trunk = trunk_idx
        print(f"   ✅ Loaded {len(self.current_trunk_data):,} samples for trunk {trunk_idx + 1}")
        
        # Force garbage collection
        gc.collect()
    
    def get_trunk_count(self):
        """Get total number of trunks"""
        return self.num_trunks
    
    def load_next_trunk(self):
        """Load next trunk if available"""
        if self.current_trunk + 1 < self.num_trunks:
            self._load_trunk(self.current_trunk + 1)
            return True
        return False
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def __len__(self):
        return len(self.current_trunk_data)
    
    def __getitem__(self, idx):
        if idx >= len(self.current_trunk_data):
            raise IndexError(f"Index {idx} out of range for current trunk")
        
        data_entry = self.current_trunk_data[idx]
        
        # Load preprocessed .npy file
        npy_path = data_entry['npy_path']
        if not os.path.isabs(npy_path):
            npy_path = os.path.join('dataset', 'e_preprocessed_img', npy_path)
        
        try:
            image_data = np.load(npy_path)
            
            # Ensure correct format: (C, H, W)
            if len(image_data.shape) == 3:
                if image_data.shape[0] == 3:  # Already CHW
                    image = torch.FloatTensor(image_data)
                elif image_data.shape[2] == 3:  # HWC -> CHW
                    image = torch.FloatTensor(image_data).permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected image shape: {image_data.shape}")
            else:
                raise ValueError(f"Expected 3D image, got shape: {image_data.shape}")
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            # Ensure correct size (256x256)
            if image.shape[1] != 256 or image.shape[2] != 256:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)
        
        except Exception as e:
            print(f"❌ Error loading {npy_path}: {e}")
            # Return dummy data as fallback
            image = torch.zeros(3, 256, 256, dtype=torch.float32)
        
        # Get PM2.5 target
        pm25_val = data_entry['pm2.5']
        if pd.isna(pm25_val) or pm25_val <= 0:
            pm25_val = 1.0  # Fallback value
        
        pm25 = torch.FloatTensor([pm25_val])
        
        return image, pm25, data_entry

class CapsNetTrainer:
    """Complete CapsNet Trainer with training, testing, and hyperparameter tuning"""
    
    def __init__(self, input_size=256, feature_dim=128, device='cuda', model_type='capsnet', 
                 use_attention_pooling=False, max_patches_per_image=50, attention_heads=4):
        if torch.cuda.is_available():
            try:
                self.device = torch.device(device)   # e.g., "cuda:0"
            except Exception as e:
                print(f"⚠️ Could not use {device}, falling back to CPU. Error: {e}")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.input_size = input_size
        self.feature_dim = feature_dim
        
        self.use_attention_pooling = use_attention_pooling
        self.max_patches_per_image = max_patches_per_image
        self.attention_heads = attention_heads
        
        # Automatically set model type for attention pooling
        if use_attention_pooling:
            self.model_type = 'capsnet_with_attention_pooling'
        else:
            self.model_type = model_type
        
        # Initialize output manager
        self.output_manager = OutputManager(model_type=self.model_type)
        
        print(f"🚀 CapsNet Trainer")
        print(f"   Device: {self.device}")
        print(f"   Input size: {input_size}x{input_size}")
        print(f"   Feature dimension: {feature_dim}")
        if use_attention_pooling:
            print(f"   Attention pooling: Enabled")
            print(f"   Max patches per image: {max_patches_per_image}")
            print(f"   Attention heads: {attention_heads}")
        
        # Initialize model
        self.feature_extractor = None
        self.temp_regressor = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def create_model(self, **model_params):
        """Create CapsNet feature extractor model"""
        self.feature_extractor = create_capsnet_feature_extractor(
            input_channels=3,
            input_size=self.input_size,
            feature_dim=self.feature_dim,
            **model_params
        ).to(self.device)
        
        # Temporary regression head for training
        dropout_rate = model_params.get('dropout_rate', 0.3)
        self.temp_regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        return self.feature_extractor
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4, optimizer_type='adam'):
        """Setup optimizer and scheduler"""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                lr=learning_rate, weight_decay=weight_decay, momentum=0.9
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5
        )
    
    def load_day_data(self, day_folder):
        """Load pre-split data from d_data_split"""
        print(f"📂 Loading data for {day_folder}...")
        
        # Load learning data (80% split)
        learning_path = f"dataset/d_data_split/{day_folder}/learning.csv"
        if not os.path.exists(learning_path):
            raise FileNotFoundError(f"Learning data not found: {learning_path}")
        
        print(f"   📄 Loading learning data from {learning_path}")
        learning_df = pd.read_csv(learning_path)
        print(f"   ✅ Learning data loaded: {len(learning_df)} entries")
        
        # Load patch metadata
        patch_metadata_path = "dataset/e_preprocessed_img/patch_metadata.csv"
        if not os.path.exists(patch_metadata_path):
            raise FileNotFoundError(f"Patch metadata not found: {patch_metadata_path}")
        
        print(f"   📄 Loading patch metadata from {patch_metadata_path}")
        patch_metadata_df = pd.read_csv(patch_metadata_path)
        day_patches = patch_metadata_df[patch_metadata_df['day'] == day_folder]
        print(f"   ✅ Patches for {day_folder}: {len(day_patches)} patches")
        
        # Clean data with progress
        print(f"   🧹 Cleaning data...")
        original_count = len(learning_df)
        learning_df = learning_df.dropna(subset=['pm2.5'])
        after_dropna = len(learning_df)
        learning_df = learning_df[learning_df['pm2.5'] > 0]
        after_positive = len(learning_df)
        learning_df = learning_df[learning_df['pm2.5'] < 500]
        final_count = len(learning_df)
        
        print(f"   ✅ Data cleaning complete:")
        print(f"      Original: {original_count} entries")
        print(f"      After dropna: {after_dropna} entries (removed {original_count - after_dropna})")
        print(f"      After >0 filter: {after_positive} entries (removed {after_dropna - after_positive})")
        print(f"      Final: {final_count} entries (removed {after_positive - final_count})")
        print(f"      PM2.5 range: {learning_df['pm2.5'].min():.2f} - {learning_df['pm2.5'].max():.2f}")
        
        return learning_df, patch_metadata_df
    
    def prepare_data(self, day_folder, test_size=0.2):
        """Prepare training and validation data"""
        print(f"🔄 Preparing data for {day_folder}...")
        
        learning_df, patch_metadata_df = self.load_day_data(day_folder)
        
        # Check if separate test file exists
        test_path = f"dataset/d_data_split/{day_folder}/test.csv"
        if os.path.exists(test_path):
            print(f"   📂 Using pre-split test data from {test_path}")
            # Use pre-split test data
            test_df = pd.read_csv(test_path)
            test_df = test_df.dropna(subset=['pm2.5'])
            test_df = test_df[test_df['pm2.5'] > 0]
            test_df = test_df[test_df['pm2.5'] < 500]
            
            print(f"   🔄 Creating training dataset...")
            train_dataset = AirQualityDataset(learning_df, patch_metadata_df, day_folder, 'learning')
            print(f"   🔄 Creating validation dataset...")
            val_dataset = AirQualityDataset(test_df, patch_metadata_df, day_folder, 'test')
        else:
            print(f"   📂 Splitting learning data ({test_size*100:.0f}% for validation)")
            # Split learning data
            train_df, val_df = train_test_split(learning_df, test_size=test_size, random_state=42)
            
            print(f"   🔄 Creating training dataset...")
            train_dataset = AirQualityDataset(train_df, patch_metadata_df, day_folder, 'train')
            print(f"   🔄 Creating validation dataset...")
            val_dataset = AirQualityDataset(val_df, patch_metadata_df, day_folder, 'val')
        
        print(f"✅ Data preparation complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset, learning_df, patch_metadata_df
    
    def prepare_trunk_data(self, day_folder, trunk_size=10000, test_size=0.2):
        """Prepare trunk-based training and validation data"""
        print(f"🔄 Preparing trunk data for {day_folder}...")
        print(f"   Trunk size: {trunk_size:,}")
        
        learning_df, patch_metadata_df = self.load_day_data(day_folder)
        
        # Check if separate test file exists
        test_path = f"dataset/d_data_split/{day_folder}/test.csv"
        if os.path.exists(test_path):
            print(f"   📂 Using pre-split test data from {test_path}")
            # Use pre-split test data
            test_df = pd.read_csv(test_path)
            test_df = test_df.dropna(subset=['pm2.5'])
            test_df = test_df[test_df['pm2.5'] > 0]
            test_df = test_df[test_df['pm2.5'] < 500]
            
            print(f"   🔄 Creating trunk training dataset...")
            train_dataset = TrunkAirQualityDataset(learning_df, patch_metadata_df, day_folder, trunk_size, 'learning')
            print(f"   🔄 Creating validation dataset...")
            val_dataset = AirQualityDataset(test_df, patch_metadata_df, day_folder, 'test')
        else:
            print(f"   📂 Splitting learning data ({test_size*100:.0f}% for validation)")
            # Split learning data
            train_df, val_df = train_test_split(learning_df, test_size=test_size, random_state=42)
            
            print(f"   🔄 Creating trunk training dataset...")
            train_dataset = TrunkAirQualityDataset(train_df, patch_metadata_df, day_folder, trunk_size, 'train')
            print(f"   🔄 Creating validation dataset...")
            val_dataset = AirQualityDataset(val_df, patch_metadata_df, day_folder, 'val')
        
        print(f"✅ Trunk data preparation complete:")
        print(f"   Training samples: {train_dataset.total_samples:,} (in {train_dataset.get_trunk_count()} trunks)")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset, learning_df, patch_metadata_df
    
    def train_epoch(self, dataloader, gradient_clip_norm=1.0):
        """Training epoch"""
        self.feature_extractor.train()
        self.temp_regressor.train()
        
        total_loss = 0
        predictions = []
        targets = []
        successful_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_images, batch_targets, batch_metadata in progress_bar:
            try:
                batch_images = batch_images.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                if batch_images.size(0) == 0:
                    continue
                
                self.optimizer.zero_grad()
                
                # Extract features and predict
                features = self.feature_extractor(batch_images)
                pred = self.temp_regressor(features)                
                loss = self.criterion(pred.squeeze(), batch_targets.squeeze())
                
                if torch.isnan(loss):
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.feature_extractor.parameters()) + list(self.temp_regressor.parameters()),
                    max_norm=gradient_clip_norm
                )
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                predictions.extend(pred.squeeze().cpu().detach().numpy())
                targets.extend(batch_targets.squeeze().cpu().detach().numpy())
                successful_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"❌ Error in training batch: {e}")
                continue
        
        if successful_batches == 0:
            raise RuntimeError("No successful training batches!")
        
        # Calculate metrics
        avg_loss = total_loss / successful_batches
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def validate_epoch(self, dataloader):
        """Validation epoch"""
        self.feature_extractor.eval()
        self.temp_regressor.eval()
        
        total_loss = 0
        predictions = []
        targets = []
        successful_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_images, batch_targets, batch_metadata in progress_bar:
                try:
                    batch_images = batch_images.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    if batch_images.size(0) == 0:
                        continue
                    
                    features = self.feature_extractor(batch_images)
                    pred = self.temp_regressor(features)
                    loss = self.criterion(pred.squeeze(), batch_targets.squeeze())
                    
                    if torch.isnan(loss):
                        continue
                    
                    total_loss += loss.item()
                    predictions.extend(pred.squeeze().cpu().numpy())
                    targets.extend(batch_targets.squeeze().cpu().numpy())
                    successful_batches += 1
                    
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    continue
        
        if successful_batches == 0:
            raise RuntimeError("No successful validation batches!")
        
        avg_loss = total_loss / successful_batches
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def train(self, train_dataset, val_dataset, epochs=30, batch_size=8, day_folder=None, **training_params):
        """Main training loop with organized outputs"""
        print(f"🚀 Starting CapsNet training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Save experiment configuration
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'training_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'feature_dim': self.feature_dim,
            'input_size': self.input_size,
            'device': str(self.device),
            **training_params
        }
        
        if day_folder:
            self.output_manager.save_experiment_info(day_folder, config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=custom_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=custom_collate_fn
        )
        
        # Reset training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Training loop
        batch_images = batch_images.to(self.device)
        batch_targets = batch_targets.to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        patience = training_params.get('early_stopping_patience', 10)
        gradient_clip_norm = training_params.get('gradient_clip_norm', 1.0)
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            try:
                # Training
                train_loss, train_metrics = self.train_epoch(train_loader, gradient_clip_norm)
                
                # Validation
                val_loss, val_metrics = self.validate_epoch(val_loader)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Save metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_metrics.append(train_metrics)
                self.val_metrics.append(val_metrics)
                
                # Print results
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
                print(f"Train R²: {train_metrics['r2']:.4f} | Val R²: {val_metrics['r2']:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model with organized path
                    best_model_path = self.output_manager.get_path(
                        'models', 'best', f'best_capsnet_{day_folder}.pth', day_folder
                    )
                    self.save_model(best_model_path, epoch, val_loss)
                    print("✅ New best model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs")
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
                
                # Save epoch checkpoint
                if epoch % 5 == 0:  # Save every 5 epochs
                    checkpoint_path = self.output_manager.get_path(
                        'checkpoints', 'epoch', f'checkpoint_epoch_{epoch+1}_{day_folder}.pth', day_folder
                    )
                    self.save_model(checkpoint_path, epoch, val_loss)
                
            except Exception as e:
                print(f"❌ Error in epoch {epoch+1}: {e}")
                traceback.print_exc()
                continue
        
        # Save final model
        final_model_path = self.output_manager.get_path(
            'models', 'final', f'final_capsnet_{day_folder}.pth', day_folder
        )
        self.save_model(final_model_path, epochs, best_val_loss)
        
        print("\n🎉 Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Create summary report
        results = {
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.train_losses),
            'final_train_loss': self.train_losses[-1] if self.train_losses else 'N/A',
            'final_val_loss': self.val_losses[-1] if self.val_losses else 'N/A',
            'best_model_path': best_model_path,
            'final_model_path': final_model_path
        }
        
        if day_folder:
            self.output_manager.create_summary_report(day_folder, results)
        
        return best_val_loss
    
    def extract_features(self, dataset, batch_size=16, day_folder=None, split_name='train'):
        """Extract features using trained model with organized outputs"""
        print(f"🔍 Extracting features from {len(dataset)} samples...")
        
        self.feature_extractor.eval()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=custom_collate_fn
        )
        
        all_features = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_images, batch_targets, batch_metadata in tqdm(dataloader, desc="Extracting features"):
                try:
                    batch_images = batch_images.to(self.device)
                    
                    if batch_images.size(0) == 0:
                        continue
                    
                    features = self.feature_extractor(batch_images)
                    all_features.extend(features.cpu().numpy())
                    all_metadata.extend(batch_metadata)
                    
                except Exception as e:
                    print(f"❌ Error extracting features: {e}")
                    continue
        
        print(f"✅ Extracted {len(all_features)} feature vectors")
        
        # Save features with organized paths
        if day_folder:
            features_path = self.output_manager.get_path(
                'features', split_name, f'capsnet_{split_name}_features_{day_folder}.npy', day_folder
            )
            metadata_path = self.output_manager.get_path(
                'metadata', split_name, f'capsnet_{split_name}_metadata_{day_folder}.csv', day_folder
            )
            
            # Save features
            np.save(features_path, all_features)
            pd.DataFrame(all_metadata).to_csv(metadata_path, index=False)
            
            print(f"💾 Features saved: {features_path}")
            print(f"💾 Metadata saved: {metadata_path}")
        
        return all_features, all_metadata
    
    def save_model(self, filepath, epoch, val_loss):
        """Save trained model"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.feature_extractor.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'input_size': self.input_size,
            'feature_dim': self.feature_dim
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 Model loaded from {filepath}")
        return checkpoint
    
    def plot_training_history(self, save_path=None, day_folder=None):
        """Plot training history with organized output"""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        # Use organized path if not specified
        if save_path is None and day_folder:
            save_path = self.output_manager.get_path(
                'plots', 'training', f'training_history_{day_folder}.png', day_folder
            )
        elif save_path is None:
            save_path = 'capsnet_training_history.png'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RMSE
        train_rmse = [m['rmse'] for m in self.train_metrics]
        val_rmse = [m['rmse'] for m in self.val_metrics]
        axes[0, 1].plot(train_rmse, label='Train RMSE', color='blue')
        axes[0, 1].plot(val_rmse, label='Val RMSE', color='red')
        axes[0, 1].set_title('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAE
        train_mae = [m['mae'] for m in self.train_metrics]
        val_mae = [m['mae'] for m in self.val_metrics]
        axes[1, 0].plot(train_mae, label='Train MAE', color='blue')
        axes[1, 0].plot(val_mae, label='Val MAE', color='red')
        axes[1, 0].set_title('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # R²
        train_r2 = [m['r2'] for m in self.train_metrics]
        val_r2 = [m['r2'] for m in self.val_metrics]
        axes[1, 1].plot(train_r2, label='Train R²', color='blue')
        axes[1, 1].plot(val_r2, label='Val R²', color='red')
        axes[1, 1].set_title('R² Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved as '{save_path}'")
        plt.show()
    
    # TESTING FUNCTIONALITY
    def quick_test(self, day_folder, max_samples=50, max_epochs=3):
        """Quick test of the complete pipeline"""
        print(f"\n🧪 Quick Test for {day_folder}")
        print("=" * 50)
        
        try:
            # Test data loading
            print("1. Testing data loading...")
            train_dataset, val_dataset, _, _ = self.prepare_data(day_folder)
            
            if max_samples and len(train_dataset) > max_samples:
                # Create smaller dataset for testing
                indices = torch.randperm(len(train_dataset))[:max_samples]
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            print(f"   ✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
            
            # Test model creation
            print("2. Testing model creation...")
            self.create_model()
            self.setup_training()
            print("   ✅ Model created successfully")
            
            # Test training loop
            print("3. Testing training loop...")
            best_loss = self.train(train_dataset, val_dataset, epochs=max_epochs, batch_size=4, day_folder=day_folder)
            print(f"   ✅ Training completed, best loss: {best_loss:.4f}")
            
            # Test feature extraction
            print("4. Testing feature extraction...")
            features, metadata = self.extract_features(train_dataset, batch_size=4, day_folder=day_folder, split_name='test')
            print(f"   ✅ Features extracted: {len(features)} samples")
            
            print("\n🎯 Quick test PASSED! Pipeline is working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Quick test FAILED: {e}")
            traceback.print_exc()
            return False
    
    def quick_test_trunk(self, day_folder, max_samples=50, max_epochs=3, trunk_size=1000):
        """Quick test of the trunk-based pipeline"""
        print(f"\n🧪 Quick Trunk Test for {day_folder}")
        print("=" * 50)
        
        try:
            # Test trunk data loading
            print("1. Testing trunk data loading...")
            trunk_train_dataset, val_dataset, _, _ = self.prepare_trunk_data(
                day_folder, trunk_size=trunk_size
            )
            
            # Limit validation dataset for testing
            if len(val_dataset) > max_samples:
                indices = torch.randperm(len(val_dataset))[:max_samples]
                val_dataset = torch.utils.data.Subset(val_dataset, indices)
            
            print(f"   ✅ Data loaded: {trunk_train_dataset.total_samples} train (in {trunk_train_dataset.get_trunk_count()} trunks), {len(val_dataset)} val")
            
            # Test model creation
            print("2. Testing model creation...")
            self.create_model()
            self.setup_training()
            print("   ✅ Model created successfully")
            
            # Test trunk training loop - simplified for testing
            print("3. Testing trunk training loop...")
            
            # Create data loaders for testing
            train_loader = DataLoader(
                trunk_train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                collate_fn=custom_collate_fn,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
            
            # Simple training test
            for epoch in range(max_epochs):
                train_loss, train_metrics = self.train_epoch(train_loader, 1.0)
                val_loss, val_metrics = self.validate_epoch(val_loader)
                print(f"   Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            print(f"   ✅ Trunk training completed, final val loss: {val_loss:.4f}")
            
            # Test feature extraction
            print("4. Testing feature extraction...")
            # Use first trunk for feature extraction test
            features, metadata = self.extract_features(trunk_train_dataset, batch_size=4, day_folder=day_folder, split_name='trunk_test')
            print(f"   ✅ Features extracted: {len(features)} samples")
            
            print("\n🎯 Quick trunk test PASSED! Pipeline is working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Quick trunk test FAILED: {e}")
            traceback.print_exc()
            return False
    
    # HYPERPARAMETER TUNING FUNCTIONALITY
    def tune_hyperparameters(self, day_folder, n_trials=50, max_epochs=20, batch_size=8, timeout=3600, enable_pruning=True, study_name=None):
        """Basic hyperparameter tuning using Optuna"""
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna is not available. Install with: pip install optuna")
            return None

        print(f"🔧 Starting basic hyperparameter tuning")
        print(f"   Trials: {n_trials}")
        print(f"   Max epochs per trial: {max_epochs}")
        print(f"   Timeout: {timeout}s ({timeout/3600:.1f}h)")
        print(f"   Pruning enabled: {enable_pruning}")
        
        # Prepare data once
        train_dataset, val_dataset, _, _ = self.prepare_data(day_folder)
        
        def objective(trial):
            try:
                # Suggest basic hyperparameters
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
                feature_dim = trial.suggest_categorical('feature_dim', [64, 128, 256, 512])
                optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'adamw'])
                weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
                batch_size_trial = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
                
                # Create model with suggested parameters
                self.feature_dim = feature_dim
                self.create_model(dropout_rate=dropout_rate)
                self.setup_training(
                    learning_rate=learning_rate, 
                    weight_decay=weight_decay,
                    optimizer_type=optimizer_type
                )
                
                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size_trial, 
                    shuffle=True, 
                    collate_fn=custom_collate_fn,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size_trial, 
                    shuffle=False, 
                    collate_fn=custom_collate_fn
                )
                
                # Training loop with pruning
                best_val_loss = float('inf')
                for epoch in range(max_epochs):
                    train_loss, _ = self.train_epoch(train_loader)
                    val_loss, _ = self.validate_epoch(val_loader)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    
                    # Report intermediate value for pruning
                    if enable_pruning:
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                
                return best_val_loss
                
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        # Create study
        if enable_pruning:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        else:
            pruner = None
            
        sampler = TPESampler(seed=42)
        
        if study_name:
            study_name = f"{study_name}_{day_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            study_name = f"capsnet_basic_tuning_{day_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            sampler=sampler,
            study_name=study_name
        )
        
        # Optimize with timeout
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Save results
        self._save_tuning_results(study, day_folder, 'basic')
        
        print(f"\n🏆 Basic tuning completed!")
        print(f"   Best trial: {study.best_trial.number}")
        print(f"   Best value: {study.best_trial.value:.4f}")
        print(f"   Best params: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def tune_hyperparameters_advanced(self, day_folder, n_trials=100, max_epochs=25, batch_size=8, 
                                    tune_architecture=True, tune_data_augmentation=True, 
                                    tune_regularization=True, timeout=7200, enable_pruning=True, study_name=None):
        """Advanced hyperparameter tuning with architecture and data augmentation parameters"""
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna is not available. Install with: pip install optuna")
            return None

        print(f"🔧 Starting advanced hyperparameter tuning")
        print(f"   Trials: {n_trials}")
        print(f"   Max epochs per trial: {max_epochs}")
        print(f"   Timeout: {timeout}s ({timeout/3600:.1f}h)")
        print(f"   Architecture tuning: {tune_architecture}")
        print(f"   Data augmentation tuning: {tune_data_augmentation}")
        print(f"   Regularization tuning: {tune_regularization}")
        
        # Prepare data once
        train_dataset, val_dataset, _, _ = self.prepare_data(day_folder)
        
        def objective(trial):
            try:
                # Basic hyperparameters
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'sgd'])
                batch_size_trial = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
                
                # Architecture parameters
                if tune_architecture:
                    feature_dim = trial.suggest_categorical('feature_dim', [64, 128, 256, 512])
                    # Could add more architecture parameters here like number of capsule layers, etc.
                else:
                    feature_dim = self.feature_dim
                
                # Regularization parameters
                if tune_regularization:
                    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
                    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
                    gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.5, 2.0)
                else:
                    dropout_rate = 0.3
                    weight_decay = 1e-4
                    gradient_clip_norm = 1.0
                
                # Training parameters
                early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 15)
                
                # Create model with suggested parameters
                self.feature_dim = feature_dim
                self.create_model(dropout_rate=dropout_rate)
                self.setup_training(
                    learning_rate=learning_rate, 
                    weight_decay=weight_decay,
                    optimizer_type=optimizer_type
                )
                
                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size_trial, 
                    shuffle=True, 
                    collate_fn=custom_collate_fn,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size_trial, 
                    shuffle=False, 
                    collate_fn=custom_collate_fn
                )
                
                # Training loop with advanced parameters
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(max_epochs):
                    train_loss, _ = self.train_epoch(train_loader, gradient_clip_norm)
                    val_loss, _ = self.validate_epoch(val_loader)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= early_stopping_patience:
                        break
                    
                    # Report intermediate value for pruning
                    if enable_pruning:
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                
                return best_val_loss
                
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                print(f"Advanced trial failed: {e}")
                return float('inf')
        
        # Create advanced study
        if enable_pruning:
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=8)
        else:
            pruner = None
            
        sampler = TPESampler(seed=42, n_startup_trials=20)
        
        if study_name:
            study_name = f"{study_name}_advanced_{day_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            study_name = f"capsnet_advanced_tuning_{day_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            sampler=sampler,
            study_name=study_name
        )
        
        # Optimize with timeout
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Save results
        self._save_tuning_results(study, day_folder, 'advanced')
        
        print(f"\n🏆 Advanced tuning completed!")
        print(f"   Best trial: {study.best_trial.number}")
        print(f"   Best value: {study.best_trial.value:.4f}")
        print(f"   Best params: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def _save_tuning_results(self, study, day_folder, tuning_type='basic'):
        """Save hyperparameter tuning results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best parameters
        best_params_path = self.output_manager.get_path(
            'hyperparameters', tuning_type, f'best_params_{tuning_type}_{day_folder}_{timestamp}.json', day_folder
        )
        with open(best_params_path, 'w') as f:
            json.dump(study.best_trial.params, f, indent=2)
        
        # Save study results
        study_results_path = self.output_manager.get_path(
            'tuning', 'results', f'study_results_{tuning_type}_{day_folder}_{timestamp}.json', day_folder
        )
        
        # Convert study to serializable format
        study_data = {
            'study_name': study.study_name,
            'direction': study.direction.name,
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_trial.value,
                'params': study.best_trial.params,
                'datetime_start': study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
                'datetime_complete': study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None
            },
            'n_trials': len(study.trials),
            'trials': []
        }
        
        # Add trial information
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            study_data['trials'].append(trial_data)
        
        with open(study_results_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        # Create tuning plots
        self._plot_tuning_results(study, day_folder, tuning_type, timestamp)
        
        print(f"💾 Tuning results saved:")
        print(f"   Best params: {best_params_path}")
        print(f"   Study results: {study_results_path}")
    
    def _plot_tuning_results(self, study, day_folder, tuning_type, timestamp):
        """Create plots for hyperparameter tuning results"""
        try:
            import optuna.visualization as vis
            
            # Optimization history
            fig = vis.plot_optimization_history(study)
            history_path = self.output_manager.get_path(
                'plots', 'tuning', f'optimization_history_{tuning_type}_{day_folder}_{timestamp}.html', day_folder
            )
            fig.write_html(history_path)
            
            # Parameter importance
            if len(study.trials) > 10:  # Need enough trials for importance
                fig = vis.plot_param_importances(study)
                importance_path = self.output_manager.get_path(
                    'plots', 'tuning', f'param_importance_{tuning_type}_{day_folder}_{timestamp}.html', day_folder
                )
                fig.write_html(importance_path)
            
            # Parallel coordinate plot
            if len(study.best_trial.params) > 1:
                fig = vis.plot_parallel_coordinate(study)
                parallel_path = self.output_manager.get_path(
                    'plots', 'tuning', f'parallel_coordinate_{tuning_type}_{day_folder}_{timestamp}.html', day_folder
                )
                fig.write_html(parallel_path)
            
            print(f"📊 Tuning plots saved in plots/tuning/")
            
        except ImportError:
            print("⚠️ Optuna visualization not available. Install with: pip install optuna[visualization]")
        except Exception as e:
            print(f"⚠️ Could not create tuning plots: {e}")
