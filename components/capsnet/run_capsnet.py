#!/usr/bin/env python3
"""CapsNet Feature Extractor for Air Quality Prediction
Handles training, testing, and feature extraction for CapsNet models"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Also add the parent directory to handle different import scenarios
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    # Try multiple import approaches
    try:
        from components.capsnet.capsnet_trainer import CapsNetTrainer
    except ImportError:
        from components.capsnet.capsnet_trainer import CapsNetTrainer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Try to add more paths
    possible_paths = [
        os.path.join(os.getcwd(), 'src'),
        os.path.join(os.getcwd(), 'src', 'training'),
        os.path.join(os.getcwd(), 'src', 'models'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
    
    try:
        from components.capsnet.capsnet_trainer import CapsNetTrainer
    except ImportError:
        try:
            from capsnet_trainer import CapsNetTrainer
        except ImportError:
            print("❌ Could not import CapsNetTrainer. Please check your file structure.")
            sys.exit(1)

# Configuration
class Config:
    # Data paths based on your exact folder structure
    DAY_FOLDERS = ['7_24_data', '10_19_data', '11_10_data']
    MATCHED_DATA_DIR = "dataset/c_matched_spatio_temporal_data"
    DATA_SPLIT_DIR = "dataset/d_data_split"
    PREPROCESSED_IMG_DIR = "dataset/e_preprocessed_img"
    FEATURES_OUTPUT_DIR = "dataset/capsnet_features"
    
    # Model parameters
    INPUT_SIZE = 256
    FEATURE_DIM = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training parameters
    EPOCHS = 30
    BATCH_SIZE = 4
    TEST_SIZE = 0.2
    
    @staticmethod
    def create_directories():
        """Create necessary directories"""
        os.makedirs(Config.FEATURES_OUTPUT_DIR, exist_ok=True)
        for day in Config.DAY_FOLDERS:
            os.makedirs(os.path.join(Config.FEATURES_OUTPUT_DIR, day), exist_ok=True)

def quick_test(day_folder, max_samples=50, max_epochs=3):
    """Run quick test to verify pipeline works"""
    print(f"\n=== Quick Test Mode ===") # Should print 'NVIDIA GeForce RTX 3050'
    
    try:
        from test_capsnet import QuickTester
        
        tester = QuickTester(
            day_folder=day_folder,
            max_samples=max_samples,
            max_epochs=max_epochs
        )
        
        success = tester.run_full_test()
        return success
        
    except ImportError:
        print("❌ test_capsnet.py not found. Please ensure it's in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def check_data_availability():
    """Check if required data files exist based on your exact folder structure"""
    print("🔍 Checking data availability...")
    
    missing_files = []
    found_files = []
    
    # Check matched spatio-temporal data
    for day in Config.DAY_FOLDERS:
        day_short = day.replace('_data', '')
        matched_file = os.path.join(Config.MATCHED_DATA_DIR, f"matched_{day_short}.csv")
        if not os.path.exists(matched_file):
            missing_files.append(matched_file)
        else:
            try:
                df = pd.read_csv(matched_file)
                found_files.append(f"{matched_file} ({len(df)} entries)")
            except Exception as e:
                missing_files.append(f"{matched_file} (error: {e})")
    
    # Check data split files
    for day in Config.DAY_FOLDERS:
        for split_type in ['learning.csv', 'test.csv']:
            split_file = os.path.join(Config.DATA_SPLIT_DIR, day, split_type)
            if not os.path.exists(split_file):
                missing_files.append(split_file)
            else:
                try:
                    df = pd.read_csv(split_file)
                    found_files.append(f"{split_file} ({len(df)} entries)")
                except Exception as e:
                    missing_files.append(f"{split_file} (error: {e})")
        
        # Check preprocessed image directory
        img_dir = os.path.join(Config.PREPROCESSED_IMG_DIR, day, 'patch')
        if not os.path.exists(img_dir):
            missing_files.append(img_dir)
        else:
            npy_count = len([f for f in os.listdir(img_dir) if f.endswith('.npy')])
            found_files.append(f"{img_dir} ({npy_count} .npy files)")
    
    # Check patch metadata
    metadata_file = os.path.join(Config.PREPROCESSED_IMG_DIR, 'patch_metadata.csv')
    if not os.path.exists(metadata_file):
        missing_files.append(metadata_file)
    else:
        try:
            df = pd.read_csv(metadata_file)
            found_files.append(f"{metadata_file} ({len(df)} patches)")
        except Exception as e:
            missing_files.append(f"{metadata_file} (error: {e})")
    
    print(f"\n✅ Found {len(found_files)} data sources:")
    for file in found_files:
        print(f"  - {file}")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} data sources:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("\n🎉 All required data files found!")
        return True

def train_feature_extractor(day_folder):
    """Train CapsNet feature extractor for a specific day"""
    print(f"\n=== Training CapsNet Feature Extractor for {day_folder} ===")
    
    # Create directories
    Config.create_directories()
    
    try:
        # Initialize trainer
        trainer = CapsNetTrainer(
            input_size=Config.INPUT_SIZE,
            feature_dim=Config.FEATURE_DIM,
            device=Config.DEVICE
        )
        
        # Prepare data
        print("Preparing data...")
        train_dataset, val_dataset, learning_df, patch_metadata_df = trainer.prepare_data(
            day_folder=day_folder,
            test_size=Config.TEST_SIZE
        )
        
        print(f"✅ Data prepared successfully!")
        print(f"   Train dataset size: {len(train_dataset)}")
        print(f"   Validation dataset size: {len(val_dataset)}")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("❌ No data found! Check your data structure.")
            return False
        
        # Train feature extractor
        print("Starting feature extractor training...")
        trainer.train(
            train_dataset, 
            val_dataset,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE
        )
        
        # Plot results
        trainer.plot_training_history()
        
        print("✅ Feature extractor training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_cross_validation(day_folder):
    """Run 5-fold cross-validation for a specific day"""
    print(f"\n=== Running Cross-Validation for {day_folder} ===")
    
    try:
        # Initialize trainer
        trainer = CapsNetTrainer(
            input_size=Config.INPUT_SIZE,
            feature_dim=Config.FEATURE_DIM,
            device=Config.DEVICE
        )
        
        # Run cross-validation
        fold_results, best_model_path = trainer.run_cross_validation(
            day_folder=day_folder,
            n_folds=5,
            epochs=25,  # Reduced epochs for CV
            batch_size=Config.BATCH_SIZE
        )
        
        print("✅ Cross-validation completed successfully!")
        print(f"Best model saved as: {best_model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_features_for_day(day_folder, model_path=None, split_type='learning'):
    """Extract features for all patches in a specific day"""
    print(f"\n=== Extracting Features for {day_folder} ({split_type}) ===")
    
    # Use best model if no path specified
    if model_path is None:
        model_path = "best_capsnet_feature_extractor.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the feature extractor first.")
        return False
    
    try:
        # Initialize feature extractor
        feature_extractor = CapsNetTrainer.create_model(
            input_channels=3,
            input_size=Config.INPUT_SIZE,
            feature_dim=Config.FEATURE_DIM
        ).to(Config.DEVICE)
        
        # Load trained weights
        checkpoint = torch.load(model_path)
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        feature_extractor.eval()
        
        print("✅ Feature extractor loaded successfully!")
        
        # Load day data
        trainer = CapsNetTrainer()
        learning_df, patch_metadata_df = trainer.load_day_data(day_folder)
        
        # Load test data if requested
        if split_type == 'test':
            test_path = f"dataset/d_data_split/{day_folder}/test.csv"
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                print(f"📊 Test data loaded: {len(test_df)} entries")
                dataset = CapsNetTrainer.AirQualityDataset(test_df, patch_metadata_df, day_folder, 'test')
            else:
                print(f"⚠️  Test file not found: {test_path}")
                return False
        else:
            # Use learning data
            dataset = CapsNetTrainer.AirQualityDataset(learning_df, patch_metadata_df, day_folder, 'learning')
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Extract features
        all_features = []
        all_metadata = []
        
        print(f"Extracting features from {len(dataset)} patches...")
        
        with torch.no_grad():
            for batch_images, batch_targets, batch_metadata in tqdm(dataloader, desc="Extracting"):
                batch_images = batch_images.to(Config.DEVICE)
                
                # Extract features
                features = feature_extractor(batch_images)
                
                all_features.extend(features.cpu().numpy())
                all_metadata.extend(batch_metadata)
        
        # Create features DataFrame
        features_df = pd.DataFrame(all_features, columns=[f'capsnet_feature_{i}' for i in range(Config.FEATURE_DIM)])
        
        # Add metadata
        for key in ['image_filename', 'timestamp', 'pm2.5', 'pm10', 'temperature', 'humidity', 'location', 'patch_idx', 'augmentations', 'npy_path']:
            if key in all_metadata[0]:
                features_df[key] = [meta[key] for meta in all_metadata]
        
        # Save features
        output_path = os.path.join(Config.FEATURES_OUTPUT_DIR, day_folder, f'capsnet_features_{split_type}.csv')
        features_df.to_csv(output_path, index=False)
        
        print(f"✅ Saved {len(features_df)} feature vectors to {output_path}")
        print(f"   Feature dimensions: {Config.FEATURE_DIM}")
        print(f"   Unique images: {features_df['image_filename'].nunique()}")
        print(f"   Total patches: {len(features_df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_features_all_days(model_path=None):
    """Extract features for all day folders (both learning and test)"""
    print(f"\n=== Extracting Features for All Days ===")
    
    success_count = 0
    total_tasks = len(Config.DAY_FOLDERS) * 2  # learning + test for each day
    
    for day_folder in Config.DAY_FOLDERS:
        for split_type in ['learning', 'test']:
            if extract_features_for_day(day_folder, model_path, split_type):
                success_count += 1
            else:
                print(f"⚠️  Failed to extract features for {day_folder} ({split_type})")
    
    print(f"\n📊 Feature extraction completed: {success_count}/{total_tasks} tasks successful")
    return success_count == total_tasks

def check_extracted_features():
    """Check extracted features for all days"""
    print("\n=== Checking Extracted Features ===")
    
    for day_folder in Config.DAY_FOLDERS:
        for split_type in ['learning', 'test']:
            feature_file = os.path.join(Config.FEATURES_OUTPUT_DIR, day_folder, f'capsnet_features_{split_type}.csv')
            if os.path.exists(feature_file):
                try:
                    df = pd.read_csv(feature_file)
                    print(f"✅ {day_folder} ({split_type}): {len(df)} patches, {Config.FEATURE_DIM} features")
                    print(f"   Unique images: {df['image_filename'].nunique()}")
                    if 'pm2.5' in df.columns:
                        print(f"   PM2.5 range: {df['pm2.5'].min():.2f} - {df['pm2.5'].max():.2f}")
                except Exception as e:
                    print(f"❌ {day_folder} ({split_type}): Error reading file - {e}")
            else:
                print(f"❌ {day_folder} ({split_type}): Features not found")

def inspect_day_data(day_folder):
    """Inspect the structure of data for a specific day"""
    print(f"\n=== Inspecting Data Structure for {day_folder} ===")
    
    try:
        # Check matched data
        day_short = day_folder.replace('_data', '')
        matched_path = f"dataset/c_matched_spatio_temporal_data/matched_{day_short}.csv"
        if os.path.exists(matched_path):
            df = pd.read_csv(matched_path)
            print(f"📊 Matched Data ({matched_path}):")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head(2))
        
        # Check data split files
        for split_type in ['learning', 'test']:
            split_path = f"dataset/d_data_split/{day_folder}/{split_type}.csv"
            if os.path.exists(split_path):
                df = pd.read_csv(split_path)
                print(f"\n📋 {split_type.title()} Data ({split_path}):")
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Sample data:")
                print(df.head(2))
        
        # Check patch metadata
        metadata_path = "dataset/e_preprocessed_img/patch_metadata.csv"
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            day_patches = df[df['day'] == day_folder]
            print(f"\n🖼️  Patch Metadata ({metadata_path}):")
            print(f"   Total patches: {len(df)}")
            print(f"   Patches for {day_folder}: {len(day_patches)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data for {day_folder}:")
            print(day_patches.head(2))
            
            # Check if .npy files exist
            if len(day_patches) > 0:
                sample_npy = day_patches['npy_path'].iloc[0]
                full_path = os.path.join('dataset', 'e_preprocessed_img', sample_npy)
                if os.path.exists(full_path):
                    try:
                        data = np.load(full_path)
                        print(f"   Sample .npy shape: {data.shape}")
                        print(f"   Sample .npy range: {data.min():.3f} - {data.max():.3f}")
                    except Exception as e:
                        print(f"   Error loading sample .npy: {e}")
                else:
                    print(f"   ⚠️  Sample .npy file not found: {full_path}")
        
    except Exception as e:
        print(f"❌ Error inspecting data: {e}")

def main():
    parser = argparse.ArgumentParser(description='CapsNet Feature Extractor')
    parser.add_argument('--mode', choices=['train', 'test', 'extract', 'tune', 'tune_advanced', 'trunk', 'trunk_optimized'],
                        required=True, help='Operation mode')
    parser.add_argument('--day', required=True, help='Day folder to process')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    # Test mode specific arguments
    parser.add_argument('--test_samples', type=int, default=50, help='Max samples for testing')
    parser.add_argument('--test_epochs', type=int, default=3, help='Max epochs for testing')
    
    # Tuning specific arguments
    parser.add_argument('--n_trials', type=int, default=50, help='Number of tuning trials')
    parser.add_argument('--tune_epochs', type=int, default=25, help='Max epochs per trial')
    parser.add_argument('--tune_timeout', type=int, default=3600, help='Timeout for tuning in seconds')
    parser.add_argument('--tune_pruning', action='store_true', help='Enable pruning for faster tuning')
    parser.add_argument('--tune_study_name', type=str, default=None, help='Study name for persistent tuning')
    
    # Advanced tuning arguments
    parser.add_argument('--tune_architecture', action='store_true', help='Include architecture parameters in tuning')
    parser.add_argument('--tune_data_augmentation', action='store_true', help='Include data augmentation in tuning')
    parser.add_argument('--tune_regularization', action='store_true', help='Include regularization parameters in tuning')
    
    # Trunk training specific arguments
    parser.add_argument('--trunk_size', type=int, default=10000, help='Samples per trunk for memory management')
    parser.add_argument('--epochs_per_trunk', type=int, default=5, help='Epochs per trunk')
    parser.add_argument('--use_trunk', action='store_true', help='Use trunk-based training for large datasets')
    
    # OPTIMIZATION arguments
    parser.add_argument('--max_patches_per_image', type=int, default=50, help='Max patches per image for optimization')
    parser.add_argument('--optimized_trunk_size', type=int, default=50000, help='Larger trunk size for optimization')
    parser.add_argument('--optimized_batch_size', type=int, default=16, help='Larger batch size for optimization')
    
    # Output organization
    parser.add_argument('--output_dir', default='outputs', help='Base output directory')
    parser.add_argument('--model_type', default='capsnet', help='Model type for organization')
    
    args = parser.parse_args()
    
    print("🚀 CapsNet Feature Extractor") # Should print 'NVIDIA GeForce RTX 3050'
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Day: {args.day}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    if args.use_trunk or args.mode == 'trunk':
        print(f"Trunk size: {args.trunk_size:,}")
        print(f"Epochs per trunk: {args.epochs_per_trunk}")
    if 'tune' in args.mode:
        print(f"Tuning trials: {args.n_trials}")
        print(f"Tuning epochs per trial: {args.tune_epochs}")
        if args.tune_timeout:
            print(f"Tuning timeout: {args.tune_timeout}s ({args.tune_timeout/3600:.1f}h)")
    print("=" * 50)
    
    # Initialize trainer with organized outputs
    trainer = CapsNetTrainer(
        input_size=256,
        feature_dim=args.feature_dim,
        device=args.device,
        model_type=args.model_type
    )
    
    try:
        if args.mode == 'test':
            print("\n🧪 Running Quick Test")
            print("-" * 30)
            
            if args.use_trunk:
                print("Using trunk-based testing...")
                success = trainer.quick_test_trunk(
                    day_folder=args.day,
                    max_samples=args.test_samples,
                    max_epochs=args.test_epochs,
                    trunk_size=min(args.trunk_size, 1000)  # Smaller trunk for testing
                )
            else:
                success = trainer.quick_test(
                    day_folder=args.day,
                    max_samples=args.test_samples,
                    max_epochs=args.test_epochs
                )
            
            if success:
                print("\n✅ Test completed successfully!")
            else:
                print("\n❌ Test failed!")
                sys.exit(1)

        elif args.mode == 'tune':
            print("\n🔧 Basic Hyperparameter Tuning")
            print("-" * 30)
            best_params = trainer.tune_hyperparameters(
                day_folder=args.day,
                n_trials=args.n_trials,
                max_epochs=args.tune_epochs,
                batch_size=args.batch_size,
                timeout=args.tune_timeout,
                enable_pruning=args.tune_pruning,
                study_name=args.tune_study_name
            )
            print(f"\n✅ Basic tuning completed! Best params: {best_params}")
            
        elif args.mode == 'tune_advanced':
            print("\n🔧 Advanced Hyperparameter Tuning")
            print("-" * 30)
            
            # Configure advanced tuning options
            tuning_config = {
                'tune_architecture': args.tune_architecture,
                'tune_data_augmentation': args.tune_data_augmentation,
                'tune_regularization': args.tune_regularization,
                'enable_pruning': args.tune_pruning,
                'timeout': args.tune_timeout,
                'study_name': args.tune_study_name
            }
            
            best_params = trainer.tune_hyperparameters_advanced(
                day_folder=args.day,
                n_trials=args.n_trials,
                max_epochs=args.tune_epochs,
                batch_size=args.batch_size,
                **tuning_config
            )
            print(f"\n✅ Advanced tuning completed! Best params: {best_params}")
        
        elif args.mode == 'train':
            print("\n🏋️ Training CapsNet")
            print("-" * 30)
            
            # Create model
            trainer.create_model()
            trainer.setup_training(learning_rate=args.learning_rate)
            
            if args.use_trunk:
                print("Using trunk-based training for large dataset...")
                
                # Prepare trunk data
                trunk_train_dataset, val_dataset, _, _ = trainer.prepare_trunk_data(
                    args.day, trunk_size=args.trunk_size
                )
                
                print(f"Dataset info:")
                print(f"   Total samples: {trunk_train_dataset.total_samples:,}")
                print(f"   Trunks: {trunk_train_dataset.get_trunk_count()}")
                print(f"   Validation samples: {len(val_dataset)}")
                
                # Confirm before starting (for large datasets)
                if trunk_train_dataset.total_samples > 100000:
                    response = input(f"\nThis will process {trunk_train_dataset.total_samples:,} samples. Continue? (y/N): ")
                    if response.lower() != 'y':
                        print("Training cancelled.")
                        sys.exit(0)
                
                # Train with trunks - simplified for testing
                print("Starting simplified trunk training...")
                best_loss = float('inf')
                
                # Create data loaders
                from torch.utils.data import DataLoader
                from capsnet_trainer import custom_collate_fn
                
                train_loader = DataLoader(
                    trunk_train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=custom_collate_fn,
                    drop_last=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=custom_collate_fn
                )
                
                # Simple training loop for testing
                for epoch in range(args.epochs_per_trunk):
                    print(f"Epoch {epoch+1}/{args.epochs_per_trunk}")
                    train_loss, train_metrics = trainer.train_epoch(train_loader, 1.0)
                    val_loss, val_metrics = trainer.validate_epoch(val_loader)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                    
                    print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"   Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
                
            else:
                print("Using standard training...")
                
                # Prepare data
                train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
                
                # Train
                best_loss = trainer.train(
                    train_dataset, val_dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    day_folder=args.day
                )
            
            # Plot training history
            trainer.plot_training_history(day_folder=args.day)
            
            print(f"\n✅ Training completed! Best loss: {best_loss:.4f}")
        
        elif args.mode == 'extract':
            print("\n🔍 Extracting Features")
            print("-" * 30)
            
            # Load trained model
            model_path = trainer.output_manager.get_path('models', 'best', f'best_capsnet_{args.day}.pth', args.day)
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = [
                    'best_capsnet_feature_extractor.pth',
                    trainer.output_manager.get_path('models', 'final', f'final_capsnet_{args.day}.pth', args.day)
                ]
                
                model_path = None
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
                
                if model_path is None:
                    print(f"❌ No trained model found. Please train a model first using --mode train")
                    sys.exit(1)
            
            # Create and load model
            trainer.create_model()
            trainer.load_model(model_path)
            
            # Prepare data
            train_dataset, val_dataset, _, _ = trainer.prepare_data(args.day)
            
            # Extract features
            print("Extracting training features...")
            train_features, train_metadata = trainer.extract_features(
                train_dataset, day_folder=args.day, split_name='train'
            )
            
            print("Extracting validation features...")
            val_features, val_metadata = trainer.extract_features(
                val_dataset, day_folder=args.day, split_name='val'
            )
            
            print(f"\n✅ Features extracted and saved!")
            print(f"Training: {len(train_features)} samples")
            print(f"Validation: {len(val_features)} samples")
    
    except Exception as e:
        print(f"\n❌ Error in {args.mode} mode: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
   
