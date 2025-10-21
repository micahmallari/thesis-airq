"""CNN Baseline Trainer - Integrated with CapsNet training system"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.capsnet_trainer import CapsNetTrainer, OutputManager
from models.cnn_baseline import create_cnn_baseline
import torch.nn as nn

class CNNBaselineTrainer(CapsNetTrainer):
    """CNN Baseline Trainer that inherits from CapsNetTrainer for consistency"""
    
    def __init__(self, input_size=256, feature_dim=128, device='cuda', backbone='resnet50', **kwargs):
        # Initialize parent class but override model type
        super().__init__(input_size, feature_dim, device, model_type='cnn_baseline')
        self.backbone = backbone
        self.cnn_kwargs = kwargs
        
        print(f"🏗️ CNN Baseline Trainer initialized")
        print(f"   Backbone: {backbone}")
        print(f"   Feature dimension: {feature_dim}")
    
    def create_model(self, **model_params):
        """Create CNN baseline model"""
        # Merge kwargs
        merged_params = {**self.cnn_kwargs, **model_params}
        
        self.feature_extractor = create_cnn_baseline(
            backbone=self.backbone,
            feature_dim=self.feature_dim,
            **merged_params
        ).to(self.device)
        
        # Temporary regression head for training (same as CapsNet)
        dropout_rate = merged_params.get('dropout_rate', 0.3)
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
        print(f"   Total CNN parameters: {total_params:,}")
        
        return self.feature_extractor

def create_cnn_trainer(backbone='resnet50', input_size=256, feature_dim=128, device='cuda', **kwargs):
    """Create CNN baseline trainer"""
    return CNNBaselineTrainer(
        input_size=input_size,
        feature_dim=feature_dim,
        device=device,
        backbone=backbone,
        **kwargs
    )