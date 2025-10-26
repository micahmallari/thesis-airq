"""CNN Baseline Feature Extractor using Pretrained Models
For comparison with CapsNet performance"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, VGG16_Weights

class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor using pretrained models as backbone
    """
    
    def __init__(self,
                 backbone='resnet50',
                 feature_dim=128,
                 pretrained=True,
                 freeze_backbone=False,
                 dropout_rate=0.3):
        super(CNNFeatureExtractor, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        
        # Create backbone
        self.backbone, backbone_features = self._create_backbone(backbone, pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"🔒 Backbone '{backbone}' frozen")
        
        # Feature extraction head
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, feature_dim)
        )
        
        # Initialize new layers
        self._initialize_head()
        
        print(f"🏗️ CNN Feature Extractor created:")
        print(f"   Backbone: {backbone}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Frozen: {freeze_backbone}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Backbone features: {backbone_features}")
    
    def _create_backbone(self, backbone, pretrained):
        """Create backbone network"""
        
        if backbone == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50(weights=None)
            
            # Remove final classification layers
            backbone = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
            backbone_features = 2048
            
        elif backbone == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
            
            backbone = nn.Sequential(*list(model.children())[:-2])
            backbone_features = 512
            
        elif backbone == 'efficientnet_b0':
            if pretrained:
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=None)
            
            # Remove classifier
            backbone = model.features
            backbone_features = 1280
            
        elif backbone == 'vgg16':
            if pretrained:
                model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            else:
                model = models.vgg16(weights=None)
            
            backbone = model.features
            backbone_features = 512
            
        elif backbone == 'mobilenet_v2':
            if pretrained:
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                model = models.mobilenet_v2(weights=None)
            
            backbone = model.features
            backbone_features = 1280
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return backbone, backbone_features
    
    def _initialize_head(self):
        """Initialize the feature extraction head"""
        for m in self.feature_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply feature head
        features = self.feature_head(features)
        
        return features
    
    def get_backbone_features(self, x):
        """Get intermediate backbone features (for analysis)"""
        with torch.no_grad():
            backbone_features = self.backbone(x)
            return backbone_features

class MultiScaleCNNFeatureExtractor(nn.Module):
    """
    Multi-scale CNN Feature Extractor
    Combines features from multiple scales for richer representation
    """
    
    def __init__(self,
                 backbone='resnet50',
                 feature_dim=128,
                 pretrained=True,
                 dropout_rate=0.3):
        super(MultiScaleCNNFeatureExtractor, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Create backbone
        if backbone == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50(weights=None)
            
            # Extract different layers for multi-scale features
            self.layer1 = nn.Sequential(*list(model.children())[:5])   # 256 channels
            self.layer2 = nn.Sequential(*list(model.children())[5:6])  # 512 channels  
            self.layer3 = nn.Sequential(*list(model.children())[6:7])  # 1024 channels
            self.layer4 = nn.Sequential(*list(model.children())[7:8])  # 2048 channels
            
            # Feature dimensions from each scale
            scale_dims = [256, 512, 1024, 2048]
            
        else:
            raise ValueError(f"Multi-scale not implemented for {backbone}")
        
        # Adaptive pooling for each scale
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((4, 4)),  # Scale 1: 4x4
            nn.AdaptiveAvgPool2d((2, 2)),  # Scale 2: 2x2  
            nn.AdaptiveAvgPool2d((1, 1)),  # Scale 3: 1x1
            nn.AdaptiveAvgPool2d((1, 1)),  # Scale 4: 1x1
        ])
        
        # Calculate total features
        total_features = scale_dims[0] * 16 + scale_dims[1] * 4 + scale_dims[2] * 1 + scale_dims[3] * 1
        
        # Feature fusion head
        self.feature_head = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, feature_dim)
        )
        
        self._initialize_head()
        
        print(f"🏗️ Multi-Scale CNN Feature Extractor created:")
        print(f"   Backbone: {backbone}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Total backbone features: {total_features}")
    
    def _initialize_head(self):
        """Initialize the feature head"""
        for m in self.feature_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with multi-scale feature extraction"""
        # Extract features at different scales
        x1 = self.layer1(x)      # Early features
        x2 = self.layer2(x1)     # Mid-level features
        x3 = self.layer3(x2)     # High-level features  
        x4 = self.layer4(x3)     # Deep features
        
        # Pool each scale to different sizes
        f1 = self.adaptive_pools[0](x1).flatten(1)  # 256 * 16 = 4096
        f2 = self.adaptive_pools[1](x2).flatten(1)  # 512 * 4 = 2048
        f3 = self.adaptive_pools[2](x3).flatten(1)  # 1024 * 1 = 1024
        f4 = self.adaptive_pools[3](x4).flatten(1)  # 2048 * 1 = 2048
        
        # Concatenate multi-scale features
        features = torch.cat([f1, f2, f3, f4], dim=1)
        
        # Apply feature head
        features = self.feature_head(features)
        
        return features

def create_cnn_baseline(backbone='resnet50',
                       feature_dim=128,
                       pretrained=True,
                       freeze_backbone=False,
                       multi_scale=False,
                       **kwargs):
    """
    Create CNN baseline feature extractor
    
    Args:
        backbone: Backbone architecture ('resnet50', 'resnet18', 'efficientnet_b0', 'vgg16', 'mobilenet_v2')
        feature_dim: Output feature dimension
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone parameters
        multi_scale: Use multi-scale feature extraction
        **kwargs: Additional parameters
    
    Returns:
        CNN feature extractor model
    """
    if multi_scale:
        return MultiScaleCNNFeatureExtractor(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            **kwargs
        )
    else:
        return CNNFeatureExtractor(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

class CNNEnsembleFeatureExtractor(nn.Module):
    """
    Ensemble of multiple CNN backbones for robust feature extraction
    """
    
    def __init__(self,
                 backbones=['resnet50', 'efficientnet_b0'],
                 feature_dim=128,
                 pretrained=True,
                 fusion_method='concat'):
        super(CNNEnsembleFeatureExtractor, self).__init__()
        
        self.backbones = backbones
        self.fusion_method = fusion_method
        
        # Create individual extractors
        self.extractors = nn.ModuleList()
        individual_dim = feature_dim // len(backbones) if fusion_method == 'concat' else feature_dim
        
        for backbone in backbones:
            extractor = create_cnn_baseline(
                backbone=backbone,
                feature_dim=individual_dim,
                pretrained=pretrained
            )
            self.extractors.append(extractor)
        
        # Fusion layer (if needed)
        if fusion_method == 'concat':
            total_dim = individual_dim * len(backbones)
            if total_dim != feature_dim:
                self.fusion_layer = nn.Linear(total_dim, feature_dim)
            else:
                self.fusion_layer = nn.Identity()
        elif fusion_method == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(len(backbones)) / len(backbones))
            self.fusion_layer = nn.Identity()
        else:  # average
            self.fusion_layer = nn.Identity()
        
        print(f"🏗️ CNN Ensemble Feature Extractor created:")
        print(f"   Backbones: {backbones}")
        print(f"   Fusion method: {fusion_method}")
        print(f"   Feature dim: {feature_dim}")
    
    def forward(self, x):
        """Forward pass through ensemble"""
        # Extract features from each backbone
        features_list = []
        for extractor in self.extractors:
            features = extractor(x)
            features_list.append(features)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused_features = torch.cat(features_list, dim=1)
            fused_features = self.fusion_layer(fused_features)
        elif self.fusion_method == 'attention':
            # Weighted combination using learned attention
            weights = F.softmax(self.attention_weights, dim=0)
            fused_features = sum(w * f for w, f in zip(weights, features_list))
        else:  # average
            fused_features = torch.stack(features_list, dim=0).mean(dim=0)
        
        return fused_features

# Utility functions for model comparison
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def compare_models():
    """Compare different CNN baseline architectures"""
    print("🔍 CNN Baseline Model Comparison")
    print("=" * 60)
    
    backbones = ['resnet18', 'resnet50', 'efficientnet_b0', 'vgg16', 'mobilenet_v2']
    feature_dim = 128
    
    results = []
    
    for backbone in backbones:
        try:
            model = create_cnn_baseline(
                backbone=backbone,
                feature_dim=feature_dim,
                pretrained=True
            )
            
            total_params, trainable_params = count_parameters(model)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                output = model(dummy_input)
            
            results.append({
                'backbone': backbone,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'output_shape': output.shape,
                'success': True
            })
            
            print(f"✅ {backbone:15} | Params: {total_params:8,} | Trainable: {trainable_params:8,} | Output: {output.shape}")
            
        except Exception as e:
            results.append({
                'backbone': backbone,
                'error': str(e),
                'success': False
            })
            print(f"❌ {backbone:15} | Error: {e}")
    
    print("\n📊 Recommendations:")
    print("   • ResNet18: Lightweight, good baseline")
    print("   • ResNet50: Balanced performance/complexity")
    print("   • EfficientNet-B0: Best efficiency")
    print("   • MobileNet-V2: Fastest inference")
    
    return results

if __name__ == "__main__":
    # Run model comparison
    compare_models()
    
    # Test multi-scale extractor
    print(f"\n🔍 Testing Multi-Scale CNN:")
    multi_scale_model = create_cnn_baseline(
        backbone='resnet50',
        feature_dim=128,
        multi_scale=True
    )
    
    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = multi_scale_model(dummy_input)
    print(f"   Multi-scale output: {output.shape}")
    
    # Test ensemble
    print(f"\n🔍 Testing CNN Ensemble:")
    ensemble_model = CNNEnsembleFeatureExtractor(
        backbones=['resnet18', 'efficientnet_b0'],
        feature_dim=128,
        fusion_method='concat'
    )
    
    with torch.no_grad():
        output = ensemble_model(dummy_input)
    print(f"   Ensemble output: {output.shape}")
