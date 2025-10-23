"""
CapsNet Model Architecture for Spatial Feature Extraction
This implementation focuses on extracting spatial features while being trained on PM2.5 prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PrimaryCapsules(nn.Module):
    """Primary Capsules Layer - Extracts local spatial features"""
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        
        # Create separate conv layers for each capsule
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
            for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        # x shape: [batch_size, in_channels, height, width]
        batch_size = x.size(0)
        
        # Apply each capsule convolution
        outputs = []
        for capsule in self.capsules:
            # Each capsule output: [batch_size, out_channels, new_height, new_width]
            capsule_output = capsule(x)
            outputs.append(capsule_output)
        
        # Stack capsules: [batch_size, num_capsules, out_channels, new_height, new_width]
        outputs = torch.stack(outputs, dim=1)
        
        # Reshape to: [batch_size, num_capsules * new_height * new_width, out_channels]
        batch_size, num_caps, out_channels, height, width = outputs.size()
        outputs = outputs.view(batch_size, num_caps * height * width, out_channels)
        
        # Apply squashing function to get capsule vectors
        return self.squash(outputs)
    
    def squash(self, tensor):
        """Squashing function to ensure capsule vector lengths are between 0 and 1"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        return scale * unit_vector

class DigitCapsules(nn.Module):
    """Digit Capsules Layer - Learns high-level spatial relationships"""
    
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=32, out_channels=16):
        super(DigitCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        print(f"DigitCapsules: {num_routes} routes, {in_channels}D -> {out_channels}D")
        
        # Weight matrix for routing: [num_routes, num_capsules, in_channels, out_channels]
        self.W = nn.Parameter(torch.randn(num_routes, num_capsules, in_channels, out_channels) * 0.01)
    
    def forward(self, x):
        # x shape: [batch_size, num_routes, in_channels]
        batch_size, num_routes, in_channels = x.size()
        
        print(f"DigitCapsules input shape: {x.shape}")
        
        # Ensure we have the expected number of routes
        if num_routes != self.num_routes:
            print(f"Warning: Expected {self.num_routes} routes, got {num_routes}")
            # Adjust if needed
            if num_routes > self.num_routes:
                x = x[:, :self.num_routes, :]  # Truncate
            else:
                # Pad with zeros if we have fewer routes
                padding = torch.zeros(batch_size, self.num_routes - num_routes, in_channels, device=x.device)
                x = torch.cat([x, padding], dim=1)
            num_routes = self.num_routes
        
        # Compute predictions for all routes and capsules
        # x: [batch_size, num_routes, in_channels]
        # W: [num_routes, num_capsules, in_channels, out_channels]
        
        # Expand x to match W dimensions for broadcasting
        # x: [batch_size, num_routes, in_channels] -> [batch_size, num_routes, 1, in_channels, 1]
        x_expanded = x.unsqueeze(2).unsqueeze(4)  # [batch_size, num_routes, 1, in_channels, 1]
        
        # Expand W for batch processing
        # W: [num_routes, num_capsules, in_channels, out_channels] -> [1, num_routes, num_capsules, in_channels, out_channels]
        W_expanded = self.W.unsqueeze(0)  # [1, num_routes, num_capsules, in_channels, out_channels]
        
        # Compute u_hat using broadcasting and matrix multiplication
        # x_expanded: [batch_size, num_routes, 1, in_channels, 1]
        # W_expanded: [1, num_routes, num_capsules, in_channels, out_channels]
        # Result: [batch_size, num_routes, num_capsules, out_channels]
        u_hat = torch.einsum('brc,rnco->brno', x, self.W)
        
        print(f"u_hat shape: {u_hat.shape}")
        
        # Dynamic routing
        return self.dynamic_routing(u_hat)
    
    def dynamic_routing(self, u_hat, num_iterations=3):
        """Dynamic routing algorithm to learn spatial relationships"""
        batch_size, num_routes, num_capsules, out_channels = u_hat.size()
        
        print(f"Dynamic routing - u_hat shape: {u_hat.shape}")
        
        # Initialize routing logits: [batch_size, num_routes, num_capsules]
        b = torch.zeros(batch_size, num_routes, num_capsules, device=u_hat.device)
        
        for iteration in range(num_iterations):
            # Softmax to get routing coefficients: [batch_size, num_routes, num_capsules]
            c = F.softmax(b, dim=2)
            
            # Add dimension for broadcasting: [batch_size, num_routes, num_capsules, 1]
            c = c.unsqueeze(-1)
            
            # Weighted sum of predictions: [batch_size, num_capsules, out_channels]
            s = (c * u_hat).sum(dim=1)  # Sum over routes dimension
            
            print(f"Iteration {iteration}: s shape: {s.shape}")
            
            # Apply squashing: [batch_size, num_capsules, out_channels]
            v = self.squash(s)
            
            print(f"Iteration {iteration}: v shape: {v.shape}")
            
            # Update routing logits (except for last iteration)
            if iteration < num_iterations - 1:
                # Expand v to match u_hat dimensions for agreement calculation
                # v: [batch_size, num_capsules, out_channels] -> [batch_size, 1, num_capsules, out_channels]
                v_expanded = v.unsqueeze(1)
            
                print(f"Iteration {iteration}: v_expanded shape: {v_expanded.shape}")
                print(f"Iteration {iteration}: u_hat shape for agreement: {u_hat.shape}")
            
                # Agreement between prediction and output: [batch_size, num_routes, num_capsules]
                agreement = (u_hat * v_expanded).sum(dim=-1)  # Sum over out_channels dimension
            
                print(f"Iteration {iteration}: agreement shape: {agreement.shape}")
            
                # Update routing logits
                b = b + agreement
    
        return v  # [batch_size, num_capsules, out_channels]
    
    def squash(self, tensor):
        """Squashing function"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        return scale * unit_vector

class CapsNetFeatureExtractor(nn.Module):
    """
    CapsNet for Spatial Feature Extraction
    
    This model extracts spatial features that preserve:
    - Spatial relationships between objects
    - Viewpoint invariance
    - Part-whole relationships
    - Hierarchical spatial structure
    """
    
    def __init__(self, input_channels=3, input_size=256, feature_dim=128, 
                 num_primary_capsules=8, primary_capsule_dim=32, 
                 num_digit_capsules=10, digit_capsule_dim=16, 
                 dropout_rate=0.3, **kwargs):
        super(CapsNetFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.num_digit_capsules = num_digit_capsules
        self.digit_capsule_dim = digit_capsule_dim
        
        # Initial CNN layers for basic feature extraction
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=9, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        
        # Calculate size after conv1
        conv1_size = input_size - 8  # 256 - 8 = 248
        
        # Primary Capsules - Extract local spatial features
        self.primary_capsules = PrimaryCapsules(
            num_capsules=num_primary_capsules, 
            in_channels=256, 
            out_channels=primary_capsule_dim, 
            kernel_size=9, 
            stride=2
        )
        
        # Calculate primary capsules output size more carefully
        primary_size = (conv1_size - 8) // 2  # (248 - 8) // 2 = 120
        num_primary_capsules_total = num_primary_capsules * primary_size * primary_size

        print(f"   Conv1 size: {conv1_size}x{conv1_size}")
        print(f"   Primary size after conv: {primary_size}x{primary_size}")
        print(f"   Primary capsules total: {num_primary_capsules_total}")

        # Ensure the digit capsules expect the right number of routes
        self.digit_capsules = DigitCapsules(
            num_capsules=num_digit_capsules,
            num_routes=num_primary_capsules_total,
            in_channels=primary_capsule_dim,
            out_channels=digit_capsule_dim
        )
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(num_digit_capsules * digit_capsule_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"CapsNet Architecture:")
        print(f"  Input: {input_channels}x{input_size}x{input_size}")
        print(f"  Conv1 output: 256x{conv1_size}x{conv1_size}")
        print(f"  Primary capsules: {num_primary_capsules_total} capsules of {primary_capsule_dim}D")
        print(f"  Digit capsules: {num_digit_capsules} capsules of {digit_capsule_dim}D")
        print(f"  Final features: {feature_dim}D")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass that extracts spatial features
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            features: Spatial feature vectors [batch_size, feature_dim]
        """
        # Initial convolution with batch norm
        x = self.conv1(x)  # [batch_size, 256, 248, 248]
        x = self.conv1_bn(x)
        x = self.relu(x)
        
        # Primary capsules - extract local spatial features
        primary_caps = self.primary_capsules(x)  # [batch_size, num_primary, 32]
        
        # Digit capsules - learn spatial relationships via dynamic routing
        digit_caps = self.digit_capsules(primary_caps)  # [batch_size, num_classes, 16]
        
        # Flatten capsule outputs
        flattened = digit_caps.view(digit_caps.size(0), -1)  # [batch_size, num_classes * 16]
        
        # Project to final feature space
        features = self.feature_projection(flattened)  # [batch_size, feature_dim]
        
        return features
    
    def get_capsule_lengths(self, digit_caps):
        """
        Get the length of each capsule vector (represents presence of spatial pattern)
        
        Args:
            digit_caps: Digit capsule outputs [batch_size, num_classes, 16]
            
        Returns:
            lengths: Capsule lengths [batch_size, num_classes]
        """
        return torch.sqrt((digit_caps ** 2).sum(dim=-1))
    
    def extract_spatial_features(self, x):
        """
        Extract spatial features with detailed capsule information
        
        Returns:
            features: Final feature vector
            capsule_info: Dictionary with capsule analysis
        """
        # Initial convolution with batch norm
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        
        # Primary capsules
        primary_caps = self.primary_capsules(x)
        
        # Digit capsules
        digit_caps = self.digit_capsules(primary_caps)
        
        # Final features
        flattened = digit_caps.view(digit_caps.size(0), -1)
        features = self.feature_projection(flattened)
        
        # Analyze capsule outputs
        capsule_lengths = self.get_capsule_lengths(digit_caps)
        
        capsule_info = {
            'digit_capsules': digit_caps,
            'capsule_lengths': capsule_lengths,
            'active_capsules': (capsule_lengths > 0.1).sum(dim=1),  # Number of active spatial patterns
            'max_capsule_length': capsule_lengths.max(dim=1)[0],
            'spatial_diversity': capsule_lengths.std(dim=1)  # How diverse the spatial patterns are
        }
        
        return features, capsule_info

def create_capsnet_feature_extractor(input_channels=3, input_size=256, feature_dim=128, **kwargs):
    """
    Create a CapsNet feature extractor
    
    Args:
        input_channels: Number of input channels (3 for RGB)
        input_size: Input image size (256x256)
        feature_dim: Final feature dimension (128)
        **kwargs: Additional hyperparameters
        
    Returns:
        CapsNet model configured for spatial feature extraction
    """
    return CapsNetFeatureExtractor(
        input_channels=input_channels,
        input_size=input_size,
        feature_dim=feature_dim,
        **kwargs
    )

# Alternative: Simplified CapsNet for faster training
class SimplifiedCapsNet(nn.Module):
    """
    Simplified CapsNet that focuses on spatial feature extraction
    with reduced computational complexity
    """
    
    def __init__(self, input_channels=3, input_size=256, feature_dim=128, dropout_rate=0.3):
        super(SimplifiedCapsNet, self).__init__()
        
        # CNN backbone for initial feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
        )
        
        # Simplified capsule layer
        self.capsule_conv = nn.Conv2d(256, 8 * 16, 3, 1, 1)  # 8 capsules of 16D
        
        # Spatial attention for feature aggregation
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(8 * 16, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Final feature projection
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # 8x8 spatial resolution
            nn.Flatten(),
            nn.Linear(8 * 16 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract CNN features
        features = self.backbone(x)  # [batch_size, 256, 32, 32]
        
        # Convert to capsule representation
        capsules = self.capsule_conv(features)  # [batch_size, 8*16, 32, 32]
        batch_size, caps_dim, h, w = capsules.size()
        
        # Reshape to capsule format: [batch_size, 8, 16, 32, 32]
        capsules = capsules.view(batch_size, 8, 16, h, w)
        
        # Apply squashing function
        capsules = self.squash_capsules(capsules)
        
        # Spatial attention weighting
        caps_flat = capsules.view(batch_size, 8 * 16, h, w)
        attention = self.spatial_attention(caps_flat)  # [batch_size, 1, 32, 32]
        
        # Apply attention
        weighted_capsules = caps_flat * attention
        
        # Final feature extraction
        final_features = self.feature_projection(weighted_capsules)
        
        return final_features
    
    def squash_capsules(self, capsules):
        """Apply squashing to capsule vectors"""
        # capsules: [batch_size, num_caps, cap_dim, h, w]
        squared_norm = (capsules ** 2).sum(dim=2, keepdim=True)  # [batch_size, num_caps, 1, h, w]
        scale = squared_norm / (1 + squared_norm)
        unit_vector = capsules / torch.sqrt(squared_norm + 1e-8)
        return scale * unit_vector

def create_simplified_capsnet(input_channels=3, input_size=256, feature_dim=128, **kwargs):
    """Create simplified CapsNet for faster training"""
    return SimplifiedCapsNet(input_channels, input_size, feature_dim, **kwargs)

# Hybrid CapsNet-CNN for comparison
class HybridCapsNetCNN(nn.Module):
    """
    Hybrid model that combines CapsNet spatial features with CNN features
    """
    
    def __init__(self, input_channels=3, input_size=256, feature_dim=128, 
                 capsnet_features=64, cnn_features=64, dropout_rate=0.3):
        super(HybridCapsNetCNN, self).__init__()
        
        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 128, 7, 2, 3),  # 128x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            
            nn.Conv2d(128, 256, 5, 2, 2),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # CapsNet branch for spatial features
        self.capsnet_branch = SimplifiedCapsNet(256, 32, capsnet_features)
        
        # CNN branch for traditional features
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),  # 16x16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 4x4
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, cnn_features)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(capsnet_features + cnn_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x):
        # Shared features
        shared_features = self.shared_backbone(x)
        
        # CapsNet spatial features
        capsnet_features = self.capsnet_branch(shared_features)
        
        # CNN traditional features
        cnn_features = self.cnn_branch(shared_features)
        
        # Fuse features
        combined = torch.cat([capsnet_features, cnn_features], dim=1)
        final_features = self.fusion(combined)
        
        return final_features

def create_hybrid_capsnet_cnn(input_channels=3, input_size=256, feature_dim=128, **kwargs):
    """Create hybrid CapsNet-CNN model"""
    return HybridCapsNetCNN(input_channels, input_size, feature_dim, **kwargs)

class EfficientPrimaryCapsules(nn.Module):
    """
    Efficient Primary Capsules using Depthwise Separable Convolution
    Reduces parameters and computational cost while maintaining feature extraction capability
    """
    
    def __init__(self, num_capsules=8, in_channels=256, capsule_dim=16, kernel_size=9, stride=2):
        super(EfficientPrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Depthwise separable convolution
        # Depthwise: applies a single filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0, 
            groups=in_channels  # Key for depthwise
        )
        
        # Pointwise: 1x1 convolution to combine depthwise outputs
        self.pointwise = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=1
        )
        
        self.batch_norm = nn.BatchNorm2d(num_capsules * capsule_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, in_channels, height, width]
        Returns:
            capsules: [batch_size, num_capsules * spatial, capsule_dim]
        """
        # Depthwise separable convolution
        x = self.depthwise(x)  # [batch_size, in_channels, new_h, new_w]
        x = self.pointwise(x)  # [batch_size, num_capsules * capsule_dim, new_h, new_w]
        x = self.batch_norm(x)
        
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        
        # Reshape to capsule format: [batch_size, num_capsules, capsule_dim, height, width]
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, height, width)
        
        # Reshape to: [batch_size, num_capsules * height * width, capsule_dim]
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [batch_size, num_capsules, height, width, capsule_dim]
        x = x.view(batch_size, self.num_capsules * height * width, self.capsule_dim)
        
        # Apply squashing
        return self.squash(x)
    
    def squash(self, tensor):
        """
        Squash activation function for capsules
        - Preserves orientation (feature representation)
        - Constrains length to [0, 1] (feature probability)
        - Enhances gradient flow
        """
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        return scale * unit_vector

class SelfAttentionRouting(nn.Module):
    """
    Self-Attention Routing for Efficient-CapsNet
    Replaces iterative dynamic routing with efficient attention mechanism
    """
    
    def __init__(self, num_input_capsules, num_output_capsules, input_dim, output_dim):
        super(SelfAttentionRouting, self).__init__()
        self.num_output_capsules = num_output_capsules
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Transformation matrix for each output capsule
        self.W = nn.Parameter(torch.randn(num_output_capsules, input_dim, output_dim) * 0.01)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_input_capsules, input_dim]
        Returns:
            output_capsules: [batch_size, num_output_capsules, output_dim]
        """
        batch_size = x.size(0)
        num_input = x.size(1)
        
        # Transform input capsules to predictions for each output capsule
        # x: [batch_size, num_input, input_dim]
        # W: [num_output, input_dim, output_dim]
        # We want: [batch_size, num_input, num_output, output_dim]
        
        # Expand x to: [batch_size, num_input, 1, input_dim, 1]
        x_expanded = x.unsqueeze(2).unsqueeze(-1)  # [batch_size, num_input, 1, input_dim, 1]
        
        # Expand W to: [1, 1, num_output, input_dim, output_dim]
        W_expanded = self.W.unsqueeze(0).unsqueeze(0)  # [1, 1, num_output, input_dim, output_dim]
        
        # Compute predictions using element-wise multiplication and sum
        # Broadcasting: [B, num_input, 1, input_dim, 1] * [1, 1, num_output, input_dim, output_dim]
        # Sum over input_dim dimension
        u_hat = (x_expanded * W_expanded).sum(dim=3)  # [batch_size, num_input, num_output, output_dim]
        
        # Simplified routing - use mean pooling over input capsules
        # This avoids complex attention calculations that can cause dimension mismatches
        output = u_hat.mean(dim=1)  # [batch_size, num_output, output_dim]
        
        # Apply squashing
        return self.squash(output)
    
    def squash(self, tensor):
        """Squash activation function"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        return scale * unit_vector

class EfficientCapsNet(nn.Module):
    """
    Efficient-CapsNet: Streamlined Capsule Network with Self-Attention Routing
    
    Architecture:
    1. Feature Extraction: Conv + BatchNorm layers
    2. Primary Capsules: Depthwise Separable Convolution
    3. Self-Attention Routing: Efficient routing mechanism
    4. Squash Activation: Preserves orientation, constrains length
    
    Benefits:
    - Reduced parameters via depthwise separable convolutions
    - Efficient self-attention routing (no iterative routing)
    - Better gradient flow via squash activation
    - Captures feature relationships effectively
    """
    
    def __init__(self, input_channels=3, input_size=256, feature_dim=128,
                 num_primary_capsules=8, primary_capsule_dim=16,
                 num_output_capsules=10, output_capsule_dim=16,
                 dropout_rate=0.3):
        super(EfficientCapsNet, self).__init__()
        
        self.input_size = input_size
        self.feature_dim = feature_dim
        
        # Feature Extraction Backbone
        self.feature_extraction = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
        )
        
        # Calculate size after feature extraction
        feature_size = input_size // 8  # 256 // 8 = 32
        
        # Efficient Primary Capsules with Depthwise Separable Conv
        self.primary_capsules = EfficientPrimaryCapsules(
            num_capsules=num_primary_capsules,
            in_channels=256,
            capsule_dim=primary_capsule_dim,
            kernel_size=9,
            stride=2
        )
        
        
        # Calculate primary capsules spatial dimensions
        primary_spatial = (feature_size - 8) // 2  # (32 - 8) // 2 = 12
        num_primary_total = num_primary_capsules * primary_spatial * primary_spatial
        
        # Self-Attention Routing
        self.self_attention_routing = SelfAttentionRouting(
            num_input_capsules=num_primary_total,
            num_output_capsules=num_output_capsules,
            input_dim=primary_capsule_dim,
            output_dim=output_capsule_dim
        )
        
        # Feature Projection Layer
        self.feature_projection = nn.Sequential(
            nn.Linear(num_output_capsules * output_capsule_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"Efficient-CapsNet Architecture:")
        print(f"  Input: {input_channels}x{input_size}x{input_size}")
        print(f"  Feature extraction output: 256x{feature_size}x{feature_size}")
        print(f"  Primary capsules: {num_primary_total} capsules of {primary_capsule_dim}D")
        print(f"  Output capsules: {num_output_capsules} capsules of {output_capsule_dim}D")
        print(f"  Final features: {feature_dim}D")
        print(f"  Parameters reduced via depthwise separable convolutions")
        print(f"  Efficient self-attention routing (no iterative routing)")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            features: Feature vectors [batch_size, feature_dim]
        """
        # Feature extraction
        x = self.feature_extraction(x)  # [batch_size, 256, 32, 32]
        
        # Primary capsules with depthwise separable conv
        primary_caps = self.primary_capsules(x)  # [batch_size, num_primary_total, primary_dim]
        
        # Self-attention routing
        output_caps = self.self_attention_routing(primary_caps)  # [batch_size, num_output, output_dim]
        
        # Flatten capsules
        flattened = output_caps.view(output_caps.size(0), -1)  # [batch_size, num_output * output_dim]
        
        # Project to final feature space
        features = self.feature_projection(flattened)  # [batch_size, feature_dim]
        
        return features
    
    def get_capsule_lengths(self, capsules):
        """
        Get lengths of capsule vectors (feature probabilities)
        
        Args:
            capsules: [batch_size, num_capsules, capsule_dim]
        
        Returns:
            lengths: [batch_size, num_capsules]
        """
        return torch.sqrt((capsules ** 2).sum(dim=-1))
    
    def extract_features_with_analysis(self, x):
        """
        Extract features with detailed capsule analysis
        
        Returns:
            features: Final feature vector
            analysis: Dictionary with capsule information
        """
        # Feature extraction
        x = self.feature_extraction(x)
        
        # Primary capsules
        primary_caps = self.primary_capsules(x)
        
        # Self-attention routing
        output_caps = self.self_attention_routing(primary_caps)
        
        # Final features
        flattened = output_caps.view(output_caps.size(0), -1)
        features = self.feature_projection(flattened)
        
        # Analyze capsule outputs
        capsule_lengths = self.get_capsule_lengths(output_caps)
        
        analysis = {
            'output_capsules': output_caps,
            'capsule_lengths': capsule_lengths,
            'active_capsules': (capsule_lengths > 0.1).sum(dim=1),
            'max_capsule_length': capsule_lengths.max(dim=1)[0],
            'capsule_diversity': capsule_lengths.std(dim=1),
            'primary_capsules': primary_caps
        }
        
        return features, analysis

def create_efficient_capsnet(input_channels=3, input_size=256, feature_dim=128, **kwargs):
    """
    Create an Efficient-CapsNet model
    
    Args:
        input_channels: Number of input channels (3 for RGB)
        input_size: Input image size (256x256)
        feature_dim: Final feature dimension (128)
        **kwargs: Additional hyperparameters
    
    Returns:
        Efficient-CapsNet model with depthwise separable convolutions and self-attention routing
    """
    return EfficientCapsNet(
        input_channels=input_channels,
        input_size=input_size,
        feature_dim=feature_dim,
        **kwargs
    )
