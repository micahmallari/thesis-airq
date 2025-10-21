import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
import json
import time
from itertools import product

class LSTMTemporalFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, 
                 activation='relu', lstm_dropout=0.0):
        super(LSTMTemporalFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Feature extraction layer (32 dimensions for hybrid integration)
        self.temporal_feature_layer = nn.Linear(hidden_size, 32)
        
        # Optional prediction layer (for CapsNet-LSTM model without LightGBM)
        self.prediction_layer = nn.Linear(32, 1)
        
    def forward(self, x, return_features_only=False):
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Use last timestep output
        
        # Extract 32-dimensional temporal features
        temporal_features = self.activation(self.temporal_feature_layer(last_output))
        temporal_features = self.dropout(temporal_features)
        
        if return_features_only:
            return temporal_features
        else:
            # For CapsNet-LSTM model (without LightGBM)
            prediction = self.prediction_layer(temporal_features)
            return temporal_features, prediction

class LSTMTemporalFeatureGenerator:
    def __init__(self, best_params=None):
        self.default_params = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.2,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'timesteps': 60,  # Default temporal window size
            'weight_decay': 0.0,
            'grad_clip': 1.0,
            'lstm_dropout': 0.0
        }
        
        self.params = best_params if best_params else self.default_params
        self.scaler = None
        self.model = None
        
    def prepare_temporal_sequences(self, temporal_data, targets, timesteps=10):
        """
        Prepare temporal sequences for a given fold
        Called during each CV fold by the full pipeline
        """
        # Scale temporal features
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            temporal_data_scaled = self.scaler.fit_transform(temporal_data)
        else:
            temporal_data_scaled = self.scaler.transform(temporal_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(temporal_data_scaled) - timesteps):
            X.append(temporal_data_scaled[i:i+timesteps])
            y.append(targets[i+timesteps])
        
        return np.array(X), np.array(y)
    
    def train_and_extract_features(self, train_temporal_data, train_targets, val_temporal_data, val_targets, timesteps=None):
        if timesteps is None:
            timesteps = self.params.get('timesteps', 60)
            
        X_train, y_train = self.prepare_temporal_sequences(train_temporal_data, train_targets, timesteps)
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        self.model = LSTMTemporalFeatureExtractor(
            input_size=train_temporal_data.shape[1],
            hidden_size=self.params['hidden_size'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            activation=self.params['activation'],
            lstm_dropout=self.params.get('lstm_dropout', 0.0)
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.params['learning_rate'],
            weight_decay=self.params.get('weight_decay', 0.0)
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                _, predictions = self.model(batch_X, return_features_only=False)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.params.get('grad_clip', 1.0)
                )
                
                optimizer.step()
                total_loss += loss.item()
        
        # Extract features from training data
        self.model.eval()
        with torch.no_grad():
            train_features = self.model(X_train_tensor, return_features_only=True).numpy()
        
   
        X_val, _ = self.prepare_temporal_sequences(val_temporal_data, val_targets, timesteps)
        X_val_tensor = torch.FloatTensor(X_val)
        
        with torch.no_grad():
            val_features = self.model(X_val_tensor, return_features_only=True).numpy()
        
      
        print(f"  LSTM Debug - Train features shape: {train_features.shape}, type: {type(train_features)}")
        print(f"  LSTM Debug - Val features shape: {val_features.shape}, type: {type(val_features)}")
        print(f"  LSTM Debug - Feature sample: {train_features[0][:5]}...")  # First 5 values
        
        return train_features, val_features, y_train, X_val.shape[0]
    
class TemporalDataLoader:

    def __init__(self, days=['7_24', '10_19', '11_10']):
        self.days = days
        self.temporal_features = ['pm10', 'temperature', 'humidity']
    
    def load_temporal_data(self, day):

        matched_file = f'dataset/c_matched_spatio_temporal_data/matched_{day}.csv'
        if not os.path.exists(matched_file):
            raise FileNotFoundError(f"Matched data not found: {matched_file}")
        
        df = pd.read_csv(matched_file)
        
        # Sort by timestamp for proper time series handling
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Get available temporal features
        available_features = [col for col in self.temporal_features if col in df.columns]
        
        if not available_features:
            raise ValueError(f"No temporal features found in {day}")
        
        # Remove rows with missing values
        required_cols = available_features + ['pm2.5']
        df_clean = df.dropna(subset=required_cols)
        
        temporal_data = df_clean[available_features].values
        targets = df_clean['pm2.5'].values
        
        return temporal_data, targets, available_features

def hyperparameter_tuning_lstm(day, max_combinations=30):

    hyperparameter_space = {
        # LSTM Architecture
        'hidden_size': [32, 64, 128, 256],           # More options for LSTM 
        'num_layers': [1, 2, 3],                     # Try deeper networks
        'dropout': [0.1, 0.2, 0.3, 0.4],           # Regularization strength
        'activation': ['relu', 'tanh', 'gelu'],      # Different activation functions
        
        # Training Parameters
        'learning_rate': [0.0001, 0.001, 0.01],     # Learning rate range
        'batch_size': [16, 32, 64],                  # Batch size options
        'epochs': [30, 50],                     # Proper training duration
        
        # Sequence Parameters
        'timesteps': [10, 30, 40, 60],               # Sequence length for temporal patterns
        
        # Regularization
        'weight_decay': [0.0, 1e-5, 1e-4],         # L2 regularization
        'grad_clip': [0.5, 1.0, 2.0],              # Gradient clipping
        
        # LSTM-specific
        'lstm_dropout': [0.0, 0.1, 0.2],           # LSTM internal dropout
    }
    
    print(f"LSTM Hyperparameter Tuning for {day}")
    print("=" * 50)
    
    # Load data
    data_loader = TemporalDataLoader()
    temporal_data, targets, feature_names = data_loader.load_temporal_data(day)
    
    n_total = len(temporal_data)
    n_learning = int(n_total * 0.8)  # Only use 80% learning set
    
    learning_temporal = temporal_data[:n_learning]  # 80% learning set only
    learning_targets = targets[:n_learning]
    

    n_tune_train = int(n_learning * 0.8) 
    
    tune_train_temporal = learning_temporal[:n_tune_train]      
    tune_train_targets = learning_targets[:n_tune_train]
    tune_val_temporal = learning_temporal[n_tune_train:]        
    tune_val_targets = learning_targets[n_tune_train:]
    
    print(f"Data split for hyperparameter tuning (separate from 5-fold CV):")
    print(f"  Total samples: {n_total}")
    print(f"  Learning set (used): {n_learning} ({n_learning/n_total*100:.1f}%)")
    print(f"  Hold-out test (untouched): {n_total - n_learning} ({(n_total - n_learning)/n_total*100:.1f}%)")
    print(f"  Tune train: {len(tune_train_temporal)} ({len(tune_train_temporal)/n_total*100:.1f}%)")
    print(f"  Tune val: {len(tune_val_temporal)} ({len(tune_val_temporal)/n_total*100:.1f}%)")
    
    # Generate parameter combinations
    keys = list(hyperparameter_space.keys())
    values = list(hyperparameter_space.values())
    all_combinations = list(product(*values))
    
    if len(all_combinations) > max_combinations:
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_combinations), size=max_combinations, replace=False)
        combinations = [all_combinations[i] for i in selected_indices]
    else:
        combinations = all_combinations
    
    param_combinations = [dict(zip(keys, combo)) for combo in combinations]
    
    best_params = None
    best_score = float('inf')
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"  Testing combination {i+1}/{len(param_combinations)}: {params}")
        
       
        lstm_gen = LSTMTemporalFeatureGenerator(params)
        
        try:
          
            train_features, val_features, train_y, val_len = lstm_gen.train_and_extract_features(
                tune_train_temporal, tune_train_targets, tune_val_temporal, tune_val_targets, 
                timesteps=params.get('timesteps', 60)
            )
            
           
            val_y = tune_val_targets[params.get('timesteps', 60):]  
            if len(val_y) == len(val_features):
                
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(train_features, train_y)
                val_pred = lr.predict(val_features)
                
                mse = mean_squared_error(val_y, val_pred)
                rmse = np.sqrt(mse)
                
                print(f"    Validation RMSE: {rmse:.4f}")
                
                if mse < best_score:
                    best_score = mse
                    best_params = params.copy()
                    print(f"     New best score: {rmse:.4f}")
                
        except Exception as e:
            print(f"     Error with params: {e}")
            continue
    
    # Save best parameters
    os.makedirs('models/lstm_temporal', exist_ok=True)
    with open(f'models/lstm_temporal/{day}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best parameters for {day}: {best_params}")
    print(f"Best validation RMSE: {np.sqrt(best_score):.4f}")
    
    return best_params
    

def main():
    """
    Main execution for LSTM temporal feature extraction
    Focused on component preparation for full pipeline CV
    """
    print("="*80)
    print("LSTM TEMPORAL FEATURE EXTRACTION - PIPELINE READY")
    print("="*80)
    print("Purpose: Prepare LSTM component for full pipeline cross-validation")
    print("- Hyperparameter tuning for each day")
    print("- Component ready for CV integration")
    print("- No standalone evaluation (done in full pipeline)")
    print("="*80)
    
    days = ['7_24', '10_19', '11_10']
    
    # Step 1: Hyperparameter tuning for each day
    print("\nStep 1: LSTM Hyperparameter Tuning")
    print("-" * 50)
    
    for day in days:
        try:
            print(f"\nTuning LSTM parameters for {day}...")
            best_params = hyperparameter_tuning_lstm(day, max_combinations=30)
            print(f" {day} tuning complete")
            
        except Exception as e:
            print(f" Error tuning {day}: {str(e)}")
    
    print(f"\n{'='*80}")
    print("LSTM COMPONENT PREPARATION COMPLETE")
    print("="*80)
    print("Best parameters saved for each day")
    print("LSTMTemporalFeatureGenerator class ready")
    print("TemporalDataLoader class ready")
    print("\nNext Steps:")
    print("1. Integration team: Use LSTMTemporalFeatureGenerator in full pipeline")
    print("2. Full pipeline CV: Wrap entire models in 5-fold cross-validation")
    print("3. Models to evaluate: CapsNet-LSTM-LightGBM, CNN-LSTM-LightGBM, CapsNet-LSTM")
    print("4. Statistical analysis: Compare 5-fold results across models")
    print("="*80)
    
    # Example usage for integration team
    print("\n" + "="*80)
    print("CORRECT 5-FOLD CV INTEGRATION:")
    print("="*80)
    print("""
# PHASE 1: Hyperparameter Tuning (run once per day)
from lstm_temporal_feature_generator import hyperparameter_tuning_lstm
best_params = hyperparameter_tuning_lstm('7_24', max_combinations=10)

# PHASE 2: 5-Fold Cross-Validation (main pipeline)
from lstm_temporal_feature_generator import LSTMTemporalFeatureGenerator, TemporalDataLoader
from sklearn.model_selection import TimeSeriesSplit

# Initialize components
data_loader = TemporalDataLoader()
lstm_generator = LSTMTemporalFeatureGenerator(best_params)

# Load data and split into learning (80%) and hold-out (20%)
temporal_data, targets, _ = data_loader.load_temporal_data('7_24')
n_total = len(temporal_data)
n_learning = int(n_total * 0.8)

learning_temporal = temporal_data[:n_learning]  # 80% learning set
learning_targets = targets[:n_learning]
holdout_temporal = temporal_data[n_learning:]   # 20% hold-out (untouched)
holdout_targets = targets[n_learning:]

# 5-Fold TimeSeriesSplit on 80% learning set
tscv = TimeSeriesSplit(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(learning_temporal)):
    print(f"Fold {fold + 1}/5")
    
    # Split data for this fold (expanding window)
    train_temporal = learning_temporal[train_idx]  # Growing train set
    val_temporal = learning_temporal[val_idx]      # Fixed-size val set
    train_targets = learning_targets[train_idx]
    val_targets = learning_targets[val_idx]
    
    # Extract temporal features for this fold
    train_temp_features, val_temp_features, _, _ = lstm_generator.train_and_extract_features(
        train_temporal, train_targets, val_temporal, val_targets
    )
    
    # Extract spatial features (CapsNet/CNN)
    train_spatial_features, val_spatial_features = capsnet_generator.train_and_extract_features(
        train_images, train_targets, val_images, val_targets
    )
    
    # Combine features
    train_combined = np.concatenate([train_temp_features, train_spatial_features], axis=1)
    val_combined = np.concatenate([val_temp_features, val_spatial_features], axis=1)
    
    # Train LightGBM meta-learner
    lightgbm_model = LGBMRegressor(best_lightgbm_params)
    lightgbm_model.fit(train_combined, train_targets)
    
    # Validate on this fold
    val_predictions = lightgbm_model.predict(val_combined)
    fold_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
    fold_scores.append(fold_rmse)
    
    print(f"  Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Calculate cross-validation performance
cv_mean = np.mean(fold_scores)
cv_std = np.std(fold_scores)
print(f"5-Fold CV Results: {cv_mean:.4f} ± {cv_std:.4f}")
""")
    print("="*80)

if __name__ == "__main__":
    main()
