import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import json
import time
from itertools import product
from tqdm import tqdm
import optuna

class LSTMTemporalFeatureExtractor(nn.Module):
    """
    LSTM component for hybrid models that produces 32-dimensional temporal features
    Compatible with: CapsNet-LSTM-LightGBM, CNN-LSTM-LightGBM, CapsNet-LSTM
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, activation='relu', lstm_dropout=0.0):
        super(LSTMTemporalFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0,
            batch_first=True
        )
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.temporal_feature_layer = nn.Linear(hidden_size, 32)
        self.prediction_layer = nn.Linear(32, 1)

    def forward(self, x, return_features_only=False):
        lstm_out, (hn, cn) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        temporal_features = self.activation(self.temporal_feature_layer(last_output))
        temporal_features = self.dropout(temporal_features)
        if return_features_only:
            return temporal_features
        else:
            prediction = self.prediction_layer(temporal_features)
            return temporal_features, prediction

class LSTMTemporalFeatureGenerator:
    """
    LSTM component for temporal feature extraction in hybrid models
    Designed to work within full pipeline cross-validation
    """
    def __init__(self, best_params=None):
        self.default_params = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.2,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'timesteps': 60,
            'weight_decay': 0.0,
            'grad_clip': 1.0,
            'lstm_dropout': 0.0
        }
        self.params = best_params if best_params else self.default_params
        self.scaler = None
        self.model = None

    def prepare_temporal_sequences(self, temporal_data, targets, timesteps=10):
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            temporal_data_scaled = self.scaler.fit_transform(temporal_data)
        else:
            temporal_data_scaled = self.scaler.transform(temporal_data)
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
        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                _, predictions = self.model(batch_X, return_features_only=False)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.params.get('grad_clip', 1.0)
                )
                optimizer.step()
                total_loss += loss.item()
        self.model.eval()
        with torch.no_grad():
            train_features = self.model(X_train_tensor, return_features_only=True).numpy()
        X_val, _ = self.prepare_temporal_sequences(val_temporal_data, val_targets, timesteps)
        X_val_tensor = torch.FloatTensor(X_val)
        with torch.no_grad():
            val_features = self.model(X_val_tensor, return_features_only=True).numpy()
        print(f"  LSTM Debug - Train features shape: {train_features.shape}, type: {type(train_features)}")
        print(f"  LSTM Debug - Val features shape: {val_features.shape}, type: {type(val_features)}")
        print(f"  LSTM Debug - Feature sample: {train_features[0][:5]}...")
        return train_features, val_features, y_train, X_val.shape[0]

class TemporalDataLoader:
    def __init__(self, days=['10_19', '11_10', '7_24']):
        self.days = days
        self.temporal_features = ['pm10', 'temperature', 'humidity']
    def load_temporal_data(self, day):
        # Use path relative to workspace root
        matched_file = f'dataset/c_matched_spatio_temporal_data/matched_{day}.csv'
        if not os.path.exists(matched_file):
            raise FileNotFoundError(f"Matched data not found: {matched_file}")
        df = pd.read_csv(matched_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        available_features = [col for col in self.temporal_features if col in df.columns]
        if not available_features:
            raise ValueError(f"No temporal features found in {day}")
        required_cols = available_features + ['pm2.5']
        df_clean = df.dropna(subset=required_cols)
        temporal_data = df_clean[available_features].values
        targets = df_clean['pm2.5'].values
        return temporal_data, targets, available_features

def hyperparameter_tuning_lstm(day, n_trials=30):
    print(f"LSTM Hyperparameter Tuning for {day}")
    print("=" * 50)
    data_loader = TemporalDataLoader()
    temporal_data, targets, feature_names = data_loader.load_temporal_data(day)
    n_total = len(temporal_data)
    n_learning = int(n_total * 0.8)
    learning_temporal = temporal_data[:n_learning]
    learning_targets = targets[:n_learning]
    n_tune_train = int(n_learning * 0.8)
    tune_train_temporal = learning_temporal[:n_tune_train]
    tune_train_targets = learning_targets[:n_tune_train]
    tune_val_temporal = learning_temporal[n_tune_train:]
    tune_val_targets = learning_targets[n_tune_train:]
    def objective(trial):
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'gelu']),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': trial.suggest_categorical('epochs', [10, 30, 50]),
            'timesteps': trial.suggest_categorical('timesteps', [10, 30, 40, 60]),
            'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4]),
            'grad_clip': trial.suggest_categorical('grad_clip', [0.5, 1.0, 2.0]),
            'lstm_dropout': trial.suggest_categorical('lstm_dropout', [0.0, 0.1, 0.2])
        }
        lstm_gen = LSTMTemporalFeatureGenerator(params)
        try:
            train_features, val_features, train_y, val_len = lstm_gen.train_and_extract_features(
                tune_train_temporal, tune_train_targets, tune_val_temporal, tune_val_targets,
                timesteps=params['timesteps']
            )
            val_y = tune_val_targets[params['timesteps']:]
            if len(val_y) == len(val_features):
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(train_features, train_y)
                val_pred = lr.predict(val_features)
                mse = mean_squared_error(val_y, val_pred)
                return mse
            else:
                return float('inf')
        except Exception as e:
            print(f"Optuna trial error: {e}")
            return float('inf')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_score = study.best_value
    os.makedirs('models/lstm_temporal', exist_ok=True)
    with open(f'models/lstm_temporal/{day}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters for {day}: {best_params}")
    print(f"Best validation RMSE: {np.sqrt(best_score):.4f}")
    return best_params
