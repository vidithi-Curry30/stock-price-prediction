import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class StockLSTM:
    def __init__(self, sequence_length=20, hidden_dim=32, num_layers=2, 
                 epochs=100, learning_rate=0.01):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_sequences(self, data):
        """Create sequences for LSTM training"""
        # Ensure data is numpy array
        if isinstance(data, pd.Series):
            data = data.values
            
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def train(self, train_data):
        """Train the LSTM model with better optimization"""
        X_train, y_train = self.prepare_sequences(train_data)
        
        if len(X_train) == 0:
            print(f"Not enough data for sequence length {self.sequence_length}")
            return
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).reshape(-1, self.sequence_length, 1)
        y_train = torch.FloatTensor(y_train)
        
        # Initialize model
        self.model = LSTMModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, test_data, train_data):
        """Make predictions with proper scaling"""
        if self.model is None:
            return np.array([])
            
        self.model.eval()
        
        # Combine train and test for proper sequence creation
        if isinstance(train_data, pd.Series):
            train_data = train_data.values
        if isinstance(test_data, pd.Series):
            test_data = test_data.values
            
        # Take last sequence_length points from train and all test
        last_train = train_data[-self.sequence_length:]
        combined = np.concatenate([last_train, test_data])
        
        # Scale
        scaled_combined = self.scaler.transform(combined.reshape(-1, 1))
        
        predictions = []
        
        # Predict for each test point
        for i in range(self.sequence_length, len(scaled_combined)):
            X_test = scaled_combined[i-self.sequence_length:i, 0]
            X_test = torch.FloatTensor(X_test).reshape(1, self.sequence_length, 1)
            
            with torch.no_grad():
                pred = self.model(X_test)
                predictions.append(pred.item())
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
