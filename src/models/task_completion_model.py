"""
Task completion prediction model.
"""
import os
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.core.base_model import BaseModel
from src.core.config import config_manager
from src.core.logger import app_logger
from src.data_processing.data_processor import TaskDataset, create_data_loaders


class TaskCompletionNN(nn.Module):
    """Neural network for task completion prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(TaskCompletionNN, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)


class TaskCompletionModel(BaseModel):
    """Model for predicting task completion."""
    
    def __init__(self, input_dim: int = None):
        """
        Initialize the task completion model.
        
        Args:
            input_dim: Dimension of input features
        """
        super().__init__("task_completion_model", "classifier")
        self.input_dim = input_dim
    
    def build(self, **kwargs) -> None:
        """Build the model architecture."""
        # Get model configuration
        hidden_dims = kwargs.get("hidden_dims", config_manager.get("models.task_classifier.hidden_dims", [128, 64]))
        dropout = kwargs.get("dropout", config_manager.get("models.task_classifier.dropout", 0.2))
        
        # Check if input_dim is provided
        if self.input_dim is None:
            self.input_dim = kwargs.get("input_dim")
            if self.input_dim is None:
                raise ValueError("input_dim must be provided either during initialization or in build()")
        
        # Build the model
        self.model = TaskCompletionNN(self.input_dim, hidden_dims, dropout)
        app_logger.info(f"Built task completion model with input_dim={self.input_dim}, hidden_dims={hidden_dims}")
    
    def train(self, train_data, validation_data=None, **kwargs) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Training data (DataLoader or tuple of (X, y))
            validation_data: Validation data (DataLoader or tuple of (X, y))
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        # Get training parameters
        epochs = kwargs.get("epochs", config_manager.get("training.epochs", 50))
        learning_rate = kwargs.get("learning_rate", config_manager.get("training.learning_rate", 0.001))
        batch_size = kwargs.get("batch_size", config_manager.get("training.batch_size", 32))
        early_stopping_patience = kwargs.get(
            "early_stopping_patience", 
            config_manager.get("training.early_stopping.patience", 5)
        )
        early_stopping_min_delta = kwargs.get(
            "early_stopping_min_delta", 
            config_manager.get("training.early_stopping.min_delta", 0.001)
        )
        
        # Prepare data loaders
        if isinstance(train_data, tuple):
            X_train, y_train = train_data
            if validation_data is not None:
                X_val, y_val = validation_data
                train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
            else:
                # Create train loader without validation
                train_dataset = TaskDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = None
        else:
            # Assume train_data is already a DataLoader
            train_loader = train_data
            val_loader = validation_data
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }
        
        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_metrics = self._evaluate_during_training(val_loader, criterion)
                
                # Update history
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])
                
                # Log progress
                app_logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}"
                )
                
                # Check for early stopping
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        app_logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # Log progress without validation
                app_logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Restore best model if early stopping occurred
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Update metadata
        self.metadata["training_history"] = history
        self.is_trained = True
        
        return history
    
    def _evaluate_during_training(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model during training.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (validation_loss, metrics_dict)
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Store predictions and targets
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0)
        }
        
        return val_loss, metrics
    
    def predict(self, data, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            data: Input data (numpy array, DataFrame, or DataLoader)
            **kwargs: Additional prediction parameters
            
        Returns:
            Numpy array of predictions
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Predictions may be unreliable.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert data to appropriate format
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to numpy array
            data = data.values.astype(np.float32)
            data_tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, DataLoader):
            # Use DataLoader directly
            all_preds = []
            with torch.no_grad():
                for batch_X, _ in data:
                    outputs = self.model(batch_X).squeeze()
                    preds = (outputs > 0.5).float().cpu().numpy()
                    all_preds.extend(preds)
            return np.array(all_preds)
        else:
            raise ValueError("Unsupported data type. Expected DataFrame, numpy array, or DataLoader.")
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(data_tensor).squeeze()
            predictions = (outputs > 0.5).float().cpu().numpy()
        
        return predictions
    
    def predict_proba(self, data, **kwargs) -> np.ndarray:
        """
        Make probability predictions with the model.
        
        Args:
            data: Input data (numpy array, DataFrame, or DataLoader)
            **kwargs: Additional prediction parameters
            
        Returns:
            Numpy array of probability predictions
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Predictions may be unreliable.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert data to appropriate format
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to numpy array
            data = data.values.astype(np.float32)
            data_tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, DataLoader):
            # Use DataLoader directly
            all_probs = []
            with torch.no_grad():
                for batch_X, _ in data:
                    outputs = self.model(batch_X).squeeze()
                    all_probs.extend(outputs.cpu().numpy())
            return np.array(all_probs)
        else:
            raise ValueError("Unsupported data type. Expected DataFrame, numpy array, or DataLoader.")
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(data_tensor).squeeze()
            probabilities = outputs.cpu().numpy()
        
        return probabilities
    
    def evaluate(self, test_data, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_data: Test data (tuple of (X, y) or DataLoader)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Evaluation may be unreliable.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare data
        if isinstance(test_data, tuple):
            X_test, y_test = test_data
            test_dataset = TaskDataset(X_test, y_test)
            batch_size = kwargs.get("batch_size", config_manager.get("training.batch_size", 32))
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        else:
            # Assume test_data is already a DataLoader
            test_loader = test_data
        
        # Make predictions
        all_probs = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                
                # Store probabilities, predictions, and targets
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                targets = batch_y.cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0)
        }
        
        # Log evaluation results
        app_logger.info(f"Evaluation results: {metrics}")
        
        # Update metadata
        self.metadata["performance_metrics"] = metrics
        
        return metrics