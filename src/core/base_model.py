"""
Base model class for all models in AstroAssistance.
"""
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.core.config import config_manager
from src.core.logger import app_logger


class BaseModel(ABC):
    """Abstract base class for all models in the system."""
    
    def __init__(self, model_name: str, model_type: str):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model (e.g., 'classifier', 'recommender')
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.metadata = {
            "name": model_name,
            "type": model_type,
            "version": "0.1.0",
            "training_history": [],
            "performance_metrics": {},
        }
    
    @abstractmethod
    def build(self, **kwargs) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, train_data, validation_data=None, **kwargs) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        pass
    
    @abstractmethod
    def predict(self, data, **kwargs) -> Any:
        """
        Make predictions with the model.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_data: Test data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model. If None, uses default path.
            
        Returns:
            Path where the model was saved
        """
        if not self.is_trained:
            app_logger.warning(f"Attempting to save untrained model {self.model_name}")
        
        if path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            models_dir = os.path.join(project_root, "models")
            os.makedirs(models_dir, exist_ok=True)
            path = os.path.join(models_dir, f"{self.model_name}.pt")
        
        # Save model weights
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            # For non-PyTorch models, use appropriate saving method
            try:
                import joblib
                joblib.dump(self.model, path)
            except Exception as e:
                app_logger.error(f"Failed to save model {self.model_name}: {str(e)}")
                raise
        
        # Save metadata
        metadata_path = os.path.splitext(path)[0] + "_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        app_logger.info(f"Model {self.model_name} saved to {path}")
        return path
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from. If None, uses default path.
        """
        if path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            path = os.path.join(project_root, "models", f"{self.model_name}.pt")
        
        # Load model weights
        if not os.path.exists(path):
            app_logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}")
        
        try:
            if isinstance(self.model, torch.nn.Module):
                self.model.load_state_dict(torch.load(path))
                self.model.eval()  # Set to evaluation mode
            else:
                # For non-PyTorch models, use appropriate loading method
                import joblib
                self.model = joblib.load(path)
        except Exception as e:
            app_logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
        
        # Load metadata if it exists
        metadata_path = os.path.splitext(path)[0] + "_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        
        self.is_trained = True
        app_logger.info(f"Model {self.model_name} loaded from {path}")
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update model metadata.
        
        Args:
            key: Metadata key to update
            value: New value
        """
        self.metadata[key] = value
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels or probabilities
            
        Returns:
            Dictionary of metrics
        """
        # For classification tasks
        if self.model_type == "classifier":
            # Convert probabilities to class predictions if needed
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                threshold = config_manager.get("evaluation.threshold", 0.5)
                y_pred_classes = (y_pred > threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred_classes),
                "precision": precision_score(y_true, y_pred_classes, average="weighted"),
                "recall": recall_score(y_true, y_pred_classes, average="weighted"),
                "f1": f1_score(y_true, y_pred_classes, average="weighted"),
            }
        
        # For regression tasks
        elif self.model_type == "regressor":
            metrics = {
                "mse": np.mean((y_true - y_pred) ** 2),
                "mae": np.mean(np.abs(y_true - y_pred)),
                "r2": 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            }
        
        # For other types of models
        else:
            metrics = {}
        
        return metrics