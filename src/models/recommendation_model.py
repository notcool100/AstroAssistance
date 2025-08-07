"""
Recommendation model for AstroAssistance.
"""
import os
from typing import Dict, List, Any, Tuple, Optional, Union

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
from src.core.data_types import Recommendation, Task, Goal, UserPreference


class RecommendationNN(nn.Module):
    """Neural network for recommendation generation."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], dropout: float = 0.3):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(RecommendationNN, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer for recommendation type classification
        self.shared_layers = nn.Sequential(*layers)
        
        # Multiple output heads for different recommendation types
        self.task_scheduling_head = nn.Linear(prev_dim, 1)
        self.task_prioritization_head = nn.Linear(prev_dim, 1)
        self.goal_suggestion_head = nn.Linear(prev_dim, 1)
        self.productivity_tip_head = nn.Linear(prev_dim, 1)
        self.work_break_head = nn.Linear(prev_dim, 1)
        self.task_grouping_head = nn.Linear(prev_dim, 1)
        
        # Recommendation type classifier
        self.rec_type_classifier = nn.Linear(prev_dim, 6)  # 6 recommendation types
        
        # Confidence score predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of output tensors
        """
        # Shared layers
        shared_features = self.shared_layers(x)
        
        # Recommendation type probabilities
        rec_type_logits = self.rec_type_classifier(shared_features)
        rec_type_probs = torch.softmax(rec_type_logits, dim=1)
        
        # Confidence score
        confidence = self.confidence_predictor(shared_features)
        
        # Individual recommendation heads
        task_scheduling = torch.sigmoid(self.task_scheduling_head(shared_features))
        task_prioritization = torch.sigmoid(self.task_prioritization_head(shared_features))
        goal_suggestion = torch.sigmoid(self.goal_suggestion_head(shared_features))
        productivity_tip = torch.sigmoid(self.productivity_tip_head(shared_features))
        work_break = torch.sigmoid(self.work_break_head(shared_features))
        task_grouping = torch.sigmoid(self.task_grouping_head(shared_features))
        
        return {
            "rec_type_logits": rec_type_logits,
            "rec_type_probs": rec_type_probs,
            "confidence": confidence,
            "task_scheduling": task_scheduling,
            "task_prioritization": task_prioritization,
            "goal_suggestion": goal_suggestion,
            "productivity_tip": productivity_tip,
            "work_break": work_break,
            "task_grouping": task_grouping
        }


class RecommendationModel(BaseModel):
    """Model for generating recommendations."""
    
    def __init__(self, input_dim: int = None):
        """
        Initialize the recommendation model.
        
        Args:
            input_dim: Dimension of input features
        """
        super().__init__("recommendation_model", "recommender")
        self.input_dim = input_dim
        self.rec_type_mapping = {
            0: "task_scheduling",
            1: "task_prioritization",
            2: "goal_suggestion",
            3: "productivity_tip",
            4: "work_break_reminder",
            5: "task_grouping"
        }
        self.rec_type_reverse_mapping = {v: k for k, v in self.rec_type_mapping.items()}
    
    def build(self, **kwargs) -> None:
        """Build the model architecture."""
        # Get model configuration
        hidden_dims = kwargs.get("hidden_dims", config_manager.get("models.recommendation_model.hidden_dims", [256, 128, 64]))
        dropout = kwargs.get("dropout", config_manager.get("models.recommendation_model.dropout", 0.3))
        
        # Check if input_dim is provided
        if self.input_dim is None:
            self.input_dim = kwargs.get("input_dim")
            if self.input_dim is None:
                raise ValueError("input_dim must be provided either during initialization or in build()")
        
        # Build the model
        self.model = RecommendationNN(self.input_dim, hidden_dims, dropout)
        app_logger.info(f"Built recommendation model with input_dim={self.input_dim}, hidden_dims={hidden_dims}")
    
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
        
        # Set up optimizer and loss functions
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss functions for different outputs
        rec_type_criterion = nn.CrossEntropyLoss()
        binary_criterion = nn.BCELoss()
        
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
                outputs = self.model(batch_X)
                
                # For simplicity, we'll assume batch_y contains the recommendation type as an integer
                # In a real implementation, you would have more complex targets
                
                # Calculate loss for recommendation type classification
                rec_type_loss = rec_type_criterion(outputs["rec_type_logits"], batch_y.long())
                
                # Calculate loss for acceptance prediction (using the appropriate head based on rec type)
                acceptance_loss = 0.0
                for i, rec_type_idx in enumerate(batch_y.long()):
                    rec_type = self.rec_type_mapping[rec_type_idx.item()]
                    acceptance_loss += binary_criterion(outputs[rec_type][i], torch.ones(1))
                
                # Calculate confidence loss (assuming higher confidence is better)
                confidence_loss = binary_criterion(outputs["confidence"], torch.ones_like(outputs["confidence"]))
                
                # Combine losses
                loss = rec_type_loss + acceptance_loss + confidence_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_metrics = self._evaluate_during_training(val_loader, rec_type_criterion, binary_criterion)
                
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
    
    def _evaluate_during_training(self, val_loader: DataLoader, rec_type_criterion: nn.Module, binary_criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model during training.
        
        Args:
            val_loader: Validation data loader
            rec_type_criterion: Loss function for recommendation type
            binary_criterion: Loss function for binary outputs
            
        Returns:
            Tuple of (validation_loss, metrics_dict)
        """
        self.model.eval()
        val_loss = 0.0
        all_rec_type_preds = []
        all_rec_type_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate loss for recommendation type classification
                rec_type_loss = rec_type_criterion(outputs["rec_type_logits"], batch_y.long())
                
                # Calculate loss for acceptance prediction
                acceptance_loss = 0.0
                for i, rec_type_idx in enumerate(batch_y.long()):
                    rec_type = self.rec_type_mapping[rec_type_idx.item()]
                    acceptance_loss += binary_criterion(outputs[rec_type][i], torch.ones(1))
                
                # Calculate confidence loss
                confidence_loss = binary_criterion(outputs["confidence"], torch.ones_like(outputs["confidence"]))
                
                # Combine losses
                loss = rec_type_loss + acceptance_loss + confidence_loss
                val_loss += loss.item()
                
                # Store predictions and targets for recommendation type
                rec_type_preds = torch.argmax(outputs["rec_type_probs"], dim=1).cpu().numpy()
                all_rec_type_preds.extend(rec_type_preds)
                all_rec_type_targets.extend(batch_y.cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Calculate metrics for recommendation type prediction
        all_rec_type_preds = np.array(all_rec_type_preds)
        all_rec_type_targets = np.array(all_rec_type_targets)
        
        metrics = {
            "accuracy": accuracy_score(all_rec_type_targets, all_rec_type_preds),
            "precision": precision_score(all_rec_type_targets, all_rec_type_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_rec_type_targets, all_rec_type_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_rec_type_targets, all_rec_type_preds, average="weighted", zero_division=0)
        }
        
        return val_loss, metrics
    
    def predict(self, data, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate recommendations.
        
        Args:
            data: Input data (numpy array, DataFrame, or DataLoader)
            **kwargs: Additional prediction parameters
            
        Returns:
            List of recommendation dictionaries
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Recommendations may be unreliable.")
        
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
            all_recommendations = []
            with torch.no_grad():
                for batch_X, _ in data:
                    batch_recommendations = self._generate_recommendations_from_outputs(self.model(batch_X))
                    all_recommendations.extend(batch_recommendations)
            return all_recommendations
        else:
            raise ValueError("Unsupported data type. Expected DataFrame, numpy array, or DataLoader.")
        
        # Generate recommendations
        with torch.no_grad():
            outputs = self.model(data_tensor)
            recommendations = self._generate_recommendations_from_outputs(outputs)
        
        return recommendations
    
    def _generate_recommendations_from_outputs(self, outputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Generate recommendation dictionaries from model outputs.
        
        Args:
            outputs: Model output dictionary
            
        Returns:
            List of recommendation dictionaries
        """
        # Get recommendation types and confidences
        rec_type_probs = outputs["rec_type_probs"].cpu().numpy()
        rec_type_indices = np.argmax(rec_type_probs, axis=1)
        confidences = outputs["confidence"].squeeze().cpu().numpy()
        
        recommendations = []
        
        for i, rec_type_idx in enumerate(rec_type_indices):
            rec_type = self.rec_type_mapping[rec_type_idx]
            confidence = confidences[i]
            
            # Create recommendation dictionary
            recommendation = {
                "recommendation_type": rec_type,
                "confidence_score": float(confidence),
                "content": self._generate_content_for_recommendation_type(rec_type),
                "explanation": self._generate_explanation_for_recommendation_type(rec_type)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_content_for_recommendation_type(self, rec_type: str) -> Dict[str, Any]:
        """
        Generate content for a specific recommendation type.
        
        Args:
            rec_type: Recommendation type
            
        Returns:
            Content dictionary
        """
        # This is a placeholder. In a real implementation, this would generate
        # meaningful content based on user data, tasks, goals, etc.
        
        if rec_type == "task_scheduling":
            return {
                "schedule_date": "2023-06-01",
                "tasks": [
                    {
                        "task_id": "task1",
                        "title": "Complete project proposal",
                        "start_time": "09:00",
                        "end_time": "10:30",
                        "priority": "high"
                    },
                    {
                        "task_id": "task2",
                        "title": "Team meeting",
                        "start_time": "11:00",
                        "end_time": "12:00",
                        "priority": "medium"
                    }
                ]
            }
        
        elif rec_type == "task_prioritization":
            return {
                "prioritization": [
                    {
                        "task_id": "task3",
                        "title": "Review documentation",
                        "current_priority": "low",
                        "recommended_priority": "medium",
                        "reason": "Dependency for other tasks"
                    }
                ]
            }
        
        elif rec_type == "goal_suggestion":
            return {
                "suggested_goals": [
                    {
                        "title": "Learn a new programming language",
                        "category": "education",
                        "description": "Expand your skill set by learning a new programming language",
                        "estimated_duration_days": 90,
                        "reason": "Based on your completed tasks"
                    }
                ]
            }
        
        elif rec_type == "productivity_tip":
            return {
                "category": "focus",
                "tip": "Use the Pomodoro technique (25 min work, 5 min break)",
                "reason": "Based on your recent work patterns"
            }
        
        elif rec_type == "work_break_reminder":
            return {
                "work_duration_minutes": 55,
                "break_duration_minutes": 5,
                "suggested_activity": "Take a short walk",
                "reason": "You've been working continuously"
            }
        
        elif rec_type == "task_grouping":
            return {
                "task_groups": [
                    {
                        "name": "Documentation tasks",
                        "tasks": [
                            {"id": "task4", "title": "Update README"},
                            {"id": "task5", "title": "Write API documentation"}
                        ],
                        "reason": "Similar categories"
                    }
                ]
            }
        
        else:
            return {}
    
    def _generate_explanation_for_recommendation_type(self, rec_type: str) -> str:
        """
        Generate explanation for a specific recommendation type.
        
        Args:
            rec_type: Recommendation type
            
        Returns:
            Explanation string
        """
        explanations = {
            "task_scheduling": "This schedule optimizes your productivity based on your work patterns.",
            "task_prioritization": "Priorities adjusted based on deadlines and dependencies.",
            "goal_suggestion": "These goals align with your current interests and activities.",
            "productivity_tip": "This tip addresses a pattern observed in your work habits.",
            "work_break_reminder": "Taking breaks at optimal intervals helps maintain peak performance.",
            "task_grouping": "Grouping these tasks can save you setup and context-switching time."
        }
        
        return explanations.get(rec_type, "")
    
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
        
        # Set up loss functions
        rec_type_criterion = nn.CrossEntropyLoss()
        binary_criterion = nn.BCELoss()
        
        # Evaluate
        val_loss, metrics = self._evaluate_during_training(test_loader, rec_type_criterion, binary_criterion)
        
        # Log evaluation results
        app_logger.info(f"Evaluation results: {metrics}")
        
        # Update metadata
        self.metadata["performance_metrics"] = metrics
        
        return metrics
    
    def generate_recommendations(self, user_id: str, tasks: List[Task], goals: List[Goal], user_preference: UserPreference) -> List[Recommendation]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            tasks: List of user's tasks
            goals: List of user's goals
            user_preference: User's preferences
            
        Returns:
            List of Recommendation objects
        """
        # This is a more realistic recommendation generation method that would be used in production
        # It would use the trained model to generate recommendations based on user data
        
        # In a real implementation, you would:
        # 1. Extract features from the user's tasks, goals, and preferences
        # 2. Prepare the features for the model
        # 3. Use the model to predict recommendation types and confidences
        # 4. Generate detailed content for each recommendation
        
        # For now, we'll just return a placeholder implementation
        from datetime import datetime
        import uuid
        
        # Generate 3 recommendations
        recommendations = []
        
        for _ in range(3):
            # Randomly select a recommendation type
            rec_type_idx = np.random.randint(0, 6)
            rec_type = self.rec_type_mapping[rec_type_idx]
            
            # Generate content and explanation
            content = self._generate_content_for_recommendation_type(rec_type)
            explanation = self._generate_explanation_for_recommendation_type(rec_type)
            
            # Create recommendation
            recommendation = Recommendation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=datetime.now(),
                recommendation_type=rec_type,
                content=content,
                confidence_score=np.random.uniform(0.7, 0.95),
                explanation=explanation,
                is_applied=False
            )
            
            recommendations.append(recommendation)
        
        return recommendations