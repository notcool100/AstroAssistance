"""
Model evaluation utilities for AstroAssistance.
"""
import os
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, roc_curve, auc
)
import torch
from torch.utils.data import DataLoader

from src.core.config import config_manager
from src.core.logger import app_logger


class ModelEvaluator:
    """Evaluates model performance with various metrics and visualizations."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if output_dir is None:
            self.output_dir = os.path.join(self.project_root, "evaluation")
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate_classifier(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None, 
                           class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for binary or multiclass)
            class_names: Names of classes
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Determine if binary or multiclass
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_classes) <= 2
        
        # Set default class names if not provided
        if class_names is None:
            if is_binary:
                class_names = ["Negative", "Positive"]
            else:
                class_names = [f"Class {i}" for i in range(len(unique_classes))]
        
        # Calculate basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }
        
        # Calculate class-specific metrics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["class_report"] = class_report
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        
        # Calculate ROC AUC if probabilities are provided
        if y_prob is not None:
            if is_binary:
                # Ensure y_prob is for the positive class
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    y_prob = y_prob[:, 1]
                
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
                
                # Calculate Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                metrics["pr_curve"] = {"precision": precision, "recall": recall}
                metrics["pr_auc"] = auc(recall, precision)
            else:
                # Multiclass ROC AUC
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
                except Exception as e:
                    app_logger.warning(f"Could not calculate multiclass ROC AUC: {str(e)}")
        
        # Generate visualizations
        self._plot_confusion_matrix(cm, class_names)
        
        if y_prob is not None and is_binary:
            self._plot_roc_curve(fpr, tpr, metrics["roc_auc"])
            self._plot_precision_recall_curve(precision, recall, metrics["pr_auc"])
        
        return metrics
    
    def evaluate_regressor(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate metrics
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        
        # Calculate additional metrics
        metrics["mape"] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics["median_ae"] = np.median(np.abs(y_true - y_pred))
        
        # Generate visualizations
        self._plot_prediction_scatter(y_true, y_pred)
        self._plot_residuals(y_true, y_pred)
        self._plot_error_distribution(y_true, y_pred)
        
        return metrics
    
    def evaluate_pytorch_model(self, model: torch.nn.Module, data_loader: DataLoader, 
                              criterion: torch.nn.Module, device: str = "cpu",
                              is_binary: bool = True) -> Dict[str, Any]:
        """
        Evaluate a PyTorch model.
        
        Args:
            model: PyTorch model
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to run evaluation on
            is_binary: Whether the task is binary classification
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        all_losses = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    # For models that return a dictionary of outputs
                    if "logits" in outputs:
                        logits = outputs["logits"]
                    elif "predictions" in outputs:
                        logits = outputs["predictions"]
                    else:
                        # Use the first value in the dictionary
                        logits = list(outputs.values())[0]
                else:
                    logits = outputs
                
                # Calculate loss
                if is_binary:
                    # Ensure y_batch has the right shape for binary classification
                    y_batch = y_batch.float().view(-1, 1)
                    loss = criterion(logits, y_batch)
                    
                    # Get probabilities and predictions
                    y_prob = torch.sigmoid(logits).cpu().numpy()
                    y_pred = (y_prob >= 0.5).astype(int)
                else:
                    loss = criterion(logits, y_batch)
                    
                    # Get probabilities and predictions
                    y_prob = torch.softmax(logits, dim=1).cpu().numpy()
                    y_pred = np.argmax(y_prob, axis=1)
                
                # Store results
                all_losses.append(loss.item())
                all_y_true.extend(y_batch.cpu().numpy())
                all_y_pred.extend(y_pred)
                all_y_prob.extend(y_prob)
        
        # Convert lists to arrays
        all_y_true = np.array(all_y_true).reshape(-1)
        all_y_pred = np.array(all_y_pred).reshape(-1)
        all_y_prob = np.array(all_y_prob)
        
        # Ensure y_prob has the right shape for binary classification
        if is_binary and all_y_prob.ndim == 1:
            all_y_prob = all_y_prob.reshape(-1, 1)
            all_y_prob = np.hstack([1 - all_y_prob, all_y_prob])
        
        # Calculate average loss
        avg_loss = np.mean(all_losses)
        
        # Evaluate model
        if is_binary:
            metrics = self.evaluate_classifier(all_y_true, all_y_pred, all_y_prob)
        else:
            metrics = self.evaluate_classifier(all_y_true, all_y_pred, all_y_prob)
        
        # Add loss to metrics
        metrics["loss"] = avg_loss
        
        return metrics
    
    def evaluate_reinforcement_learning_model(self, model, env, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate a reinforcement learning model.
        
        Args:
            model: RL model (e.g., from stable-baselines3)
            env: Environment to evaluate in
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Initialize metrics
        episode_rewards = []
        episode_lengths = []
        
        # Run evaluation episodes
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        # Calculate metrics
        metrics = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths)
        }
        
        # Generate visualizations
        self._plot_reward_distribution(episode_rewards)
        
        return metrics
    
    def evaluate_recommendation_model(self, model, test_data: Union[pd.DataFrame, np.ndarray], 
                                     user_ids: List[str], ground_truth: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a recommendation model.
        
        Args:
            model: Recommendation model
            test_data: Test data
            user_ids: List of user IDs to generate recommendations for
            ground_truth: Dictionary mapping user IDs to lists of ground truth items
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate recommendations for each user
        user_recommendations = {}
        
        for user_id in user_ids:
            # Filter test data for this user
            if isinstance(test_data, pd.DataFrame) and "user_id" in test_data.columns:
                user_data = test_data[test_data["user_id"] == user_id]
            else:
                user_data = test_data
            
            # Generate recommendations
            recommendations = model.predict(user_data)
            
            # Store recommendations
            user_recommendations[user_id] = recommendations
        
        # Calculate metrics
        metrics = {
            "num_users": len(user_ids),
            "avg_recommendations_per_user": np.mean([len(recs) for recs in user_recommendations.values()])
        }
        
        # Calculate recommendation type distribution
        rec_types = []
        for recs in user_recommendations.values():
            for rec in recs:
                if isinstance(rec, dict) and "recommendation_type" in rec:
                    rec_types.append(rec["recommendation_type"])
        
        if rec_types:
            from collections import Counter
            type_counts = Counter(rec_types)
            metrics["recommendation_type_distribution"] = {
                rec_type: count / len(rec_types) for rec_type, count in type_counts.items()
            }
        
        # Calculate relevance metrics if ground truth is provided
        if ground_truth is not None:
            precision_at_k = []
            recall_at_k = []
            
            for user_id in user_ids:
                if user_id in ground_truth:
                    # Get recommendations and ground truth for this user
                    recs = user_recommendations[user_id]
                    truth = ground_truth[user_id]
                    
                    # Extract recommendation IDs
                    if isinstance(recs[0], dict) and "id" in recs[0]:
                        rec_ids = [rec["id"] for rec in recs]
                    else:
                        rec_ids = [str(rec) for rec in recs]
                    
                    # Calculate precision@k and recall@k
                    k = len(rec_ids)
                    relevant_recs = [rec_id for rec_id in rec_ids if rec_id in truth]
                    
                    precision = len(relevant_recs) / k if k > 0 else 0
                    recall = len(relevant_recs) / len(truth) if len(truth) > 0 else 0
                    
                    precision_at_k.append(precision)
                    recall_at_k.append(recall)
            
            if precision_at_k:
                metrics["mean_precision_at_k"] = np.mean(precision_at_k)
                metrics["mean_recall_at_k"] = np.mean(recall_at_k)
        
        # Generate visualizations
        if "recommendation_type_distribution" in metrics:
            self._plot_recommendation_type_distribution(metrics["recommendation_type_distribution"])
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()
    
    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under the ROC curve
        """
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"))
        plt.close()
    
    def _plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray, pr_auc: float) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under the precision-recall curve
        """
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "precision_recall_curve.png"))
        plt.close()
    
    def _plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot scatter plot of true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=2)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("True vs Predicted Values")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "prediction_scatter.png"))
        plt.close()
    
    def _plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "residuals.png"))
        plt.close()
    
    def _plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot error distribution.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        errors = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        sns.histplot(errors, kde=True)
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_distribution.png"))
        plt.close()
    
    def _plot_reward_distribution(self, rewards: List[float]) -> None:
        """
        Plot reward distribution for RL model.
        
        Args:
            rewards: List of episode rewards
        """
        plt.figure(figsize=(10, 8))
        sns.histplot(rewards, kde=True)
        plt.axvline(x=np.mean(rewards), color="r", linestyle="--", label=f"Mean: {np.mean(rewards):.2f}")
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "reward_distribution.png"))
        plt.close()
    
    def _plot_recommendation_type_distribution(self, type_distribution: Dict[str, float]) -> None:
        """
        Plot recommendation type distribution.
        
        Args:
            type_distribution: Dictionary mapping recommendation types to proportions
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(type_distribution.keys()), y=list(type_distribution.values()))
        plt.xlabel("Recommendation Type")
        plt.ylabel("Proportion")
        plt.title("Recommendation Type Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "recommendation_type_distribution.png"))
        plt.close()
    
    def generate_evaluation_report(self, metrics: Dict[str, Any], model_name: str) -> str:
        """
        Generate a text report of evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model
            
        Returns:
            Report as a string
        """
        report = f"Evaluation Report for {model_name}\n"
        report += "=" * 50 + "\n\n"
        
        # Add metrics to report
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                # Skip complex nested dictionaries
                if metric_name in ["class_report", "confusion_matrix", "roc_curve", "pr_curve"]:
                    continue
                
                # Add nested metrics
                report += f"{metric_name}:\n"
                for sub_name, sub_value in metric_value.items():
                    if isinstance(sub_value, (int, float)):
                        report += f"  {sub_name}: {sub_value:.4f}\n"
                    else:
                        report += f"  {sub_name}: {sub_value}\n"
            elif isinstance(metric_value, (int, float)):
                report += f"{metric_name}: {metric_value:.4f}\n"
            else:
                report += f"{metric_name}: {metric_value}\n"
        
        # Save report to file
        report_path = os.path.join(self.output_dir, f"{model_name}_evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        return report


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classifier(y_test, y_pred, y_prob)
    
    # Generate report
    report = evaluator.generate_evaluation_report(metrics, "RandomForest")
    print(report)