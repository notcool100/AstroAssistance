"""
Training script for AstroAssistance models.
"""
import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow
import wandb

from src.core.config import config_manager
from src.core.logger import app_logger
from src.data_processing.data_processor import DataProcessor, TaskDataset, create_data_loaders
from src.models.task_completion_model import TaskCompletionModel
from src.models.recommendation_model import RecommendationModel
from src.models.reinforcement_learning_model import ReinforcementLearningModel


def setup_experiment_tracking(experiment_name: str, run_name: str = None, tracking_uri: str = None) -> None:
    """
    Set up experiment tracking with MLflow.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run
        tracking_uri: MLflow tracking URI
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start run
    mlflow.start_run(run_name=run_name)
    
    # Log config parameters
    mlflow.log_params({
        "model_type": config_manager.get("models.base_model.type"),
        "embedding_dim": config_manager.get("models.base_model.embedding_dim"),
        "num_heads": config_manager.get("models.base_model.num_heads"),
        "num_layers": config_manager.get("models.base_model.num_layers"),
        "dropout": config_manager.get("models.base_model.dropout"),
        "batch_size": config_manager.get("training.batch_size"),
        "learning_rate": config_manager.get("training.learning_rate"),
        "epochs": config_manager.get("training.epochs")
    })


def train_task_completion_model(data_processor: DataProcessor, use_wandb: bool = False) -> TaskCompletionModel:
    """
    Train the task completion prediction model.
    
    Args:
        data_processor: Data processor instance
        use_wandb: Whether to use Weights & Biases for tracking
        
    Returns:
        Trained TaskCompletionModel
    """
    app_logger.info("Training task completion model")
    
    # Load training data
    train_data = data_processor.load_processed_data("task_completion_train.pkl")
    val_data = data_processor.load_processed_data("task_completion_val.pkl")
    test_data = data_processor.load_processed_data("task_completion_test.pkl")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Create data loaders
    batch_size = config_manager.get("training.batch_size", 32)
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = TaskCompletionModel(input_dim=input_dim)
    
    # Build model
    hidden_dims = config_manager.get("models.task_classifier.hidden_dims", [128, 64])
    dropout = config_manager.get("models.task_classifier.dropout", 0.2)
    model.build(hidden_dims=hidden_dims, dropout=dropout)
    
    # Set up experiment tracking
    if use_wandb:
        wandb.init(
            project="astro-assistance",
            name="task_completion_model",
            config={
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": config_manager.get("training.learning_rate", 0.001),
                "epochs": config_manager.get("training.epochs", 50)
            }
        )
    
    # Train model
    history = model.train(
        (X_train, y_train),
        (X_val, y_val),
        epochs=config_manager.get("training.epochs", 50),
        learning_rate=config_manager.get("training.learning_rate", 0.001),
        batch_size=batch_size
    )
    
    # Evaluate model
    metrics = model.evaluate((X_test, y_test))
    
    # Log metrics
    if use_wandb:
        wandb.log(metrics)
        
        # Log training history
        for i, (train_loss, val_loss, val_acc, val_prec, val_rec, val_f1) in enumerate(zip(
            history["train_loss"],
            history["val_loss"],
            history["val_accuracy"],
            history["val_precision"],
            history["val_recall"],
            history["val_f1"]
        )):
            wandb.log({
                "epoch": i + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1
            })
    
    # Save model
    model.save()
    
    # Finish tracking
    if use_wandb:
        wandb.finish()
    
    return model


def train_recommendation_model(data_processor: DataProcessor, use_wandb: bool = False) -> RecommendationModel:
    """
    Train the recommendation model.
    
    Args:
        data_processor: Data processor instance
        use_wandb: Whether to use Weights & Biases for tracking
        
    Returns:
        Trained RecommendationModel
    """
    app_logger.info("Training recommendation model")
    
    # Load training data
    train_data = data_processor.load_processed_data("recommendation_acceptance_train.pkl")
    val_data = data_processor.load_processed_data("recommendation_acceptance_val.pkl")
    test_data = data_processor.load_processed_data("recommendation_acceptance_test.pkl")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Create data loaders
    batch_size = config_manager.get("training.batch_size", 32)
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = RecommendationModel(input_dim=input_dim)
    
    # Build model
    hidden_dims = config_manager.get("models.recommendation_model.hidden_dims", [256, 128, 64])
    dropout = config_manager.get("models.recommendation_model.dropout", 0.3)
    model.build(hidden_dims=hidden_dims, dropout=dropout)
    
    # Set up experiment tracking
    if use_wandb:
        wandb.init(
            project="astro-assistance",
            name="recommendation_model",
            config={
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": config_manager.get("training.learning_rate", 0.001),
                "epochs": config_manager.get("training.epochs", 50)
            }
        )
    
    # Train model
    history = model.train(
        (X_train, y_train),
        (X_val, y_val),
        epochs=config_manager.get("training.epochs", 50),
        learning_rate=config_manager.get("training.learning_rate", 0.001),
        batch_size=batch_size
    )
    
    # Evaluate model
    metrics = model.evaluate((X_test, y_test))
    
    # Log metrics
    if use_wandb:
        wandb.log(metrics)
        
        # Log training history
        for i, (train_loss, val_loss, val_acc, val_prec, val_rec, val_f1) in enumerate(zip(
            history["train_loss"],
            history["val_loss"],
            history["val_accuracy"],
            history["val_precision"],
            history["val_recall"],
            history["val_f1"]
        )):
            wandb.log({
                "epoch": i + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1
            })
    
    # Save model
    model.save()
    
    # Finish tracking
    if use_wandb:
        wandb.finish()
    
    return model


def train_reinforcement_learning_model(use_wandb: bool = False) -> ReinforcementLearningModel:
    """
    Train the reinforcement learning model.
    
    Args:
        use_wandb: Whether to use Weights & Biases for tracking
        
    Returns:
        Trained ReinforcementLearningModel
    """
    app_logger.info("Training reinforcement learning model")
    
    # Initialize model
    model = ReinforcementLearningModel()
    
    # Build model
    algorithm = config_manager.get("models.reinforcement_learning.algorithm", "ppo")
    gamma = config_manager.get("models.reinforcement_learning.gamma", 0.99)
    learning_rate = config_manager.get("models.reinforcement_learning.learning_rate", 0.0003)
    model.build(algorithm=algorithm, gamma=gamma, learning_rate=learning_rate)
    
    # Set up experiment tracking
    if use_wandb:
        wandb.init(
            project="astro-assistance",
            name="reinforcement_learning_model",
            config={
                "algorithm": algorithm,
                "gamma": gamma,
                "learning_rate": learning_rate,
                "total_timesteps": 100000
            }
        )
    
    # Train model
    history = model.train(total_timesteps=100000, eval_freq=10000)
    
    # Evaluate model
    metrics = model.evaluate(n_eval_episodes=20)
    
    # Log metrics
    if use_wandb:
        wandb.log(metrics)
        
        # Log training history
        for i, (mean_reward, std_reward, timestep) in enumerate(zip(
            history["mean_reward"],
            history["std_reward"],
            history["timesteps"]
        )):
            wandb.log({
                "timestep": timestep,
                "mean_reward": mean_reward,
                "std_reward": std_reward
            })
    
    # Save model
    model.save()
    
    # Finish tracking
    if use_wandb:
        wandb.finish()
    
    return model


def main():
    """Main function to train all models."""
    parser = argparse.ArgumentParser(description="Train AstroAssistance models")
    parser.add_argument("--model", type=str, choices=["all", "task", "recommendation", "rl"], default="all",
                        help="Model to train (default: all)")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for tracking")
    parser.add_argument("--mlflow", action="store_true", help="Use MLflow for tracking")
    parser.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI")
    args = parser.parse_args()
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Set up MLflow tracking if requested
    if args.mlflow:
        setup_experiment_tracking("astro-assistance", tracking_uri=args.mlflow_uri)
    
    # Train models
    if args.model in ["all", "task"]:
        task_model = train_task_completion_model(data_processor, use_wandb=args.wandb)
    
    if args.model in ["all", "recommendation"]:
        rec_model = train_recommendation_model(data_processor, use_wandb=args.wandb)
    
    if args.model in ["all", "rl"]:
        rl_model = train_reinforcement_learning_model(use_wandb=args.wandb)
    
    # End MLflow run if active
    if args.mlflow:
        mlflow.end_run()
    
    app_logger.info("Model training complete")


if __name__ == "__main__":
    main()