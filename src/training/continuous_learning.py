"""
Continuous learning system for AstroAssistance.

This module implements the continuous learning system that allows models to improve over time
based on user feedback and new data.
"""
import os
import time
import json
import pickle
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import threading
import queue
import numpy as np
import pandas as pd
import torch

from src.core.config import config_manager
from src.core.logger import app_logger
from src.core.data_types import (
    Task, TaskPriority, TaskStatus, TaskCategory,
    Reminder, Goal, UserPreference, UserActivity,
    LearningFeedback, Recommendation, RecurrencePattern
)
from src.data_processing.data_processor import DataProcessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.models.task_completion_model import TaskCompletionModel
from src.models.recommendation_model import RecommendationModel
from src.models.reinforcement_learning_model import ReinforcementLearningModel
from src.training.model_evaluation import ModelEvaluator


class ContinuousLearningSystem:
    """
    System for continuous model improvement based on feedback and new data.
    
    This class implements a background process that periodically:
    1. Collects new data and feedback
    2. Processes and integrates the new data
    3. Retrains models when sufficient new data is available
    4. Evaluates model performance
    5. Deploys improved models
    """
    
    def __init__(self):
        """Initialize the continuous learning system."""
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.project_root, "data")
        self.models_dir = os.path.join(self.project_root, "models")
        self.feedback_dir = os.path.join(self.data_dir, "feedback")
        
        # Create directories if they don't exist
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_evaluator = ModelEvaluator()
        
        # Initialize models
        self.task_completion_model = None
        self.recommendation_model = None
        self.reinforcement_learning_model = None
        
        # Initialize feedback queue
        self.feedback_queue = queue.Queue()
        
        # Initialize tracking variables
        self.last_retrain_time = datetime.now()
        self.feedback_count_since_last_retrain = 0
        self.new_data_count_since_last_retrain = 0
        
        # Load configuration
        self.retrain_interval_hours = config_manager.get("continuous_learning.retrain_interval_hours", 24)
        self.min_feedback_for_retrain = config_manager.get("continuous_learning.min_feedback_for_retrain", 50)
        self.min_new_data_for_retrain = config_manager.get("continuous_learning.min_new_data_for_retrain", 100)
        self.max_feedback_buffer_size = config_manager.get("continuous_learning.max_feedback_buffer_size", 1000)
        
        # Initialize background thread
        self.running = False
        self.background_thread = None
    
    def start(self) -> None:
        """Start the continuous learning system."""
        if self.running:
            app_logger.warning("Continuous learning system is already running")
            return
        
        # Load models
        self._load_models()
        
        # Start background thread
        self.running = True
        self.background_thread = threading.Thread(target=self._background_process, daemon=True)
        self.background_thread.start()
        
        app_logger.info("Continuous learning system started")
    
    def stop(self) -> None:
        """Stop the continuous learning system."""
        if not self.running:
            app_logger.warning("Continuous learning system is not running")
            return
        
        # Stop background thread
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        app_logger.info("Continuous learning system stopped")
    
    def add_feedback(self, feedback: LearningFeedback) -> None:
        """
        Add feedback to the learning system.
        
        Args:
            feedback: Feedback object
        """
        # Add to queue
        self.feedback_queue.put(feedback)
        
        # Save feedback to file
        self._save_feedback(feedback)
        
        # Update counter
        self.feedback_count_since_last_retrain += 1
        
        app_logger.debug(f"Added feedback: {feedback.id}")
    
    def add_task_data(self, task: Task) -> None:
        """
        Add task data to the learning system.
        
        Args:
            task: Task object
        """
        # Save task to file
        self._save_task(task)
        
        # Update counter
        self.new_data_count_since_last_retrain += 1
        
        app_logger.debug(f"Added task data: {task.id}")
    
    def add_user_activity(self, activity: UserActivity) -> None:
        """
        Add user activity data to the learning system.
        
        Args:
            activity: UserActivity object
        """
        # Save activity to file
        self._save_activity(activity)
        
        # Update counter
        self.new_data_count_since_last_retrain += 1
        
        app_logger.debug(f"Added user activity: {activity.id}")
    
    def _background_process(self) -> None:
        """Background process for continuous learning."""
        while self.running:
            try:
                # Check if it's time to retrain
                should_retrain = self._should_retrain()
                
                if should_retrain:
                    app_logger.info("Starting model retraining process")
                    
                    # Process new data
                    self._process_new_data()
                    
                    # Retrain models
                    self._retrain_models()
                    
                    # Evaluate models
                    self._evaluate_models()
                    
                    # Deploy models
                    self._deploy_models()
                    
                    # Update tracking variables
                    self.last_retrain_time = datetime.now()
                    self.feedback_count_since_last_retrain = 0
                    self.new_data_count_since_last_retrain = 0
                    
                    app_logger.info("Model retraining process completed")
                
                # Process feedback queue
                self._process_feedback_queue()
                
                # Sleep for a while
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                app_logger.error(f"Error in continuous learning background process: {str(e)}")
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _should_retrain(self) -> bool:
        """
        Determine if models should be retrained.
        
        Returns:
            True if models should be retrained, False otherwise
        """
        # Check if enough time has passed since last retrain
        time_since_last_retrain = datetime.now() - self.last_retrain_time
        time_threshold_met = time_since_last_retrain.total_seconds() >= self.retrain_interval_hours * 3600
        
        # Check if enough feedback has been collected
        feedback_threshold_met = self.feedback_count_since_last_retrain >= self.min_feedback_for_retrain
        
        # Check if enough new data has been collected
        data_threshold_met = self.new_data_count_since_last_retrain >= self.min_new_data_for_retrain
        
        # Determine if we should retrain
        should_retrain = time_threshold_met and (feedback_threshold_met or data_threshold_met)
        
        if should_retrain:
            app_logger.info(
                f"Retraining triggered: Time threshold: {time_threshold_met}, "
                f"Feedback threshold: {feedback_threshold_met} ({self.feedback_count_since_last_retrain}/{self.min_feedback_for_retrain}), "
                f"Data threshold: {data_threshold_met} ({self.new_data_count_since_last_retrain}/{self.min_new_data_for_retrain})"
            )
        
        return should_retrain
    
    def _process_feedback_queue(self) -> None:
        """Process feedback from the queue."""
        # Process up to 100 feedback items at a time
        for _ in range(min(100, self.feedback_queue.qsize())):
            try:
                # Get feedback from queue
                feedback = self.feedback_queue.get_nowait()
                
                # Process feedback
                self._process_feedback(feedback)
                
                # Mark as done
                self.feedback_queue.task_done()
            
            except queue.Empty:
                break
            except Exception as e:
                app_logger.error(f"Error processing feedback: {str(e)}")
    
    def _process_feedback(self, feedback: LearningFeedback) -> None:
        """
        Process a single feedback item.
        
        Args:
            feedback: Feedback object
        """
        # Extract recommendation type
        recommendation_id = feedback.recommendation_id
        feedback_type = feedback.feedback_type
        
        # Update recommendation model weights based on feedback
        if self.recommendation_model and self.recommendation_model.is_trained:
            try:
                # In a real implementation, this would update model weights
                # For now, we'll just log the feedback
                app_logger.debug(f"Processing feedback for recommendation {recommendation_id}: {feedback_type}")
            except Exception as e:
                app_logger.error(f"Error updating recommendation model weights: {str(e)}")
        
        # Update reinforcement learning model based on feedback
        if self.reinforcement_learning_model and self.reinforcement_learning_model.is_trained:
            try:
                # In a real implementation, this would update RL model
                # For now, we'll just log the feedback
                app_logger.debug(f"Processing feedback for RL model: {feedback_type}")
            except Exception as e:
                app_logger.error(f"Error updating RL model: {str(e)}")
    
    def _process_new_data(self) -> None:
        """Process new data for model training."""
        app_logger.info("Processing new data")
        
        try:
            # Prepare all datasets
            dataset_paths = self.data_processor.prepare_all_datasets()
            
            # Create enhanced features
            tasks_df = self.data_processor.load_processed_data("processed_tasks.csv")
            activities_df = self.data_processor.load_processed_data("processed_activities.csv")
            recommendations_df = self.data_processor.load_processed_data("processed_recommendations.csv")
            
            # Load user preferences
            preferences_data = self.data_processor.load_json_data(
                os.path.join(self.data_processor.synthetic_data_dir, "user_preferences.json")
            )
            preferences_df = pd.DataFrame(preferences_data)
            
            # Create enhanced features
            enhanced_dfs = self.feature_engineer.create_all_features(
                tasks_df, activities_df, recommendations_df, preferences_df
            )
            
            # Save enhanced features
            for name, df in enhanced_dfs.items():
                self.data_processor.save_processed_data(df, f"enhanced_{name}.csv")
            
            app_logger.info("New data processed successfully")
        
        except Exception as e:
            app_logger.error(f"Error processing new data: {str(e)}")
    
    def _retrain_models(self) -> None:
        """Retrain models with new data."""
        app_logger.info("Retraining models")
        
        try:
            # Retrain task completion model
            self._retrain_task_completion_model()
            
            # Retrain recommendation model
            self._retrain_recommendation_model()
            
            # Retrain reinforcement learning model
            self._retrain_reinforcement_learning_model()
            
            app_logger.info("Models retrained successfully")
        
        except Exception as e:
            app_logger.error(f"Error retraining models: {str(e)}")
    
    def _retrain_task_completion_model(self) -> None:
        """Retrain the task completion model."""
        app_logger.info("Retraining task completion model")
        
        try:
            # Load training data
            train_data = self.data_processor.load_processed_data("task_completion_train.pkl")
            val_data = self.data_processor.load_processed_data("task_completion_val.pkl")
            
            # Initialize model if not already loaded
            if self.task_completion_model is None:
                X_train, _ = train_data
                input_dim = X_train.shape[1]
                self.task_completion_model = TaskCompletionModel(input_dim=input_dim)
                self.task_completion_model.build()
            
            # Train model
            self.task_completion_model.train(train_data, val_data)
            
            # Save model
            self.task_completion_model.save()
            
            app_logger.info("Task completion model retrained successfully")
        
        except Exception as e:
            app_logger.error(f"Error retraining task completion model: {str(e)}")
    
    def _retrain_recommendation_model(self) -> None:
        """Retrain the recommendation model."""
        app_logger.info("Retraining recommendation model")
        
        try:
            # Load training data
            train_data = self.data_processor.load_processed_data("recommendation_acceptance_train.pkl")
            val_data = self.data_processor.load_processed_data("recommendation_acceptance_val.pkl")
            
            # Initialize model if not already loaded
            if self.recommendation_model is None:
                X_train, _ = train_data
                input_dim = X_train.shape[1]
                self.recommendation_model = RecommendationModel(input_dim=input_dim)
                self.recommendation_model.build()
            
            # Train model
            self.recommendation_model.train(train_data, val_data)
            
            # Save model
            self.recommendation_model.save()
            
            app_logger.info("Recommendation model retrained successfully")
        
        except Exception as e:
            app_logger.error(f"Error retraining recommendation model: {str(e)}")
    
    def _retrain_reinforcement_learning_model(self) -> None:
        """Retrain the reinforcement learning model."""
        app_logger.info("Retraining reinforcement learning model")
        
        try:
            # Initialize model if not already loaded
            if self.reinforcement_learning_model is None:
                self.reinforcement_learning_model = ReinforcementLearningModel()
                self.reinforcement_learning_model.build()
            
            # Train model
            self.reinforcement_learning_model.train(total_timesteps=10000)  # Reduced for faster retraining
            
            # Save model
            self.reinforcement_learning_model.save()
            
            app_logger.info("Reinforcement learning model retrained successfully")
        
        except Exception as e:
            app_logger.error(f"Error retraining reinforcement learning model: {str(e)}")
    
    def _evaluate_models(self) -> None:
        """Evaluate models after retraining."""
        app_logger.info("Evaluating models")
        
        try:
            # Evaluate task completion model
            self._evaluate_task_completion_model()
            
            # Evaluate recommendation model
            self._evaluate_recommendation_model()
            
            # Evaluate reinforcement learning model
            self._evaluate_reinforcement_learning_model()
            
            app_logger.info("Models evaluated successfully")
        
        except Exception as e:
            app_logger.error(f"Error evaluating models: {str(e)}")
    
    def _evaluate_task_completion_model(self) -> None:
        """Evaluate the task completion model."""
        if self.task_completion_model is None or not self.task_completion_model.is_trained:
            app_logger.warning("Task completion model is not trained, skipping evaluation")
            return
        
        try:
            # Load test data
            test_data = self.data_processor.load_processed_data("task_completion_test.pkl")
            
            # Evaluate model
            metrics = self.task_completion_model.evaluate(test_data)
            
            # Generate evaluation report
            self.model_evaluator.generate_evaluation_report(metrics, "task_completion_model")
            
            app_logger.info(f"Task completion model evaluation: {metrics}")
        
        except Exception as e:
            app_logger.error(f"Error evaluating task completion model: {str(e)}")
    
    def _evaluate_recommendation_model(self) -> None:
        """Evaluate the recommendation model."""
        if self.recommendation_model is None or not self.recommendation_model.is_trained:
            app_logger.warning("Recommendation model is not trained, skipping evaluation")
            return
        
        try:
            # Load test data
            test_data = self.data_processor.load_processed_data("recommendation_acceptance_test.pkl")
            
            # Evaluate model
            metrics = self.recommendation_model.evaluate(test_data)
            
            # Generate evaluation report
            self.model_evaluator.generate_evaluation_report(metrics, "recommendation_model")
            
            app_logger.info(f"Recommendation model evaluation: {metrics}")
        
        except Exception as e:
            app_logger.error(f"Error evaluating recommendation model: {str(e)}")
    
    def _evaluate_reinforcement_learning_model(self) -> None:
        """Evaluate the reinforcement learning model."""
        if self.reinforcement_learning_model is None or not self.reinforcement_learning_model.is_trained:
            app_logger.warning("Reinforcement learning model is not trained, skipping evaluation")
            return
        
        try:
            # Evaluate model
            metrics = self.reinforcement_learning_model.evaluate(n_eval_episodes=10)
            
            # Generate evaluation report
            self.model_evaluator.generate_evaluation_report(metrics, "reinforcement_learning_model")
            
            app_logger.info(f"Reinforcement learning model evaluation: {metrics}")
        
        except Exception as e:
            app_logger.error(f"Error evaluating reinforcement learning model: {str(e)}")
    
    def _deploy_models(self) -> None:
        """Deploy trained models to production."""
        app_logger.info("Deploying models")
        
        try:
            # In a real implementation, this would deploy models to a production environment
            # For now, we'll just copy the models to a "deployed" directory
            deployed_dir = os.path.join(self.models_dir, "deployed")
            os.makedirs(deployed_dir, exist_ok=True)
            
            # Copy task completion model
            if self.task_completion_model and self.task_completion_model.is_trained:
                model_path = self.task_completion_model.get_model_path()
                if os.path.exists(model_path):
                    import shutil
                    dest_path = os.path.join(deployed_dir, os.path.basename(model_path))
                    shutil.copy2(model_path, dest_path)
                    app_logger.info(f"Deployed task completion model to {dest_path}")
            
            # Copy recommendation model
            if self.recommendation_model and self.recommendation_model.is_trained:
                model_path = self.recommendation_model.get_model_path()
                if os.path.exists(model_path):
                    import shutil
                    dest_path = os.path.join(deployed_dir, os.path.basename(model_path))
                    shutil.copy2(model_path, dest_path)
                    app_logger.info(f"Deployed recommendation model to {dest_path}")
            
            # Copy reinforcement learning model
            if self.reinforcement_learning_model and self.reinforcement_learning_model.is_trained:
                model_path = self.reinforcement_learning_model.get_model_path()
                if os.path.exists(model_path):
                    import shutil
                    dest_path = os.path.join(deployed_dir, os.path.basename(model_path))
                    shutil.copy2(model_path, dest_path)
                    app_logger.info(f"Deployed reinforcement learning model to {dest_path}")
            
            app_logger.info("Models deployed successfully")
        
        except Exception as e:
            app_logger.error(f"Error deploying models: {str(e)}")
    
    def _load_models(self) -> None:
        """Load trained models."""
        app_logger.info("Loading models")
        
        try:
            # Load task completion model
            try:
                self.task_completion_model = TaskCompletionModel()
                self.task_completion_model.load()
                app_logger.info("Task completion model loaded successfully")
            except Exception as e:
                app_logger.warning(f"Could not load task completion model: {str(e)}")
                self.task_completion_model = None
            
            # Load recommendation model
            try:
                self.recommendation_model = RecommendationModel()
                self.recommendation_model.load()
                app_logger.info("Recommendation model loaded successfully")
            except Exception as e:
                app_logger.warning(f"Could not load recommendation model: {str(e)}")
                self.recommendation_model = None
            
            # Load reinforcement learning model
            try:
                self.reinforcement_learning_model = ReinforcementLearningModel()
                self.reinforcement_learning_model.load()
                app_logger.info("Reinforcement learning model loaded successfully")
            except Exception as e:
                app_logger.warning(f"Could not load reinforcement learning model: {str(e)}")
                self.reinforcement_learning_model = None
        
        except Exception as e:
            app_logger.error(f"Error loading models: {str(e)}")
    
    def _save_feedback(self, feedback: LearningFeedback) -> None:
        """
        Save feedback to file.
        
        Args:
            feedback: Feedback object
        """
        # Create feedback directory if it doesn't exist
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Save feedback to file
        file_path = os.path.join(self.feedback_dir, f"{feedback.id}.json")
        
        with open(file_path, "w") as f:
            json.dump(feedback.dict(), f, default=str)
    
    def _save_task(self, task: Task) -> None:
        """
        Save task to file.
        
        Args:
            task: Task object
        """
        # Create tasks directory if it doesn't exist
        tasks_dir = os.path.join(self.data_dir, "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        # Save task to file
        file_path = os.path.join(tasks_dir, f"{task.id}.json")
        
        with open(file_path, "w") as f:
            json.dump(task.dict(), f, default=str)
    
    def _save_activity(self, activity: UserActivity) -> None:
        """
        Save user activity to file.
        
        Args:
            activity: UserActivity object
        """
        # Create activities directory if it doesn't exist
        activities_dir = os.path.join(self.data_dir, "activities")
        os.makedirs(activities_dir, exist_ok=True)
        
        # Save activity to file
        file_path = os.path.join(activities_dir, f"{activity.id}.json")
        
        with open(file_path, "w") as f:
            json.dump(activity.dict(), f, default=str)


# Singleton instance
continuous_learning_system = ContinuousLearningSystem()


def start_continuous_learning():
    """Start the continuous learning system."""
    continuous_learning_system.start()


def stop_continuous_learning():
    """Stop the continuous learning system."""
    continuous_learning_system.stop()


def add_feedback(feedback: LearningFeedback):
    """
    Add feedback to the continuous learning system.
    
    Args:
        feedback: Feedback object
    """
    continuous_learning_system.add_feedback(feedback)


def add_task_data(task: Task):
    """
    Add task data to the continuous learning system.
    
    Args:
        task: Task object
    """
    continuous_learning_system.add_task_data(task)


def add_user_activity(activity: UserActivity):
    """
    Add user activity data to the continuous learning system.
    
    Args:
        activity: UserActivity object
    """
    continuous_learning_system.add_user_activity(activity)


if __name__ == "__main__":
    # Start continuous learning system
    start_continuous_learning()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        # Stop continuous learning system
        stop_continuous_learning()