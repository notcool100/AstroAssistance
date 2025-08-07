"""
Reinforcement learning model for continuous improvement.
"""
import os
from typing import Dict, List, Any, Tuple, Optional, Union
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from src.core.base_model import BaseModel
from src.core.config import config_manager
from src.core.logger import app_logger
from src.core.data_types import (
    Task, TaskPriority, TaskStatus, TaskCategory,
    Reminder, Goal, UserPreference, UserActivity,
    LearningFeedback, Recommendation
)


class ProductivityEnv(gym.Env):
    """
    Reinforcement learning environment for productivity optimization.
    
    This environment simulates a productivity assistant making recommendations
    to a user and receiving feedback.
    """
    
    def __init__(self, user_data: Dict[str, Any] = None):
        """
        Initialize the environment.
        
        Args:
            user_data: Dictionary containing user data (tasks, goals, preferences, etc.)
        """
        super(ProductivityEnv, self).__init__()
        
        # Store user data
        self.user_data = user_data or {}
        
        # Define action space
        # Actions represent different types of recommendations and their parameters
        # Action space: [recommendation_type, priority_level, time_allocation, ...]
        self.action_space = spaces.MultiDiscrete([
            6,  # Recommendation type (6 types)
            4,  # Priority level (4 levels)
            5,  # Time allocation (5 levels)
            3,  # Notification urgency (3 levels)
            2   # Include explanation (yes/no)
        ])
        
        # Define observation space
        # Observations represent the user's state
        # [task_count, overdue_tasks, completed_tasks, user_productivity_score, time_of_day, ...]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([100, 50, 1000, 1.0, 23, 6, 1.0, 1.0, 1.0, 10]),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation
        """
        # Initialize with random or user-specific state
        if self.user_data:
            # Use real user data to initialize state
            self.state = self._get_state_from_user_data()
        else:
            # Initialize with random state for training
            self.state = np.array([
                random.randint(5, 30),     # task_count
                random.randint(0, 10),     # overdue_tasks
                random.randint(10, 100),   # completed_tasks
                random.uniform(0.3, 0.8),  # user_productivity_score
                random.randint(8, 20),     # time_of_day
                random.randint(0, 6),      # day_of_week
                random.uniform(0.0, 1.0),  # energy_level
                random.uniform(0.0, 1.0),  # focus_level
                random.uniform(0.0, 1.0),  # stress_level
                random.randint(1, 5)       # consecutive_work_hours
            ], dtype=np.float32)
        
        # Reset episode variables
        self.steps = 0
        self.max_steps = 10
        self.recommendations_accepted = 0
        self.recommendations_rejected = 0
        
        # Return initial observation
        return self.state, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Unpack action
        rec_type, priority, time_allocation, notification_urgency, include_explanation = action
        
        # Map action to recommendation
        recommendation = self._action_to_recommendation(action)
        
        # Simulate user response to recommendation
        accepted, feedback = self._simulate_user_response(recommendation)
        
        # Update state based on user response
        self._update_state(accepted, recommendation)
        
        # Calculate reward
        reward = self._calculate_reward(accepted, recommendation)
        
        # Update counters
        self.steps += 1
        if accepted:
            self.recommendations_accepted += 1
        else:
            self.recommendations_rejected += 1
        
        # Check if episode is done
        done = self.steps >= self.max_steps
        
        # Additional info
        info = {
            "recommendation": recommendation,
            "accepted": accepted,
            "feedback": feedback,
            "recommendations_accepted": self.recommendations_accepted,
            "recommendations_rejected": self.recommendations_rejected
        }
        
        return self.state, reward, done, False, info
    
    def _get_state_from_user_data(self) -> np.ndarray:
        """
        Extract state from user data.
        
        Returns:
            State array
        """
        # Extract tasks
        tasks = self.user_data.get("tasks", [])
        
        # Count tasks
        task_count = len(tasks)
        
        # Count overdue tasks
        overdue_tasks = sum(1 for task in tasks if 
                           task.due_date and task.due_date < datetime.now() and 
                           task.status != TaskStatus.COMPLETED)
        
        # Count completed tasks
        completed_tasks = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        
        # Calculate productivity score (simplified)
        if task_count > 0:
            productivity_score = completed_tasks / (task_count + overdue_tasks)
        else:
            productivity_score = 0.5  # Default
        
        # Get current time
        now = datetime.now()
        time_of_day = now.hour
        day_of_week = now.weekday()
        
        # Simulate energy, focus, and stress levels
        # In a real system, these could come from user feedback or sensors
        energy_level = random.uniform(0.3, 0.9)
        focus_level = random.uniform(0.3, 0.9)
        stress_level = random.uniform(0.1, 0.7)
        
        # Simulate consecutive work hours
        consecutive_work_hours = random.randint(1, 6)
        
        return np.array([
            task_count,
            overdue_tasks,
            completed_tasks,
            productivity_score,
            time_of_day,
            day_of_week,
            energy_level,
            focus_level,
            stress_level,
            consecutive_work_hours
        ], dtype=np.float32)
    
    def _action_to_recommendation(self, action) -> Dict[str, Any]:
        """
        Convert action to recommendation.
        
        Args:
            action: Action array
            
        Returns:
            Recommendation dictionary
        """
        rec_type, priority, time_allocation, notification_urgency, include_explanation = action
        
        # Map recommendation type
        rec_type_mapping = {
            0: "task_scheduling",
            1: "task_prioritization",
            2: "goal_suggestion",
            3: "productivity_tip",
            4: "work_break_reminder",
            5: "task_grouping"
        }
        
        # Map priority
        priority_mapping = {
            0: TaskPriority.LOW,
            1: TaskPriority.MEDIUM,
            2: TaskPriority.HIGH,
            3: TaskPriority.CRITICAL
        }
        
        # Map time allocation (in minutes)
        time_allocation_mapping = {
            0: 15,
            1: 30,
            2: 60,
            3: 90,
            4: 120
        }
        
        # Map notification urgency
        notification_urgency_mapping = {
            0: "low",
            1: "medium",
            2: "high"
        }
        
        # Create recommendation
        recommendation = {
            "type": rec_type_mapping[rec_type],
            "priority": priority_mapping[priority],
            "time_allocation": time_allocation_mapping[time_allocation],
            "notification_urgency": notification_urgency_mapping[notification_urgency],
            "include_explanation": bool(include_explanation)
        }
        
        return recommendation
    
    def _simulate_user_response(self, recommendation) -> Tuple[bool, Dict[str, Any]]:
        """
        Simulate user response to recommendation.
        
        Args:
            recommendation: Recommendation dictionary
            
        Returns:
            Tuple of (accepted, feedback)
        """
        # Extract state variables
        task_count, overdue_tasks, completed_tasks, productivity_score, \
        time_of_day, day_of_week, energy_level, focus_level, stress_level, \
        consecutive_work_hours = self.state
        
        # Base acceptance probability
        base_prob = 0.7
        
        # Adjust based on recommendation type and user state
        if recommendation["type"] == "work_break_reminder":
            # More likely to accept breaks when stress is high or consecutive work hours are high
            prob_modifier = 0.1 * stress_level + 0.05 * consecutive_work_hours
        elif recommendation["type"] == "task_scheduling":
            # More likely to accept scheduling when task count is high
            prob_modifier = 0.01 * task_count - 0.1 * stress_level
        elif recommendation["type"] == "productivity_tip":
            # More likely to accept tips when productivity is low
            prob_modifier = 0.2 * (1 - productivity_score)
        else:
            prob_modifier = 0
        
        # Adjust based on recommendation parameters
        if recommendation["priority"] == TaskPriority.CRITICAL and stress_level > 0.7:
            # Less likely to accept critical tasks when stressed
            prob_modifier -= 0.2
        
        if recommendation["notification_urgency"] == "high" and focus_level > 0.8:
            # Less likely to accept urgent notifications when focused
            prob_modifier -= 0.15
        
        if recommendation["include_explanation"]:
            # More likely to accept with explanation
            prob_modifier += 0.1
        
        # Calculate final probability
        acceptance_prob = min(max(base_prob + prob_modifier, 0.1), 0.95)
        
        # Determine if accepted
        accepted = random.random() < acceptance_prob
        
        # Generate feedback
        if accepted:
            feedback = {
                "rating": random.uniform(3.5, 5.0),
                "comment": "Helpful recommendation" if random.random() < 0.3 else None
            }
        else:
            feedback = {
                "rating": random.uniform(1.0, 3.0),
                "comment": "Not relevant right now" if random.random() < 0.3 else None
            }
        
        return accepted, feedback
    
    def _update_state(self, accepted: bool, recommendation: Dict[str, Any]) -> None:
        """
        Update state based on user response.
        
        Args:
            accepted: Whether the recommendation was accepted
            recommendation: Recommendation dictionary
        """
        # Extract state variables
        task_count, overdue_tasks, completed_tasks, productivity_score, \
        time_of_day, day_of_week, energy_level, focus_level, stress_level, \
        consecutive_work_hours = self.state
        
        # Update state based on recommendation type and acceptance
        if accepted:
            if recommendation["type"] == "work_break_reminder":
                # Taking a break reduces stress and resets consecutive work hours
                stress_level = max(0.1, stress_level - 0.2)
                consecutive_work_hours = 1
                energy_level = min(1.0, energy_level + 0.1)
            
            elif recommendation["type"] == "task_scheduling":
                # Scheduling tasks increases productivity
                productivity_score = min(1.0, productivity_score + 0.05)
            
            elif recommendation["type"] == "task_prioritization":
                # Prioritizing tasks reduces overdue tasks
                overdue_tasks = max(0, overdue_tasks - 1)
            
            elif recommendation["type"] == "goal_suggestion":
                # New goals might add tasks
                task_count += 1
            
            # General effects of accepted recommendations
            focus_level = min(1.0, focus_level + 0.05)
        else:
            # Rejected recommendations might increase stress slightly
            stress_level = min(1.0, stress_level + 0.05)
        
        # Natural state evolution
        consecutive_work_hours = min(10, consecutive_work_hours + 0.5)
        energy_level = max(0.1, energy_level - 0.05)
        
        # Update state
        self.state = np.array([
            task_count,
            overdue_tasks,
            completed_tasks,
            productivity_score,
            time_of_day,
            day_of_week,
            energy_level,
            focus_level,
            stress_level,
            consecutive_work_hours
        ], dtype=np.float32)
    
    def _calculate_reward(self, accepted: bool, recommendation: Dict[str, Any]) -> float:
        """
        Calculate reward based on user response.
        
        Args:
            accepted: Whether the recommendation was accepted
            recommendation: Recommendation dictionary
            
        Returns:
            Reward value
        """
        # Base reward
        if accepted:
            base_reward = 1.0
        else:
            base_reward = -0.5
        
        # Extract state variables
        task_count, overdue_tasks, completed_tasks, productivity_score, \
        time_of_day, day_of_week, energy_level, focus_level, stress_level, \
        consecutive_work_hours = self.state
        
        # Additional reward based on state improvement
        additional_reward = 0.0
        
        if accepted:
            # Reward for reducing stress when it's high
            if recommendation["type"] == "work_break_reminder" and stress_level > 0.7:
                additional_reward += 0.5
            
            # Reward for scheduling tasks when there are many
            if recommendation["type"] == "task_scheduling" and task_count > 15:
                additional_reward += 0.3
            
            # Reward for prioritizing tasks when there are overdue ones
            if recommendation["type"] == "task_prioritization" and overdue_tasks > 5:
                additional_reward += 0.4
        
        # Penalty for inappropriate recommendations
        if recommendation["type"] == "work_break_reminder" and consecutive_work_hours < 2:
            # Penalize suggesting breaks too soon
            additional_reward -= 0.3
        
        if recommendation["notification_urgency"] == "high" and focus_level > 0.8:
            # Penalize high urgency notifications during focus
            additional_reward -= 0.2
        
        # Calculate final reward
        reward = base_reward + additional_reward
        
        return reward


class ReinforcementLearningModel(BaseModel):
    """Reinforcement learning model for continuous improvement."""
    
    def __init__(self):
        """Initialize the reinforcement learning model."""
        super().__init__("reinforcement_learning_model", "rl_agent")
        self.env = None
        self.model = None
    
    def build(self, **kwargs) -> None:
        """Build the model architecture."""
        # Get model configuration
        algorithm = kwargs.get("algorithm", config_manager.get("models.reinforcement_learning.algorithm", "ppo"))
        gamma = kwargs.get("gamma", config_manager.get("models.reinforcement_learning.gamma", 0.99))
        learning_rate = kwargs.get("learning_rate", config_manager.get("models.reinforcement_learning.learning_rate", 0.0003))
        
        # Create environment
        self.env = DummyVecEnv([lambda: ProductivityEnv()])
        
        # Create model
        if algorithm.lower() == "ppo":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                gamma=gamma,
                learning_rate=learning_rate,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        app_logger.info(f"Built reinforcement learning model with algorithm={algorithm}")
    
    def train(self, train_data=None, validation_data=None, **kwargs) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Not used for RL model
            validation_data: Not used for RL model
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        # Get training parameters
        total_timesteps = kwargs.get("total_timesteps", 100000)
        eval_freq = kwargs.get("eval_freq", 10000)
        
        # Training history
        history = {
            "mean_reward": [],
            "std_reward": [],
            "timesteps": []
        }
        
        # Define callback for evaluation
        def eval_callback(locals_, globals_):
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.env,
                n_eval_episodes=10,
                deterministic=True
            )
            
            # Update history
            history["mean_reward"].append(mean_reward)
            history["std_reward"].append(std_reward)
            history["timesteps"].append(locals_["self"].num_timesteps)
            
            # Log progress
            app_logger.info(
                f"Timestep: {locals_['self'].num_timesteps}, "
                f"Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}"
            )
            
            return True
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback if eval_freq > 0 else None,
            log_interval=1000
        )
        
        # Update metadata
        self.metadata["training_history"] = history
        self.is_trained = True
        
        return history
    
    def predict(self, data, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            data: Input data (observation)
            **kwargs: Additional prediction parameters
            
        Returns:
            Numpy array of actions
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Predictions may be unreliable.")
        
        # Convert data to appropriate format
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to numpy array
            data = data.values.astype(np.float32)
        elif not isinstance(data, np.ndarray):
            raise ValueError("Unsupported data type. Expected DataFrame or numpy array.")
        
        # Ensure data has the right shape
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Make prediction (get action)
        action, _ = self.model.predict(data, deterministic=kwargs.get("deterministic", True))
        
        return action
    
    def evaluate(self, test_data=None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_data: Not used for RL model
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Evaluation may be unreliable.")
        
        # Get evaluation parameters
        n_eval_episodes = kwargs.get("n_eval_episodes", 10)
        
        # Evaluate policy
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        # Calculate metrics
        metrics = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward)
        }
        
        # Log evaluation results
        app_logger.info(f"Evaluation results: {metrics}")
        
        # Update metadata
        self.metadata["performance_metrics"] = metrics
        
        return metrics
    
    def get_recommendation(self, user_state: np.ndarray) -> Dict[str, Any]:
        """
        Get a recommendation based on user state.
        
        Args:
            user_state: User state array
            
        Returns:
            Recommendation dictionary
        """
        # Check if model is trained
        if not self.is_trained:
            app_logger.warning("Model is not trained. Recommendations may be unreliable.")
        
        # Get action from model
        action, _ = self.model.predict(user_state.reshape(1, -1), deterministic=True)
        
        # Convert action to recommendation
        env = ProductivityEnv()  # Create a temporary environment
        recommendation = env._action_to_recommendation(action[0])
        
        return recommendation