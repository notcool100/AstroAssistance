#!/usr/bin/env python3
"""
AI Model Training Script
-----------------------
This script trains the AI models using data from the database.

Usage:
    python train_models.py
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger('AstroAssistance-Training')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)

class TaskCompletionPredictor:
    """Model to predict whether a task will be completed on time."""
    
    def __init__(self):
        self.model = None
        self.model_path = MODELS_DIR / "task_completion_predictor.joblib"
        
    def preprocess_data(self, tasks_data):
        """Preprocess task data for training."""
        logger.info("Preprocessing task data for completion prediction")
        
        # Convert to DataFrame if it's a list
        if isinstance(tasks_data, list):
            tasks_df = pd.DataFrame(tasks_data)
        else:
            tasks_df = tasks_data
            
        # Feature engineering
        tasks_df['dueDate'] = pd.to_datetime(tasks_df['dueDate'])
        tasks_df['createdAt'] = pd.to_datetime(tasks_df['createdAt'])
        tasks_df['time_to_deadline'] = (tasks_df['dueDate'] - tasks_df['createdAt']).dt.total_seconds() / 3600  # hours
        
        # Create target variable: was the task completed on time?
        tasks_df['completed_on_time'] = False
        completed_mask = tasks_df['completed'] == True
        
        if 'completedAt' in tasks_df.columns:
            tasks_df.loc[completed_mask, 'completed_on_time'] = (
                pd.to_datetime(tasks_df.loc[completed_mask, 'completedAt']) <= 
                tasks_df.loc[completed_mask, 'dueDate']
            )
        
        # Select features
        features = [
            'priority', 'category', 'time_to_deadline', 'estimatedDuration'
        ]
        
        # Handle missing values
        tasks_df['estimatedDuration'] = tasks_df['estimatedDuration'].fillna(60)  # Default to 1 hour
        
        # Filter rows with all required features and a due date
        valid_rows = tasks_df[features].notna().all(axis=1) & tasks_df['dueDate'].notna()
        X = tasks_df.loc[valid_rows, features]
        y = tasks_df.loc[valid_rows, 'completed_on_time']
        
        # Rename columns to match prediction script
        X = X.rename(columns={'estimatedDuration': 'estimated_duration'})
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        
        return X, y
    
    def build_pipeline(self):
        """Build the ML pipeline."""
        # Define preprocessing for categorical features
        categorical_features = ['priority', 'category']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Define preprocessing for numerical features
        numerical_features = ['time_to_deadline', 'estimated_duration']
        numerical_transformer = StandardScaler()
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        return pipeline
    
    def train(self, tasks_data):
        """Train the task completion prediction model."""
        logger.info("Training task completion prediction model")
        
        # Preprocess data
        X, y = self.preprocess_data(tasks_data)
        
        if len(X) < 10:
            logger.warning("Not enough data for training. Need at least 10 samples.")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train pipeline
        self.model = self.build_pipeline()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Task completion prediction accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return True

class TaskDurationPredictor:
    """Model to predict how long a task will take to complete."""
    
    def __init__(self):
        self.model = None
        self.model_path = MODELS_DIR / "task_duration_predictor.joblib"
        
    def preprocess_data(self, tasks_data):
        """Preprocess task data for training."""
        logger.info("Preprocessing task data for duration prediction")
        
        # Convert to DataFrame if it's a list
        if isinstance(tasks_data, list):
            tasks_df = pd.DataFrame(tasks_data)
        else:
            tasks_df = tasks_data
        
        # We need tasks that have been completed
        completed_tasks = tasks_df[tasks_df['completed'] == True]
        
        if len(completed_tasks) < 10:
            logger.warning("Not enough completed tasks for training")
            return None, None
        
        # Feature engineering
        # Extract features from tags
        completed_tasks['tag_count'] = completed_tasks['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Select features
        features = [
            'priority', 'category', 'estimatedDuration', 'tag_count'
        ]
        
        # Handle missing values
        completed_tasks['estimatedDuration'] = completed_tasks['estimatedDuration'].fillna(60)
        
        # Calculate actual duration if possible
        if 'completedAt' in completed_tasks.columns and 'createdAt' in completed_tasks.columns:
            completed_tasks['actual_duration'] = (
                pd.to_datetime(completed_tasks['completedAt']) - 
                pd.to_datetime(completed_tasks['createdAt'])
            ).dt.total_seconds() / 60  # minutes
        else:
            # If we don't have completion time, we can't train this model
            logger.warning("Completion time data not available")
            return None, None
        
        # Filter out unreasonable durations (e.g., tasks that took months)
        completed_tasks = completed_tasks[completed_tasks['actual_duration'] <= 60 * 24 * 7]  # Max 1 week
        
        # Filter rows with all required features
        valid_rows = completed_tasks[features].notna().all(axis=1) & completed_tasks['actual_duration'].notna()
        X = completed_tasks.loc[valid_rows, features]
        y = completed_tasks.loc[valid_rows, 'actual_duration']
        
        # Rename columns to match prediction script
        X = X.rename(columns={'estimatedDuration': 'estimated_duration'})
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        
        return X, y
    
    def build_pipeline(self):
        """Build the ML pipeline."""
        # Define preprocessing for categorical features
        categorical_features = ['priority', 'category']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Define preprocessing for numerical features
        numerical_features = ['estimated_duration', 'tag_count']
        numerical_transformer = StandardScaler()
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        return pipeline
    
    def train(self, tasks_data):
        """Train the task duration prediction model."""
        logger.info("Training task duration prediction model")
        
        # Preprocess data
        X, y = self.preprocess_data(tasks_data)
        
        if X is None or len(X) < 10:
            logger.warning("Not enough data for training. Need at least 10 samples.")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train pipeline
        self.model = self.build_pipeline()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        logger.info(f"Task duration prediction RMSE: {rmse:.4f} minutes")
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return True

def load_data_from_file():
    """Load data from file for training (when database is not available)."""
    logger.info("Loading data from file")
    
    # Check if sample data exists
    sample_tasks_path = DATA_DIR / "tasks_export.json"
    
    if not os.path.exists(sample_tasks_path):
        logger.warning(f"Sample data not found at {sample_tasks_path}")
        # Generate synthetic data for training
        return generate_synthetic_data()
    
    # Load sample tasks
    with open(sample_tasks_path, 'r') as f:
        tasks = json.load(f)
    
    logger.info(f"Loaded {len(tasks)} tasks from file")
    return tasks

def generate_synthetic_data():
    """Generate synthetic data for training when real data is not available."""
    logger.info("Generating synthetic data for training")
    
    num_tasks = 200
    synthetic_tasks = []
    
    categories = ['WORK', 'PERSONAL', 'HEALTH', 'EDUCATION', 'FINANCE', 'SOCIAL', 'OTHER']
    priorities = ['LOW', 'MEDIUM', 'HIGH', 'URGENT']
    
    for i in range(num_tasks):
        # Generate random dates
        created_at = datetime.now() - pd.Timedelta(days=np.random.randint(1, 60))
        due_date = created_at + pd.Timedelta(days=np.random.randint(1, 30))
        
        # Randomly decide if task is completed
        is_completed = np.random.random() < 0.6
        
        # Generate estimated duration
        estimated_duration = np.random.randint(15, 240)  # 15 min to 4 hours
        
        # For completed tasks, generate completion date
        completed_at = None
        if is_completed:
            # Determine if task was completed on time (70% chance)
            completed_on_time = np.random.random() < 0.7
            
            if completed_on_time:
                completed_at = due_date - pd.Timedelta(hours=np.random.randint(1, 48))
            else:
                completed_at = due_date + pd.Timedelta(hours=np.random.randint(1, 48))
        
        # Create synthetic task
        synthetic_task = {
            'id': f"syn_task_{i}",
            'title': f"Synthetic Task {i}",
            'description': f"This is a synthetic task for training purposes {i}",
            'category': np.random.choice(categories),
            'priority': np.random.choice(priorities),
            'completed': is_completed,
            'dueDate': due_date.isoformat(),
            'createdAt': created_at.isoformat(),
            'updatedAt': (created_at + pd.Timedelta(days=np.random.randint(1, 10))).isoformat(),
            'completedAt': completed_at.isoformat() if completed_at else None,
            'estimatedDuration': estimated_duration,
            'tags': np.random.choice(['important', 'urgent', 'meeting', 'email', 'report', 'call'], 
                                    size=np.random.randint(0, 4), 
                                    replace=False).tolist(),
            'userId': "synthetic_user"
        }
        
        synthetic_tasks.append(synthetic_task)
    
    logger.info(f"Generated {len(synthetic_tasks)} synthetic tasks for training")
    
    # Save synthetic data for future use
    with open(DATA_DIR / "synthetic_tasks.json", 'w') as f:
        json.dump(synthetic_tasks, f, indent=2)
    
    return synthetic_tasks

def main():
    """Main training function."""
    logger.info("Starting AstroAssistance AI training pipeline")
    
    # Load data
    tasks_data = load_data_from_file()
    
    if not tasks_data:
        logger.error("Failed to load training data")
        return
    
    # Train task completion prediction model
    completion_predictor = TaskCompletionPredictor()
    completion_success = completion_predictor.train(tasks_data)
    
    # Train task duration prediction model
    duration_predictor = TaskDurationPredictor()
    duration_success = duration_predictor.train(tasks_data)
    
    if not completion_success and not duration_success:
        logger.error("All model training failed")
        return
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()