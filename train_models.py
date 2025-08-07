#!/usr/bin/env python3
"""
AstroAssistance AI Training Pipeline
-----------------------------------
This script implements the training pipeline for AstroAssistance's AI models.
It can be run manually or scheduled to run periodically.

Author: Senior AI Engineer
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib

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
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, DATA_DIR / "processed", MODELS_DIR, LOGS_DIR]:
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
        tasks_df['due_date'] = pd.to_datetime(tasks_df['due_date'])
        tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
        tasks_df['time_to_deadline'] = (tasks_df['due_date'] - tasks_df['created_at']).dt.total_seconds() / 3600  # hours
        
        # Create target variable: was the task completed on time?
        tasks_df['completed_on_time'] = False
        completed_mask = tasks_df['status'] == 'COMPLETED'
        
        if 'completed_at' in tasks_df.columns:
            tasks_df.loc[completed_mask, 'completed_on_time'] = (
                pd.to_datetime(tasks_df.loc[completed_mask, 'completed_at']) <= 
                tasks_df.loc[completed_mask, 'due_date']
            )
        
        # Select features
        features = [
            'priority', 'category', 'time_to_deadline', 'estimated_duration'
        ]
        
        # Handle missing values
        tasks_df['estimated_duration'] = tasks_df['estimated_duration'].fillna(60)  # Default to 1 hour
        
        # Filter rows with all required features
        valid_rows = tasks_df[features].notna().all(axis=1)
        X = tasks_df.loc[valid_rows, features]
        y = tasks_df.loc[valid_rows, 'completed_on_time']
        
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
    
    def predict(self, task_data):
        """Predict whether a task will be completed on time."""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                logger.error("Model not trained yet")
                return None
        
        # Preprocess single task
        task_df = pd.DataFrame([task_data])
        task_df['due_date'] = pd.to_datetime(task_df['due_date'])
        task_df['created_at'] = pd.to_datetime(task_df['created_at'])
        task_df['time_to_deadline'] = (task_df['due_date'] - task_df['created_at']).dt.total_seconds() / 3600
        
        # Select features
        features = [
            'priority', 'category', 'time_to_deadline', 'estimated_duration'
        ]
        
        # Handle missing values
        task_df['estimated_duration'] = task_df['estimated_duration'].fillna(60)
        
        # Make prediction
        X = task_df[features]
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]  # Probability of class 1 (completed on time)
        
        return {
            'will_complete_on_time': bool(prediction),
            'confidence': float(probability)
        }

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
        
        # We need tasks that have been completed and have actual_duration
        completed_tasks = tasks_df[
            (tasks_df['status'] == 'COMPLETED') & 
            (tasks_df['actual_duration'].notna())
        ]
        
        if len(completed_tasks) < 10:
            logger.warning("Not enough completed tasks with duration data")
            return None, None
        
        # Feature engineering
        # Extract features from tags
        completed_tasks['tag_count'] = completed_tasks['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Select features
        features = [
            'priority', 'category', 'estimated_duration', 'tag_count'
        ]
        
        # Handle missing values
        completed_tasks['estimated_duration'] = completed_tasks['estimated_duration'].fillna(60)
        
        # Filter rows with all required features
        valid_rows = completed_tasks[features].notna().all(axis=1)
        X = completed_tasks.loc[valid_rows, features]
        y = completed_tasks.loc[valid_rows, 'actual_duration']
        
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
    
    def predict(self, task_data):
        """Predict how long a task will take to complete."""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                logger.error("Model not trained yet")
                return None
        
        # Preprocess single task
        task_df = pd.DataFrame([task_data])
        
        # Extract features from tags
        task_df['tag_count'] = task_df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Select features
        features = [
            'priority', 'category', 'estimated_duration', 'tag_count'
        ]
        
        # Handle missing values
        task_df['estimated_duration'] = task_df['estimated_duration'].fillna(60)
        
        # Make prediction
        X = task_df[features]
        predicted_duration = self.model.predict(X)[0]
        
        return {
            'predicted_duration': float(predicted_duration),
            'estimated_duration': float(task_data.get('estimated_duration', 0))
        }

class RecommendationEngine:
    """Engine to generate personalized recommendations."""
    
    def __init__(self):
        self.task_completion_predictor = TaskCompletionPredictor()
        self.task_duration_predictor = TaskDurationPredictor()
        
    def load_models(self):
        """Load trained models."""
        # Check if models exist and load them
        completion_model_path = MODELS_DIR / "task_completion_predictor.joblib"
        duration_model_path = MODELS_DIR / "task_duration_predictor.joblib"
        
        if os.path.exists(completion_model_path):
            self.task_completion_predictor.model = joblib.load(completion_model_path)
            logger.info("Loaded task completion prediction model")
        
        if os.path.exists(duration_model_path):
            self.task_duration_predictor.model = joblib.load(duration_model_path)
            logger.info("Loaded task duration prediction model")
    
    def generate_recommendations(self, tasks_data, user_preferences=None):
        """Generate personalized recommendations based on tasks and user preferences."""
        logger.info("Generating recommendations")
        
        # Load models if not already loaded
        if self.task_completion_predictor.model is None or self.task_duration_predictor.model is None:
            self.load_models()
            
            # If models still not loaded, they don't exist yet
            if self.task_completion_predictor.model is None:
                logger.warning("Task completion prediction model not available")
            if self.task_duration_predictor.model is None:
                logger.warning("Task duration prediction model not available")
        
        recommendations = []
        
        # Convert to DataFrame if it's a list
        if isinstance(tasks_data, list):
            tasks_df = pd.DataFrame(tasks_data)
        else:
            tasks_df = tasks_data
        
        # Filter for active tasks
        active_tasks = tasks_df[tasks_df['status'] != 'COMPLETED']
        
        if len(active_tasks) == 0:
            logger.info("No active tasks to generate recommendations for")
            return recommendations
        
        # 1. Identify tasks at risk of not being completed on time
        if self.task_completion_predictor.model is not None:
            for _, task in active_tasks.iterrows():
                task_dict = task.to_dict()
                prediction = self.task_completion_predictor.predict(task_dict)
                
                if prediction and not prediction['will_complete_on_time'] and prediction['confidence'] > 0.7:
                    recommendations.append({
                        'id': f"rec_completion_{task['id']}",
                        'title': f"Task at risk: {task['title']}",
                        'description': f"This task is at risk of not being completed on time. Consider prioritizing it or adjusting the deadline.",
                        'type': "task_completion_risk",
                        'priority': 3,
                        'created_at': datetime.now().isoformat(),
                        'expires_at': None,
                        'user_id': task['user_id'],
                        'metadata': {
                            'task_id': task['id'],
                            'confidence': prediction['confidence']
                        }
                    })
        
        # 2. Identify tasks that might take longer than estimated
        if self.task_duration_predictor.model is not None:
            for _, task in active_tasks.iterrows():
                task_dict = task.to_dict()
                prediction = self.task_duration_predictor.predict(task_dict)
                
                if prediction and prediction['predicted_duration'] > prediction['estimated_duration'] * 1.5:
                    recommendations.append({
                        'id': f"rec_duration_{task['id']}",
                        'title': f"Duration warning: {task['title']}",
                        'description': f"This task might take longer than estimated. Consider allocating more time.",
                        'type': "task_duration_warning",
                        'priority': 2,
                        'created_at': datetime.now().isoformat(),
                        'expires_at': None,
                        'user_id': task['user_id'],
                        'metadata': {
                            'task_id': task['id'],
                            'estimated_duration': prediction['estimated_duration'],
                            'predicted_duration': prediction['predicted_duration']
                        }
                    })
        
        # 3. Prioritize tasks based on due date and importance
        high_priority_tasks = active_tasks[active_tasks['priority'].isin(['HIGH', 'URGENT'])]
        if len(high_priority_tasks) > 0:
            # Sort by due date
            high_priority_tasks['due_date'] = pd.to_datetime(high_priority_tasks['due_date'])
            high_priority_tasks = high_priority_tasks.sort_values('due_date')
            
            if len(high_priority_tasks) > 0:
                next_task = high_priority_tasks.iloc[0]
                recommendations.append({
                    'id': f"rec_priority_{next_task['id']}",
                    'title': f"Prioritize: {next_task['title']}",
                    'description': f"This high-priority task has the nearest deadline. Consider working on it next.",
                    'type': "task_prioritization",
                    'priority': 3,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': None,
                    'user_id': next_task['user_id'],
                    'metadata': {
                        'task_id': next_task['id']
                    }
                })
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

def load_sample_data():
    """Load sample data for training."""
    logger.info("Loading sample data")
    
    # Check if sample data exists
    sample_tasks_path = DATA_DIR / "synthetic" / "tasks.json"
    
    if not os.path.exists(sample_tasks_path):
        logger.error(f"Sample data not found at {sample_tasks_path}")
        return None
    
    # Load sample tasks
    with open(sample_tasks_path, 'r') as f:
        tasks = json.load(f)
    
    # Enhance sample data for training
    enhanced_tasks = []
    
    for task in tasks:
        # Convert string dates to datetime objects
        for date_field in ['due_date', 'created_at', 'updated_at']:
            if task.get(date_field):
                task[date_field] = task[date_field]
        
        # Add actual_duration for completed tasks
        if task.get('status') == 'COMPLETED':
            # Actual duration is estimated +/- 30%
            variation = np.random.uniform(0.7, 1.3)
            task['actual_duration'] = int(task.get('estimated_duration', 60) * variation)
            task['completed_at'] = (
                pd.to_datetime(task['created_at']) + 
                pd.Timedelta(minutes=task['actual_duration'])
            ).isoformat()
        
        enhanced_tasks.append(task)
    
    # Generate more synthetic data for better training
    num_additional_tasks = 100
    logger.info(f"Generating {num_additional_tasks} additional synthetic tasks for training")
    
    categories = ['WORK', 'PERSONAL', 'HEALTH', 'EDUCATION', 'FINANCE', 'SOCIAL', 'OTHER']
    priorities = ['LOW', 'MEDIUM', 'HIGH', 'URGENT']
    statuses = ['NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'CANCELLED']
    
    for i in range(num_additional_tasks):
        # Generate random dates
        created_at = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 60))
        due_date = created_at + pd.Timedelta(days=np.random.randint(1, 30))
        
        # Randomly decide if task is completed
        is_completed = np.random.random() < 0.6
        status = 'COMPLETED' if is_completed else np.random.choice([s for s in statuses if s != 'COMPLETED'])
        
        # Generate estimated and actual durations
        estimated_duration = np.random.randint(15, 240)  # 15 min to 4 hours
        
        # For completed tasks, generate actual duration and completion date
        completed_at = None
        actual_duration = None
        
        if is_completed:
            # Determine if task was completed on time (70% chance)
            completed_on_time = np.random.random() < 0.7
            
            if completed_on_time:
                completed_at = due_date - pd.Timedelta(hours=np.random.randint(1, 48))
            else:
                completed_at = due_date + pd.Timedelta(hours=np.random.randint(1, 48))
            
            # Actual duration is estimated +/- 50%
            variation = np.random.uniform(0.5, 1.5)
            actual_duration = int(estimated_duration * variation)
        
        # Create synthetic task
        synthetic_task = {
            'id': f"syn_task_{i}",
            'title': f"Synthetic Task {i}",
            'description': f"This is a synthetic task for training purposes {i}",
            'category': np.random.choice(categories),
            'priority': np.random.choice(priorities),
            'status': status,
            'due_date': due_date.isoformat(),
            'created_at': created_at.isoformat(),
            'updated_at': (created_at + pd.Timedelta(days=np.random.randint(1, 10))).isoformat(),
            'completed_at': completed_at.isoformat() if completed_at else None,
            'estimated_duration': estimated_duration,
            'actual_duration': actual_duration,
            'tags': np.random.choice(['important', 'urgent', 'meeting', 'email', 'report', 'call'], 
                                    size=np.random.randint(0, 4), 
                                    replace=False).tolist(),
            'user_id': "user123"
        }
        
        enhanced_tasks.append(synthetic_task)
    
    logger.info(f"Loaded and enhanced {len(enhanced_tasks)} tasks for training")
    return enhanced_tasks

def save_recommendations(recommendations):
    """Save generated recommendations to a file."""
    recommendations_path = DATA_DIR / "synthetic" / "recommendations.json"
    
    with open(recommendations_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"Saved {len(recommendations)} recommendations to {recommendations_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="AstroAssistance AI Training Pipeline")
    parser.add_argument("--generate-recommendations", action="store_true", help="Generate recommendations after training")
    args = parser.parse_args()
    
    logger.info("Starting AstroAssistance AI training pipeline")
    
    # Load sample data
    tasks_data = load_sample_data()
    
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
    
    # Generate recommendations if requested
    if args.generate_recommendations:
        logger.info("Generating recommendations based on trained models")
        recommendation_engine = RecommendationEngine()
        recommendations = recommendation_engine.generate_recommendations(tasks_data)
        
        # Save recommendations
        save_recommendations(recommendations)
    
    logger.info("AI training pipeline completed")

if __name__ == "__main__":
    main()