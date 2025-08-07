#!/usr/bin/env python3
"""
Task Prediction Script
---------------------
This script predicts whether a task will be completed on time
and how long it will actually take.

Usage:
    python predict_task.py --task '{"id": "123", "title": "Example Task", ...}'
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prediction.log')
    ]
)
logger = logging.getLogger('AstroAssistance-Prediction')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def load_models():
    """Load the trained models."""
    completion_model_path = MODELS_DIR / "task_completion_predictor.joblib"
    duration_model_path = MODELS_DIR / "task_duration_predictor.joblib"
    
    completion_model = None
    duration_model = None
    
    if os.path.exists(completion_model_path):
        try:
            completion_model = joblib.load(completion_model_path)
            logger.info("Loaded task completion prediction model")
        except Exception as e:
            logger.error(f"Failed to load completion model: {e}")
    
    if os.path.exists(duration_model_path):
        try:
            duration_model = joblib.load(duration_model_path)
            logger.info("Loaded task duration prediction model")
        except Exception as e:
            logger.error(f"Failed to load duration model: {e}")
    
    return completion_model, duration_model

def preprocess_task(task_data):
    """Preprocess task data for prediction."""
    # Convert to DataFrame
    task_df = pd.DataFrame([task_data])
    
    # Process dates
    if 'dueDate' in task_data and task_data['dueDate']:
        task_df['dueDate'] = pd.to_datetime(task_data['dueDate'])
    else:
        task_df['dueDate'] = pd.NaT
    
    if 'createdAt' in task_data and task_data['createdAt']:
        task_df['createdAt'] = pd.to_datetime(task_data['createdAt'])
    else:
        task_df['createdAt'] = pd.to_datetime('now')
    
    # Calculate time to deadline in hours
    if not pd.isna(task_df['dueDate'].iloc[0]):
        task_df['time_to_deadline'] = (task_df['dueDate'] - task_df['createdAt']).dt.total_seconds() / 3600
    else:
        task_df['time_to_deadline'] = 168  # Default to 1 week (168 hours)
    
    # Map priority to standard format
    priority_map = {
        'low': 'LOW',
        'medium': 'MEDIUM',
        'high': 'HIGH',
        'urgent': 'URGENT'
    }
    
    if 'priority' in task_data:
        task_df['priority'] = task_data['priority'].upper()
    else:
        task_df['priority'] = 'MEDIUM'
    
    # Map category to standard format
    if 'category' in task_data:
        task_df['category'] = task_data['category'].upper()
    else:
        task_df['category'] = 'OTHER'
    
    # Handle estimated duration
    if 'estimatedDuration' in task_data and task_data['estimatedDuration']:
        task_df['estimated_duration'] = task_data['estimatedDuration']
    else:
        task_df['estimated_duration'] = 60  # Default to 1 hour
    
    # Extract tag count for duration prediction
    if 'tags' in task_data and isinstance(task_data['tags'], list):
        task_df['tag_count'] = len(task_data['tags'])
    else:
        task_df['tag_count'] = 0
    
    return task_df

def predict_task_completion(model, task_df):
    """Predict whether a task will be completed on time."""
    if model is None:
        # Fallback prediction if model is not available
        logger.warning("Completion model not available, using fallback prediction")
        
        # Simple heuristic: high priority tasks with close deadlines are less likely to be completed on time
        if task_df['priority'].iloc[0] in ['HIGH', 'URGENT'] and task_df['time_to_deadline'].iloc[0] < 48:
            will_complete = False
            confidence = 0.7
        else:
            will_complete = True
            confidence = 0.6
        
        return {
            'will_complete_on_time': will_complete,
            'confidence': confidence,
            'model_used': 'fallback'
        }
    
    try:
        # Select features for prediction
        features = ['priority', 'category', 'time_to_deadline', 'estimated_duration']
        X = task_df[features]
        
        # Make prediction
        prediction = bool(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])  # Probability of class 1 (completed on time)
        
        return {
            'will_complete_on_time': prediction,
            'confidence': probability,
            'model_used': 'ml_model'
        }
    except Exception as e:
        logger.error(f"Error predicting task completion: {e}")
        return {
            'will_complete_on_time': True,
            'confidence': 0.5,
            'model_used': 'error_fallback',
            'error': str(e)
        }

def predict_task_duration(model, task_df):
    """Predict how long a task will take to complete."""
    if model is None:
        # Fallback prediction if model is not available
        logger.warning("Duration model not available, using fallback prediction")
        
        # Simple heuristic: tasks typically take 20% longer than estimated
        estimated_duration = task_df['estimated_duration'].iloc[0]
        predicted_duration = estimated_duration * 1.2
        
        return {
            'predicted_duration': float(predicted_duration),
            'estimated_duration': float(estimated_duration),
            'model_used': 'fallback'
        }
    
    try:
        # Select features for prediction
        features = ['priority', 'category', 'estimated_duration', 'tag_count']
        X = task_df[features]
        
        # Make prediction
        predicted_duration = float(model.predict(X)[0])
        
        return {
            'predicted_duration': predicted_duration,
            'estimated_duration': float(task_df['estimated_duration'].iloc[0]),
            'model_used': 'ml_model'
        }
    except Exception as e:
        logger.error(f"Error predicting task duration: {e}")
        estimated_duration = task_df['estimated_duration'].iloc[0]
        return {
            'predicted_duration': float(estimated_duration * 1.2),
            'estimated_duration': float(estimated_duration),
            'model_used': 'error_fallback',
            'error': str(e)
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Task Prediction Script")
    parser.add_argument('--task', required=True, help='Task data in JSON format')
    args = parser.parse_args()
    
    try:
        # Parse task data
        task_data = json.loads(args.task)
        
        # Load models
        completion_model, duration_model = load_models()
        
        # Preprocess task
        task_df = preprocess_task(task_data)
        
        # Make predictions
        completion_prediction = predict_task_completion(completion_model, task_df)
        duration_prediction = predict_task_duration(duration_model, task_df)
        
        # Combine predictions
        result = {
            'task_id': task_data.get('id', 'unknown'),
            'completion_prediction': completion_prediction,
            'duration_prediction': duration_prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        error_result = {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()