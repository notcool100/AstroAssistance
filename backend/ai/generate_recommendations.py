#!/usr/bin/env python3
"""
Recommendation Generation Script
-------------------------------
This script generates personalized recommendations based on user data.

Usage:
    python generate_recommendations.py --user "user_id" --count 3 --tasks "[...]" --goals "[...]" --preferences "{...}"
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recommendations.log')
    ]
)
logger = logging.getLogger('AstroAssistance-Recommendations')

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

def preprocess_task(task):
    """Preprocess a task for prediction."""
    # Convert to DataFrame
    task_df = pd.DataFrame([task])
    
    # Process dates
    if 'dueDate' in task and task['dueDate']:
        task_df['dueDate'] = pd.to_datetime(task['dueDate'])
    else:
        task_df['dueDate'] = pd.NaT
    
    if 'createdAt' in task and task['createdAt']:
        task_df['createdAt'] = pd.to_datetime(task['createdAt'])
    else:
        task_df['createdAt'] = pd.to_datetime('now')
    
    # Calculate time to deadline in hours
    if not pd.isna(task_df['dueDate'].iloc[0]):
        task_df['time_to_deadline'] = (task_df['dueDate'] - task_df['createdAt']).dt.total_seconds() / 3600
    else:
        task_df['time_to_deadline'] = 168  # Default to 1 week (168 hours)
    
    # Standardize priority
    if 'priority' in task:
        task_df['priority'] = task['priority'].upper()
    else:
        task_df['priority'] = 'MEDIUM'
    
    # Standardize category
    if 'category' in task:
        task_df['category'] = task['category'].upper()
    else:
        task_df['category'] = 'OTHER'
    
    # Handle estimated duration
    if 'estimatedDuration' in task and task['estimatedDuration']:
        task_df['estimated_duration'] = task['estimatedDuration']
    else:
        task_df['estimated_duration'] = 60  # Default to 1 hour
    
    # Extract tag count for duration prediction
    if 'tags' in task and isinstance(task['tags'], list):
        task_df['tag_count'] = len(task['tags'])
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
        
        return will_complete, confidence
    
    try:
        # Select features for prediction
        features = ['priority', 'category', 'time_to_deadline', 'estimated_duration']
        X = task_df[features]
        
        # Make prediction
        prediction = bool(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])  # Probability of class 1 (completed on time)
        
        return prediction, probability
    except Exception as e:
        logger.error(f"Error predicting task completion: {e}")
        return True, 0.5

def predict_task_duration(model, task_df):
    """Predict how long a task will take to complete."""
    if model is None:
        # Fallback prediction if model is not available
        logger.warning("Duration model not available, using fallback prediction")
        
        # Simple heuristic: tasks typically take 20% longer than estimated
        estimated_duration = task_df['estimated_duration'].iloc[0]
        predicted_duration = estimated_duration * 1.2
        
        return predicted_duration
    
    try:
        # Select features for prediction
        features = ['priority', 'category', 'estimated_duration', 'tag_count']
        X = task_df[features]
        
        # Make prediction
        predicted_duration = float(model.predict(X)[0])
        
        return predicted_duration
    except Exception as e:
        logger.error(f"Error predicting task duration: {e}")
        estimated_duration = task_df['estimated_duration'].iloc[0]
        return estimated_duration * 1.2

def generate_recommendations(user_id, tasks, goals, preferences, count=3):
    """Generate personalized recommendations based on user data."""
    logger.info(f"Generating recommendations for user: {user_id}")
    
    # Load models
    completion_model, duration_model = load_models()
    
    recommendations = []
    
    # 1. Task completion recommendations
    for task in tasks:
        # Skip completed tasks
        if task.get('completed', False):
            continue
        
        # Preprocess task
        task_df = preprocess_task(task)
        
        # Predict task completion
        will_complete, confidence = predict_task_completion(completion_model, task_df)
        
        # If task is predicted to not be completed on time with high confidence
        if not will_complete and confidence > 0.65:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "type": "task_completion",
                "content": f"Your task '{task['title']}' might not be completed on time. Consider prioritizing it.",
                "reason": "Based on your past task completion patterns, this task has a high risk of missing its deadline.",
                "metadata": {
                    "task_id": task['id'],
                    "confidence": confidence,
                    "ai_generated": True
                }
            })
    
    # 2. Task duration recommendations
    for task in tasks:
        # Skip completed tasks
        if task.get('completed', False):
            continue
        
        # Preprocess task
        task_df = preprocess_task(task)
        
        # Predict task duration
        predicted_duration = predict_task_duration(duration_model, task_df)
        estimated_duration = task_df['estimated_duration'].iloc[0]
        
        # If predicted duration is significantly longer than estimated
        if predicted_duration > estimated_duration * 1.5:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "type": "task_duration",
                "content": f"Your task '{task['title']}' might take longer than estimated. Consider allocating more time.",
                "reason": f"Based on similar tasks, this might take {int(predicted_duration)} minutes instead of the estimated {int(estimated_duration)} minutes.",
                "metadata": {
                    "task_id": task['id'],
                    "estimated_duration": float(estimated_duration),
                    "predicted_duration": float(predicted_duration),
                    "ai_generated": True
                }
            })
    
    # 3. Task prioritization recommendations
    high_priority_tasks = [t for t in tasks if t.get('priority', '').upper() in ['HIGH', 'URGENT'] and not t.get('completed', False)]
    if high_priority_tasks:
        # Sort by due date
        high_priority_tasks.sort(key=lambda t: t.get('dueDate', datetime.max.isoformat()))
        
        recommendations.append({
            "id": str(uuid.uuid4()),
            "type": "task_prioritization",
            "content": f"Focus on '{high_priority_tasks[0]['title']}' as your next task.",
            "reason": "This high-priority task has the nearest deadline among your pending tasks.",
            "metadata": {
                "task_id": high_priority_tasks[0]['id'],
                "ai_generated": True
            }
        })
    
    # 4. Goal progress recommendations
    low_progress_goals = [g for g in goals if g.get('progress', 0) < 0.3 and not g.get('completed', False)]
    if low_progress_goals:
        recommendations.append({
            "id": str(uuid.uuid4()),
            "type": "goal_progress",
            "content": f"Your goal '{low_progress_goals[0]['title']}' needs attention with only {int(low_progress_goals[0].get('progress', 0) * 100)}% progress.",
            "reason": "Regular progress on goals leads to successful completion.",
            "metadata": {
                "goal_id": low_progress_goals[0]['id'],
                "ai_generated": True
            }
        })
    
    # 5. Work-life balance recommendations
    if preferences and preferences.get('breakReminders'):
        recommendations.append({
            "id": str(uuid.uuid4()),
            "type": "wellbeing",
            "content": "Take regular breaks to maintain productivity.",
            "reason": "Research shows that short breaks every 90 minutes can improve focus and productivity.",
            "metadata": {
                "ai_generated": True
            }
        })
    
    # 6. Time management recommendations
    if len(tasks) > 5:
        recommendations.append({
            "id": str(uuid.uuid4()),
            "type": "time_management",
            "content": "Consider grouping similar tasks together to improve efficiency.",
            "reason": "Task batching can reduce context switching and improve overall productivity.",
            "metadata": {
                "ai_generated": True
            }
        })
    
    # Ensure we have at least the requested number of recommendations
    if len(recommendations) < count:
        generic_recommendations = [
            {
                "id": str(uuid.uuid4()),
                "type": "productivity",
                "content": "Try the Pomodoro Technique: 25 minutes of focused work followed by a 5-minute break.",
                "reason": "This technique can help maintain focus and prevent burnout.",
                "metadata": {"ai_generated": True}
            },
            {
                "id": str(uuid.uuid4()),
                "type": "productivity",
                "content": "Consider organizing your tasks by priority and deadline.",
                "reason": "Organized task lists improve productivity and reduce stress.",
                "metadata": {"ai_generated": True}
            },
            {
                "id": str(uuid.uuid4()),
                "type": "wellbeing",
                "content": "Schedule dedicated time for deep work without distractions.",
                "reason": "Deep work sessions can significantly improve productivity and quality of output.",
                "metadata": {"ai_generated": True}
            },
            {
                "id": str(uuid.uuid4()),
                "type": "productivity",
                "content": "Review your completed tasks at the end of each day to track progress.",
                "reason": "Reflecting on accomplishments can boost motivation and help plan for tomorrow.",
                "metadata": {"ai_generated": True}
            }
        ]
        
        # Add generic recommendations until we reach the desired count
        for rec in generic_recommendations:
            if len(recommendations) >= count:
                break
            recommendations.append(rec)
    
    # Take only the requested number of recommendations
    return recommendations[:count]

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Recommendation Generation Script")
    parser.add_argument('--user', required=True, help='User ID')
    parser.add_argument('--count', type=int, default=3, help='Number of recommendations to generate')
    parser.add_argument('--tasks', required=True, help='Tasks data in JSON format')
    parser.add_argument('--goals', required=True, help='Goals data in JSON format')
    parser.add_argument('--preferences', required=True, help='User preferences in JSON format')
    args = parser.parse_args()
    
    try:
        # Parse input data
        user_id = args.user
        count = args.count
        tasks = json.loads(args.tasks)
        goals = json.loads(args.goals)
        preferences = json.loads(args.preferences)
        
        # Generate recommendations
        recommendations = generate_recommendations(user_id, tasks, goals, preferences, count)
        
        # Output recommendations as JSON
        print(json.dumps(recommendations))
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        error_result = {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps([]))
        sys.exit(1)

if __name__ == "__main__":
    main()