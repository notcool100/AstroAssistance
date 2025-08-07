#!/usr/bin/env python3
"""
Recommendation Generation Script

This script generates personalized recommendations based on user data.
"""

import os
import sys
import json
import argparse
import logging
import random
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recommendation_generator')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate personalized recommendations')
    parser.add_argument('--user', type=str, required=True, help='User ID')
    parser.add_argument('--count', type=int, default=3, help='Number of recommendations to generate')
    parser.add_argument('--tasks', type=str, required=True, help='User tasks in JSON format')
    parser.add_argument('--goals', type=str, required=True, help='User goals in JSON format')
    parser.add_argument('--preferences', type=str, required=True, help='User preferences in JSON format')
    return parser.parse_args()

def analyze_tasks(tasks):
    """Analyze tasks to identify patterns and issues."""
    insights = []
    
    # Check for overdue tasks
    overdue_tasks = [t for t in tasks if t.get('dueDate') and 
                     datetime.fromisoformat(t.get('dueDate').replace('Z', '+00:00')) < datetime.now()]
    if overdue_tasks:
        insights.append({
            'type': 'task_overdue',
            'content': f"You have {len(overdue_tasks)} overdue tasks. Consider rescheduling or prioritizing them.",
            'reason': 'Overdue tasks can cause stress and affect productivity',
            'metadata': {'confidence': 0.9}
        })
    
    # Check for tasks with approaching deadlines
    upcoming_tasks = [t for t in tasks if t.get('dueDate') and 
                      datetime.now() < datetime.fromisoformat(t.get('dueDate').replace('Z', '+00:00')) < 
                      datetime.now() + timedelta(days=2)]
    if upcoming_tasks:
        insights.append({
            'type': 'task_upcoming',
            'content': f"'{upcoming_tasks[0].get('title')}' is due soon. Make it a priority.",
            'reason': 'Tasks with approaching deadlines need immediate attention',
            'metadata': {'confidence': 0.85}
        })
    
    # Check for high priority tasks
    high_priority = [t for t in tasks if t.get('priority') == 'HIGH']
    if high_priority:
        insights.append({
            'type': 'task_priority',
            'content': f"Focus on completing '{high_priority[0].get('title')}' as it's a high priority task.",
            'reason': 'High priority tasks should be completed first',
            'metadata': {'confidence': 0.8}
        })
    
    # Check for task clustering
    categories = {}
    for task in tasks:
        category = task.get('category', 'UNCATEGORIZED')
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    
    if categories:
        most_common = max(categories.items(), key=lambda x: x[1])
        if most_common[1] > 3:
            insights.append({
                'type': 'task_clustering',
                'content': f"You have {most_common[1]} tasks in the '{most_common[0]}' category. Consider spreading your focus across different areas.",
                'reason': 'Balancing tasks across categories can improve overall productivity',
                'metadata': {'confidence': 0.7}
            })
    
    return insights

def analyze_goals(goals):
    """Analyze goals to identify patterns and issues."""
    insights = []
    
    # Check for goals with low progress
    low_progress = [g for g in goals if g.get('progress', 0) < 0.3]
    if low_progress:
        insights.append({
            'type': 'goal_progress',
            'content': f"Your goal '{low_progress[0].get('title')}' has low progress. Consider breaking it down into smaller tasks.",
            'reason': 'Breaking down goals into smaller tasks makes them more achievable',
            'metadata': {'confidence': 0.85}
        })
    
    # Check for goals with approaching deadlines
    upcoming_goals = [g for g in goals if g.get('dueDate') and 
                      datetime.now() < datetime.fromisoformat(g.get('dueDate').replace('Z', '+00:00')) < 
                      datetime.now() + timedelta(days=7)]
    if upcoming_goals:
        insights.append({
            'type': 'goal_deadline',
            'content': f"Your goal '{upcoming_goals[0].get('title')}' is due soon. Focus on making progress.",
            'reason': 'Goals with approaching deadlines need focused attention',
            'metadata': {'confidence': 0.8}
        })
    
    return insights

def generate_productivity_tips():
    """Generate general productivity tips."""
    tips = [
        {
            'type': 'productivity',
            'content': "Try the Pomodoro Technique: 25 minutes of focused work followed by a 5-minute break.",
            'reason': 'Time-boxing can improve focus and productivity',
            'metadata': {'confidence': 0.75}
        },
        {
            'type': 'productivity',
            'content': "Consider using the 2-minute rule: if a task takes less than 2 minutes, do it immediately.",
            'reason': 'Small tasks can accumulate and cause mental overhead',
            'metadata': {'confidence': 0.7}
        },
        {
            'type': 'productivity',
            'content': "Try batching similar tasks together to reduce context switching.",
            'reason': 'Context switching can reduce productivity by up to 40%',
            'metadata': {'confidence': 0.8}
        },
        {
            'type': 'productivity',
            'content': "Schedule your most important tasks during your peak energy hours.",
            'reason': 'Aligning tasks with your energy levels improves efficiency',
            'metadata': {'confidence': 0.75}
        },
        {
            'type': 'wellbeing',
            'content': "Take regular breaks to prevent burnout and maintain productivity.",
            'reason': 'Regular breaks improve overall productivity and mental health',
            'metadata': {'confidence': 0.85}
        },
        {
            'type': 'wellbeing',
            'content': "Consider implementing a daily mindfulness practice to improve focus.",
            'reason': 'Mindfulness can reduce stress and improve concentration',
            'metadata': {'confidence': 0.7}
        }
    ]
    return random.sample(tips, min(3, len(tips)))

def main():
    """Main function to generate recommendations."""
    args = parse_arguments()
    
    try:
        # Parse the input data
        user_id = args.user
        count = args.count
        tasks = json.loads(args.tasks)
        goals = json.loads(args.goals)
        preferences = json.loads(args.preferences)
        
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Analyze user data
        task_insights = analyze_tasks(tasks)
        goal_insights = analyze_goals(goals)
        general_tips = generate_productivity_tips()
        
        # Combine all insights
        all_insights = task_insights + goal_insights + general_tips
        
        # Select the top recommendations based on confidence
        all_insights.sort(key=lambda x: x.get('metadata', {}).get('confidence', 0), reverse=True)
        recommendations = all_insights[:count]
        
        # Output the recommendations as JSON
        print(json.dumps(recommendations))
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()