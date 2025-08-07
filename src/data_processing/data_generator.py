"""
Synthetic data generator for AstroAssistance.
"""
import os
import json
import random
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import uuid

import numpy as np
import pandas as pd
from faker import Faker

from src.core.data_types import (
    Task, TaskPriority, TaskStatus, TaskCategory,
    Reminder, Goal, UserPreference, UserActivity,
    LearningFeedback, Recommendation, RecurrencePattern
)
from src.core.config import config_manager
from src.core.logger import app_logger


class DataGenerator:
    """Generates synthetic data for training and testing."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)
        
        # Get project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.project_root, "data", "synthetic")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_user_ids(self, num_users: int = 100) -> List[str]:
        """
        Generate a list of user IDs.
        
        Args:
            num_users: Number of users to generate
            
        Returns:
            List of user IDs
        """
        return [str(uuid.uuid4()) for _ in range(num_users)]
    
    def generate_tasks(self, user_ids: List[str], num_tasks_per_user: int = 50) -> List[Task]:
        """
        Generate synthetic tasks.
        
        Args:
            user_ids: List of user IDs
            num_tasks_per_user: Number of tasks to generate per user
            
        Returns:
            List of Task objects
        """
        app_logger.info(f"Generating {num_tasks_per_user} tasks for {len(user_ids)} users")
        
        tasks = []
        
        for user_id in user_ids:
            for _ in range(num_tasks_per_user):
                # Generate random dates within a reasonable range
                created_at = self.faker.date_time_between(start_date="-90d", end_date="now")
                due_date = self.faker.date_time_between(start_date=created_at, end_date="+30d")
                
                # Randomly decide if task is completed
                is_completed = random.random() < 0.4
                completed_at = None
                if is_completed:
                    completed_at = self.faker.date_time_between(start_date=created_at, end_date=due_date)
                    status = TaskStatus.COMPLETED
                else:
                    status = random.choice([s for s in TaskStatus if s != TaskStatus.COMPLETED])
                
                # Generate estimated and actual durations
                estimated_duration = random.randint(15, 240)  # 15 min to 4 hours
                actual_duration = None
                if is_completed:
                    # Actual duration is estimated +/- 30%
                    variation = random.uniform(0.7, 1.3)
                    actual_duration = int(estimated_duration * variation)
                
                # Create task
                task = Task(
                    id=str(uuid.uuid4()),
                    title=self.faker.sentence(nb_words=6)[:-1],  # Remove period
                    description=self.faker.paragraph() if random.random() < 0.7 else None,
                    category=random.choice(list(TaskCategory)),
                    priority=random.choice(list(TaskPriority)),
                    status=status,
                    due_date=due_date,
                    created_at=created_at,
                    updated_at=self.faker.date_time_between(start_date=created_at, end_date="now"),
                    completed_at=completed_at,
                    estimated_duration=estimated_duration,
                    actual_duration=actual_duration,
                    tags=[self.faker.word() for _ in range(random.randint(0, 3))],
                    recurrence=random.choice(list(RecurrencePattern)),
                    user_id=user_id
                )
                tasks.append(task)
        
        return tasks
    
    def generate_reminders(self, tasks: List[Task], reminder_probability: float = 0.6) -> List[Reminder]:
        """
        Generate synthetic reminders based on tasks.
        
        Args:
            tasks: List of Task objects
            reminder_probability: Probability of a task having a reminder
            
        Returns:
            List of Reminder objects
        """
        app_logger.info(f"Generating reminders for {len(tasks)} tasks with probability {reminder_probability}")
        
        reminders = []
        
        for task in tasks:
            if random.random() < reminder_probability and task.due_date:
                # Create reminder some time before the due date
                hours_before = random.randint(1, 48)
                reminder_time = task.due_date - timedelta(hours=hours_before)
                
                # Only create reminder if it's in the future relative to task creation
                if reminder_time > task.created_at:
                    reminder = Reminder(
                        id=str(uuid.uuid4()),
                        task_id=task.id,
                        title=f"Reminder: {task.title}",
                        description=f"Due in {hours_before} hours" if random.random() < 0.5 else None,
                        reminder_time=reminder_time,
                        created_at=task.created_at,
                        is_completed=task.status == TaskStatus.COMPLETED,
                        recurrence=task.recurrence,
                        user_id=task.user_id
                    )
                    reminders.append(reminder)
        
        return reminders
    
    def generate_goals(self, user_ids: List[str], num_goals_per_user: int = 10) -> List[Goal]:
        """
        Generate synthetic goals.
        
        Args:
            user_ids: List of user IDs
            num_goals_per_user: Number of goals to generate per user
            
        Returns:
            List of Goal objects
        """
        app_logger.info(f"Generating {num_goals_per_user} goals for {len(user_ids)} users")
        
        goals = []
        
        for user_id in user_ids:
            for _ in range(num_goals_per_user):
                # Generate random dates
                start_date = self.faker.date_time_between(start_date="-180d", end_date="-30d")
                target_date = self.faker.date_time_between(start_date="+30d", end_date="+365d")
                
                # Randomly decide progress and status
                progress = random.random()
                if progress < 0.1:
                    status = TaskStatus.NOT_STARTED
                elif progress > 0.95:
                    status = TaskStatus.COMPLETED
                    completed_at = self.faker.date_time_between(start_date=start_date, end_date="now")
                else:
                    status = TaskStatus.IN_PROGRESS
                    completed_at = None
                
                # Create milestones
                num_milestones = random.randint(2, 5)
                milestones = []
                for i in range(num_milestones):
                    milestone_progress = (i + 1) / num_milestones
                    milestone_completed = milestone_progress <= progress
                    
                    milestone = {
                        "id": str(uuid.uuid4()),
                        "title": self.faker.sentence(nb_words=4)[:-1],
                        "description": self.faker.sentence() if random.random() < 0.5 else None,
                        "target_progress": milestone_progress,
                        "is_completed": milestone_completed,
                        "completed_at": self.faker.date_time_between(start_date=start_date, end_date="now") if milestone_completed else None
                    }
                    milestones.append(milestone)
                
                # Create goal
                goal = Goal(
                    id=str(uuid.uuid4()),
                    title=self.faker.sentence(nb_words=5)[:-1],
                    description=self.faker.paragraph() if random.random() < 0.8 else None,
                    category=random.choice(list(TaskCategory)),
                    start_date=start_date,
                    target_date=target_date,
                    completed_at=completed_at,
                    status=status,
                    progress=progress,
                    milestones=milestones,
                    metrics={
                        "importance": random.randint(1, 10),
                        "difficulty": random.randint(1, 10)
                    },
                    notes=self.faker.paragraph() if random.random() < 0.3 else None,
                    user_id=user_id
                )
                goals.append(goal)
        
        return goals
    
    def generate_user_preferences(self, user_ids: List[str]) -> List[UserPreference]:
        """
        Generate synthetic user preferences.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            List of UserPreference objects
        """
        app_logger.info(f"Generating preferences for {len(user_ids)} users")
        
        preferences = []
        
        for user_id in user_ids:
            # Generate random productivity peak hours (1-3 ranges)
            num_peak_ranges = random.randint(1, 3)
            peak_hours = []
            for _ in range(num_peak_ranges):
                start_hour = random.randint(6, 20)
                duration = random.randint(2, 4)
                peak_hours.extend([(start_hour + i) % 24 for i in range(duration)])
            
            # Generate work days (most users work weekdays, some have custom schedules)
            if random.random() < 0.8:
                work_days = [0, 1, 2, 3, 4]  # Monday to Friday
            else:
                work_days = sorted(random.sample(range(7), random.randint(3, 6)))
            
            # Generate work hours
            work_start = random.randint(7, 10)
            work_end = random.randint(16, 19)
            
            # Create preference
            preference = UserPreference(
                id=str(uuid.uuid4()),
                user_id=user_id,
                productivity_peak_hours=sorted(peak_hours),
                work_days=work_days,
                work_hours={"start": work_start, "end": work_end},
                preferred_task_duration=random.choice([15, 25, 30, 45, 60, 90]),
                break_duration=random.choice([5, 10, 15, 20]),
                notification_preferences={
                    "email": random.random() < 0.7,
                    "push": random.random() < 0.9,
                    "in_app": random.random() < 0.95
                },
                theme=random.choice(["light", "dark", "system"]),
                language=random.choice(["en", "es", "fr", "de", "zh", "ja"]),
                timezone=self.faker.timezone()
            )
            preferences.append(preference)
        
        return preferences
    
    def generate_user_activities(self, tasks: List[Task], goals: List[Goal], num_activities: int = 1000) -> List[UserActivity]:
        """
        Generate synthetic user activities.
        
        Args:
            tasks: List of Task objects
            goals: List of Goal objects
            num_activities: Number of activities to generate
            
        Returns:
            List of UserActivity objects
        """
        app_logger.info(f"Generating {num_activities} user activities")
        
        activities = []
        
        # Combine tasks and goals for sampling
        all_items = [(task, "task") for task in tasks] + [(goal, "goal") for goal in goals]
        
        for _ in range(num_activities):
            # Select a random item (task or goal)
            item, item_type = random.choice(all_items)
            user_id = item.user_id
            
            # Generate timestamp
            if hasattr(item, 'created_at') and hasattr(item, 'updated_at'):
                timestamp = self.faker.date_time_between(start_date=item.created_at, end_date=item.updated_at)
            else:
                timestamp = self.faker.date_time_between(start_date="-90d", end_date="now")
            
            # Generate activity type and data based on item type
            if item_type == "task":
                activity_type = random.choice([
                    "task_created", "task_updated", "task_completed", 
                    "task_rescheduled", "task_priority_changed"
                ])
                
                if activity_type == "task_created":
                    activity_data = {"task_id": item.id, "title": item.title}
                elif activity_type == "task_updated":
                    activity_data = {
                        "task_id": item.id,
                        "title": item.title,
                        "changes": {
                            "description": random.random() < 0.3,
                            "due_date": random.random() < 0.5,
                            "priority": random.random() < 0.4
                        }
                    }
                elif activity_type == "task_completed":
                    activity_data = {
                        "task_id": item.id,
                        "title": item.title,
                        "completion_time": str(timestamp),
                        "duration": item.actual_duration if item.actual_duration else random.randint(15, 180)
                    }
                elif activity_type == "task_rescheduled":
                    activity_data = {
                        "task_id": item.id,
                        "title": item.title,
                        "old_due_date": str(item.due_date - timedelta(days=random.randint(1, 7))),
                        "new_due_date": str(item.due_date)
                    }
                else:  # task_priority_changed
                    old_priority = random.choice([p for p in TaskPriority if p != item.priority])
                    activity_data = {
                        "task_id": item.id,
                        "title": item.title,
                        "old_priority": old_priority,
                        "new_priority": item.priority
                    }
            else:  # goal
                activity_type = random.choice([
                    "goal_created", "goal_updated", "goal_progress_changed",
                    "goal_completed", "goal_milestone_reached"
                ])
                
                if activity_type == "goal_created":
                    activity_data = {"goal_id": item.id, "title": item.title}
                elif activity_type == "goal_updated":
                    activity_data = {
                        "goal_id": item.id,
                        "title": item.title,
                        "changes": {
                            "description": random.random() < 0.3,
                            "target_date": random.random() < 0.4
                        }
                    }
                elif activity_type == "goal_progress_changed":
                    old_progress = max(0, item.progress - random.uniform(0.05, 0.2))
                    activity_data = {
                        "goal_id": item.id,
                        "title": item.title,
                        "old_progress": old_progress,
                        "new_progress": item.progress
                    }
                elif activity_type == "goal_completed":
                    activity_data = {
                        "goal_id": item.id,
                        "title": item.title,
                        "completion_time": str(timestamp)
                    }
                else:  # goal_milestone_reached
                    if item.milestones:
                        milestone = random.choice(item.milestones)
                        activity_data = {
                            "goal_id": item.id,
                            "title": item.title,
                            "milestone_id": milestone["id"],
                            "milestone_title": milestone["title"],
                            "progress": milestone["target_progress"]
                        }
                    else:
                        continue  # Skip if no milestones
            
            # Generate context
            context = {
                "time_of_day": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "device": random.choice(["desktop", "mobile", "tablet"]),
                "location": random.choice(["home", "work", "commuting", "other"])
            }
            
            # Create activity
            activity = UserActivity(
                id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=timestamp,
                activity_type=activity_type,
                activity_data=activity_data,
                context=context
            )
            activities.append(activity)
        
        return activities
    
    def generate_recommendations(self, tasks: List[Task], goals: List[Goal], user_preferences: List[UserPreference], num_recommendations: int = 500) -> List[Recommendation]:
        """
        Generate synthetic recommendations.
        
        Args:
            tasks: List of Task objects
            goals: List of Goal objects
            user_preferences: List of UserPreference objects
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of Recommendation objects
        """
        app_logger.info(f"Generating {num_recommendations} recommendations")
        
        recommendations = []
        
        # Create user lookup dictionaries
        user_tasks = {}
        user_goals = {}
        user_prefs = {}
        
        for task in tasks:
            if task.user_id not in user_tasks:
                user_tasks[task.user_id] = []
            user_tasks[task.user_id].append(task)
        
        for goal in goals:
            if goal.user_id not in user_goals:
                user_goals[goal.user_id] = []
            user_goals[goal.user_id].append(goal)
        
        for pref in user_preferences:
            user_prefs[pref.user_id] = pref
        
        # Generate recommendations
        for _ in range(num_recommendations):
            # Select a random user who has tasks, goals, and preferences
            valid_users = list(set(user_tasks.keys()) & set(user_goals.keys()) & set(user_prefs.keys()))
            if not valid_users:
                app_logger.warning("No valid users found for generating recommendations")
                break
                
            user_id = random.choice(valid_users)
            user_pref = user_prefs[user_id]
            
            # Generate timestamp
            timestamp = self.faker.date_time_between(start_date="-30d", end_date="now")
            
            # Generate recommendation type
            rec_type = random.choice([
                "task_scheduling", "task_prioritization", "goal_suggestion",
                "productivity_tip", "work_break_reminder", "task_grouping"
            ])
            
            # Generate content based on recommendation type
            if rec_type == "task_scheduling":
                # Recommend a schedule for tasks
                if user_id in user_tasks and len(user_tasks[user_id]) > 0:
                    tasks_to_schedule = random.sample(
                        [t for t in user_tasks[user_id] if t.status != TaskStatus.COMPLETED],
                        min(5, len([t for t in user_tasks[user_id] if t.status != TaskStatus.COMPLETED]))
                    )
                    
                    if not tasks_to_schedule:
                        continue
                    
                    schedule = []
                    current_time = datetime.combine(timestamp.date(), datetime.min.time()) + timedelta(hours=user_pref.work_hours["start"])
                    
                    for task in tasks_to_schedule:
                        duration = task.estimated_duration or random.randint(30, 120)
                        schedule.append({
                            "task_id": task.id,
                            "title": task.title,
                            "start_time": str(current_time),
                            "end_time": str(current_time + timedelta(minutes=duration)),
                            "priority": task.priority
                        })
                        current_time += timedelta(minutes=duration + user_pref.break_duration)
                    
                    content = {
                        "schedule_date": str(timestamp.date()),
                        "tasks": schedule
                    }
                else:
                    continue
            
            elif rec_type == "task_prioritization":
                # Recommend task priority changes
                if user_id in user_tasks and len(user_tasks[user_id]) > 0:
                    tasks_to_prioritize = random.sample(
                        [t for t in user_tasks[user_id] if t.status != TaskStatus.COMPLETED],
                        min(3, len([t for t in user_tasks[user_id] if t.status != TaskStatus.COMPLETED]))
                    )
                    
                    if not tasks_to_prioritize:
                        continue
                    
                    prioritization = []
                    for task in tasks_to_prioritize:
                        new_priority = random.choice([p for p in TaskPriority if p != task.priority])
                        prioritization.append({
                            "task_id": task.id,
                            "title": task.title,
                            "current_priority": task.priority,
                            "recommended_priority": new_priority,
                            "reason": random.choice([
                                "Due date approaching",
                                "Aligns with current goals",
                                "Dependency for other tasks",
                                "Quick win opportunity",
                                "Based on your past completion patterns"
                            ])
                        })
                    
                    content = {"prioritization": prioritization}
                else:
                    continue
            
            elif rec_type == "goal_suggestion":
                # Suggest new goals
                suggested_goals = []
                for _ in range(random.randint(1, 3)):
                    suggested_goals.append({
                        "title": self.faker.sentence(nb_words=5)[:-1],
                        "category": random.choice(list(TaskCategory)),
                        "description": self.faker.paragraph() if random.random() < 0.7 else None,
                        "estimated_duration_days": random.randint(7, 90),
                        "reason": random.choice([
                            "Based on your completed tasks",
                            "Aligns with your current interests",
                            "Popular goal among similar users",
                            "Builds on your recent achievements",
                            "Helps develop new skills"
                        ])
                    })
                
                content = {"suggested_goals": suggested_goals}
            
            elif rec_type == "productivity_tip":
                # Suggest productivity tips
                tip_categories = ["time_management", "focus", "energy", "planning", "delegation"]
                selected_category = random.choice(tip_categories)
                
                tips = {
                    "time_management": [
                        "Try time-blocking your calendar",
                        "Use the Pomodoro technique (25 min work, 5 min break)",
                        "Batch similar tasks together",
                        "Schedule your most important tasks during your peak productivity hours"
                    ],
                    "focus": [
                        "Turn off notifications during deep work",
                        "Use website blockers during focused work sessions",
                        "Try noise-cancelling headphones or background noise",
                        "Clear your desk before starting work"
                    ],
                    "energy": [
                        "Take short walks between tasks",
                        "Try the 2-minute rule: if it takes less than 2 minutes, do it now",
                        "Schedule breaks every 90 minutes",
                        "Alternate between high and low energy tasks"
                    ],
                    "planning": [
                        "Plan tomorrow's tasks at the end of today",
                        "Limit your daily to-do list to 3 major items",
                        "Review your goals weekly",
                        "Use the Eisenhower matrix to prioritize tasks"
                    ],
                    "delegation": [
                        "Identify tasks that could be delegated or automated",
                        "Create templates for recurring tasks",
                        "Use automation tools for repetitive work",
                        "Consider the 80/20 rule: focus on the 20% of work that produces 80% of results"
                    ]
                }
                
                selected_tip = random.choice(tips[selected_category])
                
                content = {
                    "category": selected_category,
                    "tip": selected_tip,
                    "reason": f"Based on your recent {random.choice(['work patterns', 'productivity data', 'task completion times', 'activity analysis'])}"
                }
            
            elif rec_type == "work_break_reminder":
                # Suggest work breaks
                work_duration = random.randint(45, 120)
                break_duration = user_pref.break_duration
                
                break_activities = [
                    "Take a short walk",
                    "Do some stretching exercises",
                    "Rest your eyes by looking at something 20 feet away",
                    "Get a glass of water",
                    "Do a quick meditation session",
                    "Stand up and move around"
                ]
                
                content = {
                    "work_duration_minutes": work_duration,
                    "break_duration_minutes": break_duration,
                    "suggested_activity": random.choice(break_activities),
                    "reason": random.choice([
                        "You've been working continuously",
                        "Taking breaks improves overall productivity",
                        "Regular breaks help maintain focus",
                        "Based on your optimal work rhythm"
                    ])
                }
            
            else:  # task_grouping
                # Suggest grouping related tasks
                if user_id in user_tasks and len(user_tasks[user_id]) >= 5:
                    all_user_tasks = [t for t in user_tasks[user_id] if t.status != TaskStatus.COMPLETED]
                    
                    if len(all_user_tasks) < 5:
                        continue
                    
                    # Create 1-2 groups
                    groups = []
                    for _ in range(random.randint(1, 2)):
                        group_size = random.randint(2, 4)
                        group_tasks = random.sample(all_user_tasks, min(group_size, len(all_user_tasks)))
                        all_user_tasks = [t for t in all_user_tasks if t not in group_tasks]
                        
                        if len(group_tasks) < 2:
                            continue
                        
                        group_name = random.choice([
                            f"{group_tasks[0].category.value.capitalize()} tasks",
                            "Related tasks",
                            "Similar priority tasks",
                            "Tasks with upcoming deadlines",
                            f"Tasks for {self.faker.day_of_week()}"
                        ])
                        
                        groups.append({
                            "name": group_name,
                            "tasks": [{"id": t.id, "title": t.title} for t in group_tasks],
                            "reason": random.choice([
                                "Similar categories",
                                "Related content",
                                "Efficient to complete together",
                                "Similar skill requirements",
                                "Optimal scheduling"
                            ])
                        })
                    
                    if not groups:
                        continue
                        
                    content = {"task_groups": groups}
                else:
                    continue
            
            # Generate confidence score
            confidence = random.uniform(0.65, 0.95)
            
            # Generate explanation
            explanations = {
                "task_scheduling": [
                    "This schedule optimizes your productivity based on your work patterns",
                    "Tasks are arranged to align with your peak productivity hours",
                    "This schedule balances urgent tasks with important long-term goals"
                ],
                "task_prioritization": [
                    "Priorities adjusted based on deadlines and dependencies",
                    "These changes will help you focus on what matters most right now",
                    "Reprioritization based on goal alignment and time constraints"
                ],
                "goal_suggestion": [
                    "These goals align with your current interests and activities",
                    "Based on analysis of your completed tasks and stated preferences",
                    "These suggestions build on your recent achievements"
                ],
                "productivity_tip": [
                    "This tip addresses a pattern observed in your work habits",
                    "This technique has helped users with similar work styles",
                    "Implementing this could improve your productivity by 15-20%"
                ],
                "work_break_reminder": [
                    "Taking breaks at optimal intervals helps maintain peak performance",
                    "Your productivity tends to decline after continuous work",
                    "Regular breaks have been shown to improve overall output quality"
                ],
                "task_grouping": [
                    "Grouping these tasks can save you setup and context-switching time",
                    "These tasks use similar resources or mental processes",
                    "Completing these tasks together creates a better workflow"
                ]
            }
            
            explanation = random.choice(explanations[rec_type])
            
            # Create recommendation
            recommendation = Recommendation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=timestamp,
                recommendation_type=rec_type,
                content=content,
                confidence_score=confidence,
                explanation=explanation,
                is_applied=random.random() < 0.7
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def generate_learning_feedback(self, recommendations: List[Recommendation], feedback_probability: float = 0.8) -> List[LearningFeedback]:
        """
        Generate synthetic learning feedback for recommendations.
        
        Args:
            recommendations: List of Recommendation objects
            feedback_probability: Probability of a recommendation having feedback
            
        Returns:
            List of LearningFeedback objects
        """
        app_logger.info(f"Generating feedback for {len(recommendations)} recommendations with probability {feedback_probability}")
        
        feedbacks = []
        
        for recommendation in recommendations:
            if random.random() < feedback_probability:
                # Determine feedback type based on whether recommendation was applied
                if recommendation.is_applied:
                    feedback_type = random.choices(
                        ["accepted", "modified"],
                        weights=[0.8, 0.2],
                        k=1
                    )[0]
                else:
                    feedback_type = "rejected"
                
                # Create modified recommendation if feedback type is "modified"
                modified_recommendation = None
                if feedback_type == "modified":
                    # Deep copy the original content
                    modified_content = recommendation.content.copy()
                    
                    # Modify based on recommendation type
                    if recommendation.recommendation_type == "task_scheduling":
                        if "tasks" in modified_content["schedule"]:
                            # Modify some task times
                            for task in modified_content["schedule"]["tasks"]:
                                if random.random() < 0.5:
                                    # Adjust start and end times
                                    start_time = datetime.fromisoformat(task["start_time"])
                                    end_time = datetime.fromisoformat(task["end_time"])
                                    
                                    # Shift by -30 to +30 minutes
                                    shift_minutes = random.randint(-30, 30)
                                    start_time += timedelta(minutes=shift_minutes)
                                    end_time += timedelta(minutes=shift_minutes)
                                    
                                    task["start_time"] = start_time.isoformat()
                                    task["end_time"] = end_time.isoformat()
                    
                    elif recommendation.recommendation_type == "task_prioritization":
                        if "prioritization" in modified_content:
                            # Modify some priorities
                            for task in modified_content["prioritization"]:
                                if random.random() < 0.5:
                                    # Choose a different priority
                                    task["recommended_priority"] = random.choice([
                                        p for p in TaskPriority 
                                        if p != task["recommended_priority"] and p != task["current_priority"]
                                    ])
                    
                    # For other recommendation types, make similar modifications
                    modified_recommendation = modified_content
                
                # Generate user comments
                user_comments = None
                if random.random() < 0.4:
                    comments_by_type = {
                        "accepted": [
                            "This works perfectly for me!",
                            "Great suggestion, very helpful.",
                            "Exactly what I needed.",
                            "This makes sense for my schedule."
                        ],
                        "modified": [
                            "Good suggestion but needed some adjustments.",
                            "I tweaked the timing to better fit my day.",
                            "Made a few changes to match my preferences.",
                            "Almost right, just needed minor modifications."
                        ],
                        "rejected": [
                            "This doesn't work with my current priorities.",
                            "Not relevant to what I'm focusing on right now.",
                            "The timing doesn't work for me.",
                            "I have a different approach in mind.",
                            "This conflicts with other commitments."
                        ]
                    }
                    user_comments = random.choice(comments_by_type[feedback_type])
                
                # Create feedback
                feedback = LearningFeedback(
                    id=str(uuid.uuid4()),
                    user_id=recommendation.user_id,
                    timestamp=self.faker.date_time_between(start_date=recommendation.timestamp, end_date=recommendation.timestamp + timedelta(hours=24)),
                    recommendation_id=recommendation.id,
                    feedback_type=feedback_type,
                    original_recommendation=recommendation.content,
                    modified_recommendation=modified_recommendation,
                    user_comments=user_comments,
                    context={
                        "time_of_day": random.randint(0, 23),
                        "day_of_week": random.randint(0, 6),
                        "device": random.choice(["desktop", "mobile", "tablet"]),
                        "location": random.choice(["home", "work", "commuting", "other"])
                    }
                )
                feedbacks.append(feedback)
        
        return feedbacks
    
    def save_data_to_json(self, data, filename: str) -> str:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save (must be JSON serializable)
            filename: Name of the file
            
        Returns:
            Path to the saved file
        """
        # Convert Pydantic models to dictionaries
        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], "dict"):
            data = [item.dict() for item in data]
        
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        app_logger.info(f"Saved data to {file_path}")
        return file_path
    
    def generate_all_data(self, num_users: int = 50) -> Dict[str, str]:
        """
        Generate all synthetic data.
        
        Args:
            num_users: Number of users to generate
            
        Returns:
            Dictionary mapping data type to file path
        """
        app_logger.info(f"Generating synthetic data for {num_users} users")
        
        # Generate user IDs
        user_ids = self.generate_user_ids(num_users)
        
        # Generate tasks
        tasks = self.generate_tasks(user_ids, num_tasks_per_user=30)
        tasks_path = self.save_data_to_json(tasks, "tasks.json")
        
        # Generate reminders
        reminders = self.generate_reminders(tasks)
        reminders_path = self.save_data_to_json(reminders, "reminders.json")
        
        # Generate goals
        goals = self.generate_goals(user_ids, num_goals_per_user=5)
        goals_path = self.save_data_to_json(goals, "goals.json")
        
        # Generate user preferences
        preferences = self.generate_user_preferences(user_ids)
        preferences_path = self.save_data_to_json(preferences, "user_preferences.json")
        
        # Generate user activities
        activities = self.generate_user_activities(tasks, goals, num_activities=500)
        activities_path = self.save_data_to_json(activities, "user_activities.json")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(tasks, goals, preferences, num_recommendations=300)
        recommendations_path = self.save_data_to_json(recommendations, "recommendations.json")
        
        # Generate learning feedback
        feedback = self.generate_learning_feedback(recommendations)
        feedback_path = self.save_data_to_json(feedback, "learning_feedback.json")
        
        # Save user IDs
        user_ids_path = self.save_data_to_json(user_ids, "user_ids.json")
        
        return {
            "user_ids": user_ids_path,
            "tasks": tasks_path,
            "reminders": reminders_path,
            "goals": goals_path,
            "user_preferences": preferences_path,
            "user_activities": activities_path,
            "recommendations": recommendations_path,
            "learning_feedback": feedback_path
        }


if __name__ == "__main__":
    generator = DataGenerator(seed=42)
    file_paths = generator.generate_all_data(num_users=50)
    print(f"Generated data files: {file_paths}")