"""
Service layer for AstroAssistance API.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from src.core.data_types import (
    Task, TaskPriority, TaskStatus, TaskCategory,
    Reminder, Goal, UserPreference, UserActivity,
    LearningFeedback, Recommendation, RecurrencePattern
)
from src.core.logger import app_logger
from src.api.models import (
    CreateTaskRequest, UpdateTaskRequest,
    CreateReminderRequest, UpdateReminderRequest,
    CreateGoalRequest, UpdateGoalRequest,
    UpdateUserPreferenceRequest, FeedbackRequest
)
from src.models.recommendation_model import RecommendationModel


# Mock database for demonstration purposes
# In a real implementation, this would be replaced with a proper database
class MockDatabase:
    """Mock database for demonstration purposes."""
    
    _tasks = {}
    _reminders = {}
    _goals = {}
    _user_preferences = {}
    _recommendations = {}
    _feedback = {}
    
    @classmethod
    def get_tasks(cls, user_id: str) -> List[Task]:
        """Get tasks for a user."""
        return [task for task in cls._tasks.values() if task.user_id == user_id]
    
    @classmethod
    def get_task(cls, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return cls._tasks.get(task_id)
    
    @classmethod
    def add_task(cls, task: Task) -> Task:
        """Add a task."""
        cls._tasks[task.id] = task
        return task
    
    @classmethod
    def update_task(cls, task: Task) -> Task:
        """Update a task."""
        cls._tasks[task.id] = task
        return task
    
    @classmethod
    def delete_task(cls, task_id: str) -> bool:
        """Delete a task."""
        if task_id in cls._tasks:
            del cls._tasks[task_id]
            return True
        return False
    
    @classmethod
    def get_reminders(cls, user_id: str) -> List[Reminder]:
        """Get reminders for a user."""
        return [reminder for reminder in cls._reminders.values() if reminder.user_id == user_id]
    
    @classmethod
    def get_reminder(cls, reminder_id: str) -> Optional[Reminder]:
        """Get a reminder by ID."""
        return cls._reminders.get(reminder_id)
    
    @classmethod
    def add_reminder(cls, reminder: Reminder) -> Reminder:
        """Add a reminder."""
        cls._reminders[reminder.id] = reminder
        return reminder
    
    @classmethod
    def update_reminder(cls, reminder: Reminder) -> Reminder:
        """Update a reminder."""
        cls._reminders[reminder.id] = reminder
        return reminder
    
    @classmethod
    def delete_reminder(cls, reminder_id: str) -> bool:
        """Delete a reminder."""
        if reminder_id in cls._reminders:
            del cls._reminders[reminder_id]
            return True
        return False
    
    @classmethod
    def get_goals(cls, user_id: str) -> List[Goal]:
        """Get goals for a user."""
        return [goal for goal in cls._goals.values() if goal.user_id == user_id]
    
    @classmethod
    def get_goal(cls, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return cls._goals.get(goal_id)
    
    @classmethod
    def add_goal(cls, goal: Goal) -> Goal:
        """Add a goal."""
        cls._goals[goal.id] = goal
        return goal
    
    @classmethod
    def update_goal(cls, goal: Goal) -> Goal:
        """Update a goal."""
        cls._goals[goal.id] = goal
        return goal
    
    @classmethod
    def delete_goal(cls, goal_id: str) -> bool:
        """Delete a goal."""
        if goal_id in cls._goals:
            del cls._goals[goal_id]
            return True
        return False
    
    @classmethod
    def get_user_preferences(cls, user_id: str) -> Optional[UserPreference]:
        """Get user preferences."""
        return cls._user_preferences.get(user_id)
    
    @classmethod
    def add_user_preferences(cls, preferences: UserPreference) -> UserPreference:
        """Add user preferences."""
        cls._user_preferences[preferences.user_id] = preferences
        return preferences
    
    @classmethod
    def update_user_preferences(cls, preferences: UserPreference) -> UserPreference:
        """Update user preferences."""
        cls._user_preferences[preferences.user_id] = preferences
        return preferences
    
    @classmethod
    def get_recommendations(cls, user_id: str) -> List[Recommendation]:
        """Get recommendations for a user."""
        return [rec for rec in cls._recommendations.values() if rec.user_id == user_id]
    
    @classmethod
    def get_recommendation(cls, recommendation_id: str) -> Optional[Recommendation]:
        """Get a recommendation by ID."""
        return cls._recommendations.get(recommendation_id)
    
    @classmethod
    def add_recommendation(cls, recommendation: Recommendation) -> Recommendation:
        """Add a recommendation."""
        cls._recommendations[recommendation.id] = recommendation
        return recommendation
    
    @classmethod
    def update_recommendation(cls, recommendation: Recommendation) -> Recommendation:
        """Update a recommendation."""
        cls._recommendations[recommendation.id] = recommendation
        return recommendation
    
    @classmethod
    def add_feedback(cls, feedback: LearningFeedback) -> LearningFeedback:
        """Add feedback."""
        cls._feedback[feedback.id] = feedback
        return feedback


class TaskService:
    """Service for task operations."""
    
    def create_task(self, user_id: str, task_request: CreateTaskRequest) -> Task:
        """
        Create a new task.
        
        Args:
            user_id: User ID
            task_request: Task creation request
            
        Returns:
            Created task
        """
        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            title=task_request.title,
            description=task_request.description,
            category=task_request.category,
            priority=task_request.priority,
            status=TaskStatus.NOT_STARTED,
            due_date=task_request.due_date,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            estimated_duration=task_request.estimated_duration,
            tags=task_request.tags,
            recurrence=task_request.recurrence,
            recurrence_end_date=task_request.recurrence_end_date,
            parent_task_id=task_request.parent_task_id,
            user_id=user_id
        )
        
        # Add task to database
        return MockDatabase.add_task(task)
    
    def get_tasks(self, user_id: str, filters: Dict[str, Any], page: int, page_size: int) -> Tuple[List[Task], int]:
        """
        Get tasks with pagination and filtering.
        
        Args:
            user_id: User ID
            filters: Filters to apply
            page: Page number
            page_size: Page size
            
        Returns:
            Tuple of (tasks, total_count)
        """
        # Get all tasks for user
        all_tasks = MockDatabase.get_tasks(user_id)
        
        # Apply filters
        filtered_tasks = all_tasks
        
        if filters.get("status"):
            filtered_tasks = [t for t in filtered_tasks if t.status.value == filters["status"]]
        
        if filters.get("category"):
            filtered_tasks = [t for t in filtered_tasks if t.category.value == filters["category"]]
        
        if filters.get("priority"):
            filtered_tasks = [t for t in filtered_tasks if t.priority.value == filters["priority"]]
        
        if filters.get("due_date_start"):
            filtered_tasks = [t for t in filtered_tasks if t.due_date and t.due_date >= filters["due_date_start"]]
        
        if filters.get("due_date_end"):
            filtered_tasks = [t for t in filtered_tasks if t.due_date and t.due_date <= filters["due_date_end"]]
        
        # Get total count
        total = len(filtered_tasks)
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_tasks = filtered_tasks[start_idx:end_idx]
        
        return paginated_tasks, total
    
    def get_task(self, user_id: str, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            user_id: User ID
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        # Get task
        task = MockDatabase.get_task(task_id)
        
        # Check if task exists and belongs to user
        if task and task.user_id == user_id:
            return task
        
        return None
    
    def update_task(self, user_id: str, task_id: str, task_request: UpdateTaskRequest) -> Optional[Task]:
        """
        Update a task.
        
        Args:
            user_id: User ID
            task_id: Task ID
            task_request: Task update request
            
        Returns:
            Updated task or None if not found
        """
        # Get task
        task = self.get_task(user_id, task_id)
        
        if task is None:
            return None
        
        # Update task fields
        if task_request.title is not None:
            task.title = task_request.title
        
        if task_request.description is not None:
            task.description = task_request.description
        
        if task_request.category is not None:
            task.category = task_request.category
        
        if task_request.priority is not None:
            task.priority = task_request.priority
        
        if task_request.status is not None:
            task.status = task_request.status
            
            # If task is completed, set completed_at
            if task.status == TaskStatus.COMPLETED and task.completed_at is None:
                task.completed_at = datetime.now()
            # If task is not completed, clear completed_at
            elif task.status != TaskStatus.COMPLETED:
                task.completed_at = None
        
        if task_request.due_date is not None:
            task.due_date = task_request.due_date
        
        if task_request.estimated_duration is not None:
            task.estimated_duration = task_request.estimated_duration
        
        if task_request.actual_duration is not None:
            task.actual_duration = task_request.actual_duration
        
        if task_request.tags is not None:
            task.tags = task_request.tags
        
        if task_request.recurrence is not None:
            task.recurrence = task_request.recurrence
        
        if task_request.recurrence_end_date is not None:
            task.recurrence_end_date = task_request.recurrence_end_date
        
        if task_request.notes is not None:
            task.notes = task_request.notes
        
        # Update updated_at
        task.updated_at = datetime.now()
        
        # Update task in database
        return MockDatabase.update_task(task)
    
    def delete_task(self, user_id: str, task_id: str) -> bool:
        """
        Delete a task.
        
        Args:
            user_id: User ID
            task_id: Task ID
            
        Returns:
            True if task was deleted, False otherwise
        """
        # Get task
        task = self.get_task(user_id, task_id)
        
        if task is None:
            return False
        
        # Delete task
        return MockDatabase.delete_task(task_id)


class ReminderService:
    """Service for reminder operations."""
    
    def create_reminder(self, user_id: str, reminder_request: CreateReminderRequest) -> Reminder:
        """
        Create a new reminder.
        
        Args:
            user_id: User ID
            reminder_request: Reminder creation request
            
        Returns:
            Created reminder
        """
        # Create reminder
        reminder = Reminder(
            id=str(uuid.uuid4()),
            task_id=reminder_request.task_id,
            title=reminder_request.title,
            description=reminder_request.description,
            reminder_time=reminder_request.reminder_time,
            created_at=datetime.now(),
            is_completed=False,
            recurrence=reminder_request.recurrence,
            recurrence_end_date=reminder_request.recurrence_end_date,
            user_id=user_id
        )
        
        # Add reminder to database
        return MockDatabase.add_reminder(reminder)
    
    def get_reminders(self, user_id: str, filters: Dict[str, Any], page: int, page_size: int) -> Tuple[List[Reminder], int]:
        """
        Get reminders with pagination and filtering.
        
        Args:
            user_id: User ID
            filters: Filters to apply
            page: Page number
            page_size: Page size
            
        Returns:
            Tuple of (reminders, total_count)
        """
        # Get all reminders for user
        all_reminders = MockDatabase.get_reminders(user_id)
        
        # Apply filters
        filtered_reminders = all_reminders
        
        if filters.get("task_id"):
            filtered_reminders = [r for r in filtered_reminders if r.task_id == filters["task_id"]]
        
        if filters.get("is_completed") is not None:
            filtered_reminders = [r for r in filtered_reminders if r.is_completed == filters["is_completed"]]
        
        if filters.get("reminder_time_start"):
            filtered_reminders = [r for r in filtered_reminders if r.reminder_time >= filters["reminder_time_start"]]
        
        if filters.get("reminder_time_end"):
            filtered_reminders = [r for r in filtered_reminders if r.reminder_time <= filters["reminder_time_end"]]
        
        # Get total count
        total = len(filtered_reminders)
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_reminders = filtered_reminders[start_idx:end_idx]
        
        return paginated_reminders, total
    
    def get_reminder(self, user_id: str, reminder_id: str) -> Optional[Reminder]:
        """
        Get a reminder by ID.
        
        Args:
            user_id: User ID
            reminder_id: Reminder ID
            
        Returns:
            Reminder or None if not found
        """
        # Get reminder
        reminder = MockDatabase.get_reminder(reminder_id)
        
        # Check if reminder exists and belongs to user
        if reminder and reminder.user_id == user_id:
            return reminder
        
        return None
    
    def update_reminder(self, user_id: str, reminder_id: str, reminder_request: UpdateReminderRequest) -> Optional[Reminder]:
        """
        Update a reminder.
        
        Args:
            user_id: User ID
            reminder_id: Reminder ID
            reminder_request: Reminder update request
            
        Returns:
            Updated reminder or None if not found
        """
        # Get reminder
        reminder = self.get_reminder(user_id, reminder_id)
        
        if reminder is None:
            return None
        
        # Update reminder fields
        if reminder_request.task_id is not None:
            reminder.task_id = reminder_request.task_id
        
        if reminder_request.title is not None:
            reminder.title = reminder_request.title
        
        if reminder_request.description is not None:
            reminder.description = reminder_request.description
        
        if reminder_request.reminder_time is not None:
            reminder.reminder_time = reminder_request.reminder_time
        
        if reminder_request.is_completed is not None:
            reminder.is_completed = reminder_request.is_completed
        
        if reminder_request.recurrence is not None:
            reminder.recurrence = reminder_request.recurrence
        
        if reminder_request.recurrence_end_date is not None:
            reminder.recurrence_end_date = reminder_request.recurrence_end_date
        
        # Update reminder in database
        return MockDatabase.update_reminder(reminder)
    
    def delete_reminder(self, user_id: str, reminder_id: str) -> bool:
        """
        Delete a reminder.
        
        Args:
            user_id: User ID
            reminder_id: Reminder ID
            
        Returns:
            True if reminder was deleted, False otherwise
        """
        # Get reminder
        reminder = self.get_reminder(user_id, reminder_id)
        
        if reminder is None:
            return False
        
        # Delete reminder
        return MockDatabase.delete_reminder(reminder_id)


class GoalService:
    """Service for goal operations."""
    
    def create_goal(self, user_id: str, goal_request: CreateGoalRequest) -> Goal:
        """
        Create a new goal.
        
        Args:
            user_id: User ID
            goal_request: Goal creation request
            
        Returns:
            Created goal
        """
        # Create goal
        goal = Goal(
            id=str(uuid.uuid4()),
            title=goal_request.title,
            description=goal_request.description,
            category=goal_request.category,
            start_date=datetime.now(),
            target_date=goal_request.target_date,
            status=TaskStatus.NOT_STARTED,
            progress=0.0,
            related_tasks=goal_request.related_tasks,
            milestones=goal_request.milestones,
            metrics=goal_request.metrics,
            notes=goal_request.notes,
            user_id=user_id
        )
        
        # Add goal to database
        return MockDatabase.add_goal(goal)
    
    def get_goals(self, user_id: str, filters: Dict[str, Any], page: int, page_size: int) -> Tuple[List[Goal], int]:
        """
        Get goals with pagination and filtering.
        
        Args:
            user_id: User ID
            filters: Filters to apply
            page: Page number
            page_size: Page size
            
        Returns:
            Tuple of (goals, total_count)
        """
        # Get all goals for user
        all_goals = MockDatabase.get_goals(user_id)
        
        # Apply filters
        filtered_goals = all_goals
        
        if filters.get("status"):
            filtered_goals = [g for g in filtered_goals if g.status.value == filters["status"]]
        
        if filters.get("category"):
            filtered_goals = [g for g in filtered_goals if g.category.value == filters["category"]]
        
        # Get total count
        total = len(filtered_goals)
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_goals = filtered_goals[start_idx:end_idx]
        
        return paginated_goals, total
    
    def get_goal(self, user_id: str, goal_id: str) -> Optional[Goal]:
        """
        Get a goal by ID.
        
        Args:
            user_id: User ID
            goal_id: Goal ID
            
        Returns:
            Goal or None if not found
        """
        # Get goal
        goal = MockDatabase.get_goal(goal_id)
        
        # Check if goal exists and belongs to user
        if goal and goal.user_id == user_id:
            return goal
        
        return None
    
    def update_goal(self, user_id: str, goal_id: str, goal_request: UpdateGoalRequest) -> Optional[Goal]:
        """
        Update a goal.
        
        Args:
            user_id: User ID
            goal_id: Goal ID
            goal_request: Goal update request
            
        Returns:
            Updated goal or None if not found
        """
        # Get goal
        goal = self.get_goal(user_id, goal_id)
        
        if goal is None:
            return None
        
        # Update goal fields
        if goal_request.title is not None:
            goal.title = goal_request.title
        
        if goal_request.description is not None:
            goal.description = goal_request.description
        
        if goal_request.category is not None:
            goal.category = goal_request.category
        
        if goal_request.target_date is not None:
            goal.target_date = goal_request.target_date
        
        if goal_request.status is not None:
            goal.status = goal_request.status
            
            # If goal is completed, set completed_at
            if goal.status == TaskStatus.COMPLETED and goal.completed_at is None:
                goal.completed_at = datetime.now()
                goal.progress = 1.0
            # If goal is not completed, clear completed_at
            elif goal.status != TaskStatus.COMPLETED:
                goal.completed_at = None
        
        if goal_request.progress is not None:
            goal.progress = goal_request.progress
            
            # Update status based on progress
            if goal.progress >= 1.0 and goal.status != TaskStatus.COMPLETED:
                goal.status = TaskStatus.COMPLETED
                goal.completed_at = datetime.now()
            elif goal.progress > 0.0 and goal.status == TaskStatus.NOT_STARTED:
                goal.status = TaskStatus.IN_PROGRESS
        
        if goal_request.related_tasks is not None:
            goal.related_tasks = goal_request.related_tasks
        
        if goal_request.milestones is not None:
            goal.milestones = goal_request.milestones
        
        if goal_request.metrics is not None:
            goal.metrics = goal_request.metrics
        
        if goal_request.notes is not None:
            goal.notes = goal_request.notes
        
        # Update goal in database
        return MockDatabase.update_goal(goal)
    
    def delete_goal(self, user_id: str, goal_id: str) -> bool:
        """
        Delete a goal.
        
        Args:
            user_id: User ID
            goal_id: Goal ID
            
        Returns:
            True if goal was deleted, False otherwise
        """
        # Get goal
        goal = self.get_goal(user_id, goal_id)
        
        if goal is None:
            return False
        
        # Delete goal
        return MockDatabase.delete_goal(goal_id)


class UserPreferenceService:
    """Service for user preference operations."""
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """
        Get user preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            User preferences or None if not found
        """
        return MockDatabase.get_user_preferences(user_id)
    
    def create_default_preferences(self, user_id: str) -> UserPreference:
        """
        Create default user preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Created user preferences
        """
        # Create default preferences
        preferences = UserPreference(
            id=str(uuid.uuid4()),
            user_id=user_id,
            productivity_peak_hours=[9, 10, 11, 14, 15, 16],
            work_days=[0, 1, 2, 3, 4],  # Monday to Friday
            work_hours={"start": 9, "end": 17},
            preferred_task_duration=30,
            break_duration=5,
            notification_preferences={
                "email": True,
                "push": True,
                "in_app": True
            },
            theme="light",
            language="en",
            timezone="UTC"
        )
        
        # Add preferences to database
        return MockDatabase.add_user_preferences(preferences)
    
    def update_user_preferences(self, user_id: str, preference_request: UpdateUserPreferenceRequest) -> UserPreference:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            preference_request: Preference update request
            
        Returns:
            Updated user preferences
        """
        # Get preferences
        preferences = self.get_user_preferences(user_id)
        
        # Create default preferences if none exist
        if preferences is None:
            preferences = self.create_default_preferences(user_id)
        
        # Update preference fields
        if preference_request.productivity_peak_hours is not None:
            preferences.productivity_peak_hours = preference_request.productivity_peak_hours
        
        if preference_request.work_days is not None:
            preferences.work_days = preference_request.work_days
        
        if preference_request.work_hours is not None:
            preferences.work_hours = preference_request.work_hours
        
        if preference_request.preferred_task_duration is not None:
            preferences.preferred_task_duration = preference_request.preferred_task_duration
        
        if preference_request.break_duration is not None:
            preferences.break_duration = preference_request.break_duration
        
        if preference_request.notification_preferences is not None:
            preferences.notification_preferences = preference_request.notification_preferences
        
        if preference_request.theme is not None:
            preferences.theme = preference_request.theme
        
        if preference_request.language is not None:
            preferences.language = preference_request.language
        
        if preference_request.timezone is not None:
            preferences.timezone = preference_request.timezone
        
        # Update preferences in database
        return MockDatabase.update_user_preferences(preferences)


class RecommendationService:
    """Service for recommendation operations."""
    
    def __init__(self):
        """Initialize the recommendation service."""
        # In a real implementation, this would load the trained model
        self.model = None
        try:
            self.model = RecommendationModel()
            self.model.load()
        except Exception as e:
            app_logger.warning(f"Could not load recommendation model: {str(e)}")
    
    def get_recommendations(self, user_id: str, filters: Dict[str, Any], page: int, page_size: int) -> Tuple[List[Recommendation], int]:
        """
        Get recommendations with pagination and filtering.
        
        Args:
            user_id: User ID
            filters: Filters to apply
            page: Page number
            page_size: Page size
            
        Returns:
            Tuple of (recommendations, total_count)
        """
        # Get all recommendations for user
        all_recommendations = MockDatabase.get_recommendations(user_id)
        
        # Apply filters
        filtered_recommendations = all_recommendations
        
        if filters.get("recommendation_type"):
            filtered_recommendations = [r for r in filtered_recommendations if r.recommendation_type == filters["recommendation_type"]]
        
        if filters.get("min_confidence") is not None:
            filtered_recommendations = [r for r in filtered_recommendations if r.confidence_score >= filters["min_confidence"]]
        
        # Get total count
        total = len(filtered_recommendations)
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_recommendations = filtered_recommendations[start_idx:end_idx]
        
        return paginated_recommendations, total
    
    def generate_recommendations(self, user_id: str, count: int) -> List[Recommendation]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            count: Number of recommendations to generate
            
        Returns:
            List of generated recommendations
        """
        # Get user data
        tasks = MockDatabase.get_tasks(user_id)
        goals = MockDatabase.get_goals(user_id)
        preferences = MockDatabase.get_user_preferences(user_id)
        
        # Create default preferences if none exist
        if preferences is None:
            preference_service = UserPreferenceService()
            preferences = preference_service.create_default_preferences(user_id)
        
        # Generate recommendations
        recommendations = []
        
        if self.model and self.model.is_trained:
            # Use trained model to generate recommendations
            model_recommendations = self.model.generate_recommendations(user_id, tasks, goals, preferences)
            recommendations.extend(model_recommendations)
        else:
            # Generate placeholder recommendations
            for i in range(count):
                recommendation = self._generate_placeholder_recommendation(user_id, i)
                recommendations.append(recommendation)
        
        # Add recommendations to database
        for recommendation in recommendations:
            MockDatabase.add_recommendation(recommendation)
        
        return recommendations
    
    def _generate_placeholder_recommendation(self, user_id: str, index: int) -> Recommendation:
        """
        Generate a placeholder recommendation.
        
        Args:
            user_id: User ID
            index: Recommendation index
            
        Returns:
            Generated recommendation
        """
        # Define recommendation types
        rec_types = [
            "task_scheduling",
            "task_prioritization",
            "goal_suggestion",
            "productivity_tip",
            "work_break_reminder",
            "task_grouping"
        ]
        
        # Select recommendation type based on index
        rec_type = rec_types[index % len(rec_types)]
        
        # Generate content based on recommendation type
        if rec_type == "task_scheduling":
            content = {
                "schedule_date": datetime.now().strftime("%Y-%m-%d"),
                "tasks": [
                    {
                        "task_id": f"task{index}",
                        "title": "Complete project proposal",
                        "start_time": "09:00",
                        "end_time": "10:30",
                        "priority": "high"
                    },
                    {
                        "task_id": f"task{index+1}",
                        "title": "Team meeting",
                        "start_time": "11:00",
                        "end_time": "12:00",
                        "priority": "medium"
                    }
                ]
            }
            explanation = "This schedule optimizes your productivity based on your work patterns."
        
        elif rec_type == "task_prioritization":
            content = {
                "prioritization": [
                    {
                        "task_id": f"task{index+2}",
                        "title": "Review documentation",
                        "current_priority": "low",
                        "recommended_priority": "medium",
                        "reason": "Dependency for other tasks"
                    }
                ]
            }
            explanation = "Priorities adjusted based on deadlines and dependencies."
        
        elif rec_type == "goal_suggestion":
            content = {
                "suggested_goals": [
                    {
                        "title": "Learn a new programming language",
                        "category": "education",
                        "description": "Expand your skill set by learning a new programming language",
                        "estimated_duration_days": 90,
                        "reason": "Based on your completed tasks"
                    }
                ]
            }
            explanation = "These goals align with your current interests and activities."
        
        elif rec_type == "productivity_tip":
            content = {
                "category": "focus",
                "tip": "Use the Pomodoro technique (25 min work, 5 min break)",
                "reason": "Based on your recent work patterns"
            }
            explanation = "This tip addresses a pattern observed in your work habits."
        
        elif rec_type == "work_break_reminder":
            content = {
                "work_duration_minutes": 55,
                "break_duration_minutes": 5,
                "suggested_activity": "Take a short walk",
                "reason": "You've been working continuously"
            }
            explanation = "Taking breaks at optimal intervals helps maintain peak performance."
        
        else:  # task_grouping
            content = {
                "task_groups": [
                    {
                        "name": "Documentation tasks",
                        "tasks": [
                            {"id": f"task{index+3}", "title": "Update README"},
                            {"id": f"task{index+4}", "title": "Write API documentation"}
                        ],
                        "reason": "Similar categories"
                    }
                ]
            }
            explanation = "Grouping these tasks can save you setup and context-switching time."
        
        # Create recommendation
        recommendation = Recommendation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(),
            recommendation_type=rec_type,
            content=content,
            confidence_score=0.85,
            explanation=explanation,
            is_applied=False
        )
        
        return recommendation
    
    def process_feedback(self, user_id: str, feedback_request: FeedbackRequest) -> bool:
        """
        Process feedback on a recommendation.
        
        Args:
            user_id: User ID
            feedback_request: Feedback request
            
        Returns:
            True if feedback was processed successfully, False otherwise
        """
        # Get recommendation
        recommendation = MockDatabase.get_recommendation(feedback_request.recommendation_id)
        
        if recommendation is None or recommendation.user_id != user_id:
            return False
        
        # Create feedback
        feedback = LearningFeedback(
            id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(),
            recommendation_id=feedback_request.recommendation_id,
            feedback_type=feedback_request.feedback_type,
            original_recommendation=recommendation.content,
            modified_recommendation=feedback_request.modified_recommendation,
            user_comments=feedback_request.user_comments,
            context={
                "time_of_day": datetime.now().hour,
                "day_of_week": datetime.now().weekday()
            }
        )
        
        # Add feedback to database
        MockDatabase.add_feedback(feedback)
        
        # Update recommendation
        recommendation.is_applied = feedback_request.feedback_type in ["accepted", "modified"]
        MockDatabase.update_recommendation(recommendation)
        
        # In a real implementation, this would trigger model retraining or updating
        
        return True