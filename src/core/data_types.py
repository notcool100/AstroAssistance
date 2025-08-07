"""
Core data types and structures for AstroAssistance.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Status options for tasks."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class TaskCategory(str, Enum):
    """Categories for tasks."""
    WORK = "work"
    PERSONAL = "personal"
    HEALTH = "health"
    FINANCE = "finance"
    EDUCATION = "education"
    SOCIAL = "social"
    OTHER = "other"


class RecurrencePattern(str, Enum):
    """Recurrence patterns for tasks and reminders."""
    NONE = "none"
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class Task(BaseModel):
    """Task data model."""
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    category: TaskCategory = TaskCategory.OTHER
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.NOT_STARTED
    due_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # in minutes
    actual_duration: Optional[int] = None  # in minutes
    tags: List[str] = []
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_end_date: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = []
    notes: Optional[str] = None
    attachments: List[str] = []
    user_id: str


class Reminder(BaseModel):
    """Reminder data model."""
    id: Optional[str] = None
    task_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    reminder_time: datetime
    created_at: datetime = Field(default_factory=datetime.now)
    is_completed: bool = False
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_end_date: Optional[datetime] = None
    user_id: str


class Goal(BaseModel):
    """Goal data model."""
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    category: TaskCategory = TaskCategory.OTHER
    start_date: datetime = Field(default_factory=datetime.now)
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    progress: float = 0.0  # 0.0 to 1.0
    related_tasks: List[str] = []
    milestones: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    notes: Optional[str] = None
    user_id: str


class UserPreference(BaseModel):
    """User preferences data model."""
    id: Optional[str] = None
    user_id: str
    productivity_peak_hours: List[int] = []  # 0-23 hours
    work_days: List[int] = [0, 1, 2, 3, 4]  # 0=Monday, 6=Sunday
    work_hours: Dict[str, Any] = {"start": 9, "end": 17}
    preferred_task_duration: int = 30  # in minutes
    break_duration: int = 5  # in minutes
    notification_preferences: Dict[str, bool] = {
        "email": True,
        "push": True,
        "in_app": True
    }
    theme: str = "light"
    language: str = "en"
    timezone: str = "UTC"


class UserActivity(BaseModel):
    """User activity data for learning."""
    id: Optional[str] = None
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    activity_type: str  # e.g., "task_completed", "goal_updated"
    activity_data: Dict[str, Any]
    context: Dict[str, Any] = {}  # time of day, device, location, etc.


class LearningFeedback(BaseModel):
    """Feedback data for model improvement."""
    id: Optional[str] = None
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    recommendation_id: str
    feedback_type: str  # "accepted", "rejected", "modified"
    original_recommendation: Dict[str, Any]
    modified_recommendation: Optional[Dict[str, Any]] = None
    user_comments: Optional[str] = None
    context: Dict[str, Any] = {}


class Recommendation(BaseModel):
    """Recommendation data model."""
    id: Optional[str] = None
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    recommendation_type: str  # "task_scheduling", "goal_suggestion", etc.
    content: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0
    explanation: Optional[str] = None
    is_applied: bool = False
    feedback: Optional[LearningFeedback] = None