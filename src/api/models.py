"""
API data models for AstroAssistance.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.core.data_types import (
    TaskPriority, TaskStatus, TaskCategory, RecurrencePattern
)


# API Request Models
class CreateTaskRequest(BaseModel):
    """Request model for creating a task."""
    title: str
    description: Optional[str] = None
    category: TaskCategory = TaskCategory.OTHER
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # in minutes
    tags: List[str] = []
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_end_date: Optional[datetime] = None
    parent_task_id: Optional[str] = None


class UpdateTaskRequest(BaseModel):
    """Request model for updating a task."""
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[TaskCategory] = None
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[datetime] = None
    estimated_duration: Optional[int] = None
    actual_duration: Optional[int] = None
    tags: Optional[List[str]] = None
    recurrence: Optional[RecurrencePattern] = None
    recurrence_end_date: Optional[datetime] = None
    notes: Optional[str] = None


class CreateReminderRequest(BaseModel):
    """Request model for creating a reminder."""
    task_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    reminder_time: datetime
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_end_date: Optional[datetime] = None


class UpdateReminderRequest(BaseModel):
    """Request model for updating a reminder."""
    task_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    reminder_time: Optional[datetime] = None
    is_completed: Optional[bool] = None
    recurrence: Optional[RecurrencePattern] = None
    recurrence_end_date: Optional[datetime] = None


class CreateGoalRequest(BaseModel):
    """Request model for creating a goal."""
    title: str
    description: Optional[str] = None
    category: TaskCategory = TaskCategory.OTHER
    target_date: Optional[datetime] = None
    related_tasks: List[str] = []
    milestones: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    notes: Optional[str] = None


class UpdateGoalRequest(BaseModel):
    """Request model for updating a goal."""
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[TaskCategory] = None
    target_date: Optional[datetime] = None
    status: Optional[TaskStatus] = None
    progress: Optional[float] = None
    related_tasks: Optional[List[str]] = None
    milestones: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class UpdateUserPreferenceRequest(BaseModel):
    """Request model for updating user preferences."""
    productivity_peak_hours: Optional[List[int]] = None
    work_days: Optional[List[int]] = None
    work_hours: Optional[Dict[str, Any]] = None
    preferred_task_duration: Optional[int] = None
    break_duration: Optional[int] = None
    notification_preferences: Optional[Dict[str, bool]] = None
    theme: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for providing feedback on a recommendation."""
    recommendation_id: str
    feedback_type: str  # "accepted", "rejected", "modified"
    modified_recommendation: Optional[Dict[str, Any]] = None
    user_comments: Optional[str] = None


# API Response Models
class TaskResponse(BaseModel):
    """Response model for a task."""
    id: str
    title: str
    description: Optional[str] = None
    category: TaskCategory
    priority: TaskPriority
    status: TaskStatus
    due_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None
    actual_duration: Optional[int] = None
    tags: List[str] = []
    recurrence: RecurrencePattern
    recurrence_end_date: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = []
    notes: Optional[str] = None
    attachments: List[str] = []
    user_id: str


class ReminderResponse(BaseModel):
    """Response model for a reminder."""
    id: str
    task_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    reminder_time: datetime
    created_at: datetime
    is_completed: bool
    recurrence: RecurrencePattern
    recurrence_end_date: Optional[datetime] = None
    user_id: str


class GoalResponse(BaseModel):
    """Response model for a goal."""
    id: str
    title: str
    description: Optional[str] = None
    category: TaskCategory
    start_date: datetime
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus
    progress: float
    related_tasks: List[str] = []
    milestones: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    notes: Optional[str] = None
    user_id: str


class UserPreferenceResponse(BaseModel):
    """Response model for user preferences."""
    id: str
    user_id: str
    productivity_peak_hours: List[int] = []
    work_days: List[int] = []
    work_hours: Dict[str, Any] = {}
    preferred_task_duration: int
    break_duration: int
    notification_preferences: Dict[str, bool] = {}
    theme: str
    language: str
    timezone: str


class RecommendationResponse(BaseModel):
    """Response model for a recommendation."""
    id: str
    user_id: str
    timestamp: datetime
    recommendation_type: str
    content: Dict[str, Any]
    confidence_score: float
    explanation: Optional[str] = None
    is_applied: bool


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    details: Optional[Dict[str, Any]] = None


# Pagination Models
class PaginatedResponse(BaseModel):
    """Base model for paginated responses."""
    total: int
    page: int
    page_size: int
    total_pages: int


class PaginatedTasksResponse(PaginatedResponse):
    """Paginated response for tasks."""
    items: List[TaskResponse]


class PaginatedRemindersResponse(PaginatedResponse):
    """Paginated response for reminders."""
    items: List[ReminderResponse]


class PaginatedGoalsResponse(PaginatedResponse):
    """Paginated response for goals."""
    items: List[GoalResponse]


class PaginatedRecommendationsResponse(PaginatedResponse):
    """Paginated response for recommendations."""
    items: List[RecommendationResponse]