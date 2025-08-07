"""
API routes for AstroAssistance.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from fastapi.security import OAuth2PasswordBearer

from src.api.models import (
    CreateTaskRequest, UpdateTaskRequest, TaskResponse,
    CreateReminderRequest, UpdateReminderRequest, ReminderResponse,
    CreateGoalRequest, UpdateGoalRequest, GoalResponse,
    UpdateUserPreferenceRequest, UserPreferenceResponse,
    FeedbackRequest, RecommendationResponse,
    PaginatedTasksResponse, PaginatedRemindersResponse,
    PaginatedGoalsResponse, PaginatedRecommendationsResponse,
    ErrorResponse
)
from src.core.data_types import (
    Task, Reminder, Goal, UserPreference, LearningFeedback, Recommendation
)
from src.core.logger import app_logger
from src.api.services import (
    TaskService, ReminderService, GoalService, 
    UserPreferenceService, RecommendationService
)


# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Create API routers
tasks_router = APIRouter(prefix="/tasks", tags=["Tasks"])
reminders_router = APIRouter(prefix="/reminders", tags=["Reminders"])
goals_router = APIRouter(prefix="/goals", tags=["Goals"])
preferences_router = APIRouter(prefix="/preferences", tags=["User Preferences"])
recommendations_router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


# Dependency to get current user ID from token
async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    """
    Get the current user ID from the authentication token.
    
    Args:
        token: Authentication token
        
    Returns:
        User ID
    """
    # In a real implementation, this would validate the token and extract the user ID
    # For now, we'll just return a placeholder user ID
    return "user123"


# Task routes
@tasks_router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_request: CreateTaskRequest,
    user_id: str = Depends(get_current_user_id)
) -> TaskResponse:
    """
    Create a new task.
    
    Args:
        task_request: Task creation request
        user_id: Current user ID
        
    Returns:
        Created task
    """
    try:
        # Create task
        task_service = TaskService()
        task = task_service.create_task(user_id, task_request)
        
        # Convert to response model
        return TaskResponse(**task.dict())
    except Exception as e:
        app_logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating task: {str(e)}"
        )


@tasks_router.get("/", response_model=PaginatedTasksResponse)
async def get_tasks(
    status: Optional[str] = Query(None, description="Filter by task status"),
    category: Optional[str] = Query(None, description="Filter by task category"),
    priority: Optional[str] = Query(None, description="Filter by task priority"),
    due_date_start: Optional[datetime] = Query(None, description="Filter by due date (start)"),
    due_date_end: Optional[datetime] = Query(None, description="Filter by due date (end)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    user_id: str = Depends(get_current_user_id)
) -> PaginatedTasksResponse:
    """
    Get tasks with pagination and filtering.
    
    Args:
        status: Filter by task status
        category: Filter by task category
        priority: Filter by task priority
        due_date_start: Filter by due date (start)
        due_date_end: Filter by due date (end)
        page: Page number
        page_size: Page size
        user_id: Current user ID
        
    Returns:
        Paginated tasks
    """
    try:
        # Create filters
        filters = {
            "status": status,
            "category": category,
            "priority": priority,
            "due_date_start": due_date_start,
            "due_date_end": due_date_end
        }
        
        # Get tasks
        task_service = TaskService()
        tasks, total = task_service.get_tasks(user_id, filters, page, page_size)
        
        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size
        
        # Convert to response models
        task_responses = [TaskResponse(**task.dict()) for task in tasks]
        
        return PaginatedTasksResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=task_responses
        )
    except Exception as e:
        app_logger.error(f"Error getting tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting tasks: {str(e)}"
        )


@tasks_router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str = Path(..., description="Task ID"),
    user_id: str = Depends(get_current_user_id)
) -> TaskResponse:
    """
    Get a task by ID.
    
    Args:
        task_id: Task ID
        user_id: Current user ID
        
    Returns:
        Task
    """
    try:
        # Get task
        task_service = TaskService()
        task = task_service.get_task(user_id, task_id)
        
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        
        # Convert to response model
        return TaskResponse(**task.dict())
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task: {str(e)}"
        )


@tasks_router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_request: UpdateTaskRequest,
    task_id: str = Path(..., description="Task ID"),
    user_id: str = Depends(get_current_user_id)
) -> TaskResponse:
    """
    Update a task.
    
    Args:
        task_request: Task update request
        task_id: Task ID
        user_id: Current user ID
        
    Returns:
        Updated task
    """
    try:
        # Update task
        task_service = TaskService()
        task = task_service.update_task(user_id, task_id, task_request)
        
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        
        # Convert to response model
        return TaskResponse(**task.dict())
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating task: {str(e)}"
        )


@tasks_router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str = Path(..., description="Task ID"),
    user_id: str = Depends(get_current_user_id)
) -> None:
    """
    Delete a task.
    
    Args:
        task_id: Task ID
        user_id: Current user ID
    """
    try:
        # Delete task
        task_service = TaskService()
        success = task_service.delete_task(user_id, task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting task: {str(e)}"
        )


# Reminder routes
@reminders_router.post("/", response_model=ReminderResponse, status_code=status.HTTP_201_CREATED)
async def create_reminder(
    reminder_request: CreateReminderRequest,
    user_id: str = Depends(get_current_user_id)
) -> ReminderResponse:
    """
    Create a new reminder.
    
    Args:
        reminder_request: Reminder creation request
        user_id: Current user ID
        
    Returns:
        Created reminder
    """
    try:
        # Create reminder
        reminder_service = ReminderService()
        reminder = reminder_service.create_reminder(user_id, reminder_request)
        
        # Convert to response model
        return ReminderResponse(**reminder.dict())
    except Exception as e:
        app_logger.error(f"Error creating reminder: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating reminder: {str(e)}"
        )


@reminders_router.get("/", response_model=PaginatedRemindersResponse)
async def get_reminders(
    task_id: Optional[str] = Query(None, description="Filter by task ID"),
    is_completed: Optional[bool] = Query(None, description="Filter by completion status"),
    reminder_time_start: Optional[datetime] = Query(None, description="Filter by reminder time (start)"),
    reminder_time_end: Optional[datetime] = Query(None, description="Filter by reminder time (end)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    user_id: str = Depends(get_current_user_id)
) -> PaginatedRemindersResponse:
    """
    Get reminders with pagination and filtering.
    
    Args:
        task_id: Filter by task ID
        is_completed: Filter by completion status
        reminder_time_start: Filter by reminder time (start)
        reminder_time_end: Filter by reminder time (end)
        page: Page number
        page_size: Page size
        user_id: Current user ID
        
    Returns:
        Paginated reminders
    """
    try:
        # Create filters
        filters = {
            "task_id": task_id,
            "is_completed": is_completed,
            "reminder_time_start": reminder_time_start,
            "reminder_time_end": reminder_time_end
        }
        
        # Get reminders
        reminder_service = ReminderService()
        reminders, total = reminder_service.get_reminders(user_id, filters, page, page_size)
        
        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size
        
        # Convert to response models
        reminder_responses = [ReminderResponse(**reminder.dict()) for reminder in reminders]
        
        return PaginatedRemindersResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=reminder_responses
        )
    except Exception as e:
        app_logger.error(f"Error getting reminders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting reminders: {str(e)}"
        )


@reminders_router.get("/{reminder_id}", response_model=ReminderResponse)
async def get_reminder(
    reminder_id: str = Path(..., description="Reminder ID"),
    user_id: str = Depends(get_current_user_id)
) -> ReminderResponse:
    """
    Get a reminder by ID.
    
    Args:
        reminder_id: Reminder ID
        user_id: Current user ID
        
    Returns:
        Reminder
    """
    try:
        # Get reminder
        reminder_service = ReminderService()
        reminder = reminder_service.get_reminder(user_id, reminder_id)
        
        if reminder is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder with ID {reminder_id} not found"
            )
        
        # Convert to response model
        return ReminderResponse(**reminder.dict())
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting reminder: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting reminder: {str(e)}"
        )


@reminders_router.put("/{reminder_id}", response_model=ReminderResponse)
async def update_reminder(
    reminder_request: UpdateReminderRequest,
    reminder_id: str = Path(..., description="Reminder ID"),
    user_id: str = Depends(get_current_user_id)
) -> ReminderResponse:
    """
    Update a reminder.
    
    Args:
        reminder_request: Reminder update request
        reminder_id: Reminder ID
        user_id: Current user ID
        
    Returns:
        Updated reminder
    """
    try:
        # Update reminder
        reminder_service = ReminderService()
        reminder = reminder_service.update_reminder(user_id, reminder_id, reminder_request)
        
        if reminder is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder with ID {reminder_id} not found"
            )
        
        # Convert to response model
        return ReminderResponse(**reminder.dict())
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating reminder: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating reminder: {str(e)}"
        )


@reminders_router.delete("/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_reminder(
    reminder_id: str = Path(..., description="Reminder ID"),
    user_id: str = Depends(get_current_user_id)
) -> None:
    """
    Delete a reminder.
    
    Args:
        reminder_id: Reminder ID
        user_id: Current user ID
    """
    try:
        # Delete reminder
        reminder_service = ReminderService()
        success = reminder_service.delete_reminder(user_id, reminder_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder with ID {reminder_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting reminder: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting reminder: {str(e)}"
        )


# Goal routes
@goals_router.post("/", response_model=GoalResponse, status_code=status.HTTP_201_CREATED)
async def create_goal(
    goal_request: CreateGoalRequest,
    user_id: str = Depends(get_current_user_id)
) -> GoalResponse:
    """
    Create a new goal.
    
    Args:
        goal_request: Goal creation request
        user_id: Current user ID
        
    Returns:
        Created goal
    """
    try:
        # Create goal
        goal_service = GoalService()
        goal = goal_service.create_goal(user_id, goal_request)
        
        # Convert to response model
        return GoalResponse(**goal.dict())
    except Exception as e:
        app_logger.error(f"Error creating goal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating goal: {str(e)}"
        )


@goals_router.get("/", response_model=PaginatedGoalsResponse)
async def get_goals(
    status: Optional[str] = Query(None, description="Filter by goal status"),
    category: Optional[str] = Query(None, description="Filter by goal category"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    user_id: str = Depends(get_current_user_id)
) -> PaginatedGoalsResponse:
    """
    Get goals with pagination and filtering.
    
    Args:
        status: Filter by goal status
        category: Filter by goal category
        page: Page number
        page_size: Page size
        user_id: Current user ID
        
    Returns:
        Paginated goals
    """
    try:
        # Create filters
        filters = {
            "status": status,
            "category": category
        }
        
        # Get goals
        goal_service = GoalService()
        goals, total = goal_service.get_goals(user_id, filters, page, page_size)
        
        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size
        
        # Convert to response models
        goal_responses = [GoalResponse(**goal.dict()) for goal in goals]
        
        return PaginatedGoalsResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=goal_responses
        )
    except Exception as e:
        app_logger.error(f"Error getting goals: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting goals: {str(e)}"
        )


@goals_router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal(
    goal_id: str = Path(..., description="Goal ID"),
    user_id: str = Depends(get_current_user_id)
) -> GoalResponse:
    """
    Get a goal by ID.
    
    Args:
        goal_id: Goal ID
        user_id: Current user ID
        
    Returns:
        Goal
    """
    try:
        # Get goal
        goal_service = GoalService()
        goal = goal_service.get_goal(user_id, goal_id)
        
        if goal is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Goal with ID {goal_id} not found"
            )
        
        # Convert to response model
        return GoalResponse(**goal.dict())
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting goal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting goal: {str(e)}"
        )


@goals_router.put("/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_request: UpdateGoalRequest,
    goal_id: str = Path(..., description="Goal ID"),
    user_id: str = Depends(get_current_user_id)
) -> GoalResponse:
    """
    Update a goal.
    
    Args:
        goal_request: Goal update request
        goal_id: Goal ID
        user_id: Current user ID
        
    Returns:
        Updated goal
    """
    try:
        # Update goal
        goal_service = GoalService()
        goal = goal_service.update_goal(user_id, goal_id, goal_request)
        
        if goal is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Goal with ID {goal_id} not found"
            )
        
        # Convert to response model
        return GoalResponse(**goal.dict())
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating goal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating goal: {str(e)}"
        )


@goals_router.delete("/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_goal(
    goal_id: str = Path(..., description="Goal ID"),
    user_id: str = Depends(get_current_user_id)
) -> None:
    """
    Delete a goal.
    
    Args:
        goal_id: Goal ID
        user_id: Current user ID
    """
    try:
        # Delete goal
        goal_service = GoalService()
        success = goal_service.delete_goal(user_id, goal_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Goal with ID {goal_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting goal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting goal: {str(e)}"
        )


# User preference routes
@preferences_router.get("/", response_model=UserPreferenceResponse)
async def get_user_preferences(
    user_id: str = Depends(get_current_user_id)
) -> UserPreferenceResponse:
    """
    Get user preferences.
    
    Args:
        user_id: Current user ID
        
    Returns:
        User preferences
    """
    try:
        # Get user preferences
        preference_service = UserPreferenceService()
        preferences = preference_service.get_user_preferences(user_id)
        
        if preferences is None:
            # Create default preferences if none exist
            preferences = preference_service.create_default_preferences(user_id)
        
        # Convert to response model
        return UserPreferenceResponse(**preferences.dict())
    except Exception as e:
        app_logger.error(f"Error getting user preferences: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user preferences: {str(e)}"
        )


@preferences_router.put("/", response_model=UserPreferenceResponse)
async def update_user_preferences(
    preference_request: UpdateUserPreferenceRequest,
    user_id: str = Depends(get_current_user_id)
) -> UserPreferenceResponse:
    """
    Update user preferences.
    
    Args:
        preference_request: Preference update request
        user_id: Current user ID
        
    Returns:
        Updated user preferences
    """
    try:
        # Update user preferences
        preference_service = UserPreferenceService()
        preferences = preference_service.update_user_preferences(user_id, preference_request)
        
        # Convert to response model
        return UserPreferenceResponse(**preferences.dict())
    except Exception as e:
        app_logger.error(f"Error updating user preferences: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user preferences: {str(e)}"
        )


# Recommendation routes
@recommendations_router.get("/", response_model=PaginatedRecommendationsResponse)
async def get_recommendations(
    recommendation_type: Optional[str] = Query(None, description="Filter by recommendation type"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Filter by minimum confidence score"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    user_id: str = Depends(get_current_user_id)
) -> PaginatedRecommendationsResponse:
    """
    Get recommendations with pagination and filtering.
    
    Args:
        recommendation_type: Filter by recommendation type
        min_confidence: Filter by minimum confidence score
        page: Page number
        page_size: Page size
        user_id: Current user ID
        
    Returns:
        Paginated recommendations
    """
    try:
        # Create filters
        filters = {
            "recommendation_type": recommendation_type,
            "min_confidence": min_confidence
        }
        
        # Get recommendations
        recommendation_service = RecommendationService()
        recommendations, total = recommendation_service.get_recommendations(user_id, filters, page, page_size)
        
        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size
        
        # Convert to response models
        recommendation_responses = [RecommendationResponse(**rec.dict()) for rec in recommendations]
        
        return PaginatedRecommendationsResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=recommendation_responses
        )
    except Exception as e:
        app_logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )


@recommendations_router.get("/generate", response_model=List[RecommendationResponse])
async def generate_recommendations(
    count: int = Query(3, ge=1, le=10, description="Number of recommendations to generate"),
    user_id: str = Depends(get_current_user_id)
) -> List[RecommendationResponse]:
    """
    Generate new recommendations.
    
    Args:
        count: Number of recommendations to generate
        user_id: Current user ID
        
    Returns:
        List of generated recommendations
    """
    try:
        # Generate recommendations
        recommendation_service = RecommendationService()
        recommendations = recommendation_service.generate_recommendations(user_id, count)
        
        # Convert to response models
        recommendation_responses = [RecommendationResponse(**rec.dict()) for rec in recommendations]
        
        return recommendation_responses
    except Exception as e:
        app_logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )


@recommendations_router.post("/{recommendation_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def provide_feedback(
    feedback_request: FeedbackRequest,
    recommendation_id: str = Path(..., description="Recommendation ID"),
    user_id: str = Depends(get_current_user_id)
) -> None:
    """
    Provide feedback on a recommendation.
    
    Args:
        feedback_request: Feedback request
        recommendation_id: Recommendation ID
        user_id: Current user ID
    """
    try:
        # Check if recommendation ID matches
        if feedback_request.recommendation_id != recommendation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Recommendation ID in path does not match ID in request body"
            )
        
        # Process feedback
        recommendation_service = RecommendationService()
        success = recommendation_service.process_feedback(user_id, feedback_request)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation with ID {recommendation_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing feedback: {str(e)}"
        )