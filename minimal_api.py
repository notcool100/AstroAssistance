"""
Minimal API server for AstroAssistance.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="AstroAssistance API",
    description="Minimal API for AstroAssistance, a self-learning productivity assistant",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Enums
class TaskPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

class TaskStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    DEFERRED = "DEFERRED"
    CANCELLED = "CANCELLED"

class TaskCategory(str, Enum):
    WORK = "WORK"
    PERSONAL = "PERSONAL"
    HEALTH = "HEALTH"
    EDUCATION = "EDUCATION"
    FINANCE = "FINANCE"
    SOCIAL = "SOCIAL"
    OTHER = "OTHER"

# Models
class Task(BaseModel):
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
    user_id: str

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    category: TaskCategory = TaskCategory.OTHER
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    estimated_duration: Optional[int] = None
    tags: List[str] = []

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[TaskCategory] = None
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[datetime] = None
    estimated_duration: Optional[int] = None
    actual_duration: Optional[int] = None
    tags: Optional[List[str]] = None

class Recommendation(BaseModel):
    id: str
    title: str
    description: str
    type: str
    priority: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    user_id: str
    metadata: Dict[str, Any] = {}

# Mock data
TASKS = []
RECOMMENDATIONS = []

# Load sample data
try:
    sample_data_path = os.path.join("data", "synthetic", "tasks.json")
    if os.path.exists(sample_data_path):
        with open(sample_data_path, "r") as f:
            task_data = json.load(f)
            for task in task_data:
                # Convert string dates to datetime
                for date_field in ["due_date", "created_at", "updated_at", "completed_at"]:
                    if task.get(date_field):
                        task[date_field] = datetime.fromisoformat(task[date_field])
                TASKS.append(Task(**task))
except Exception as e:
    print(f"Error loading sample data: {e}")

# Generate sample recommendations
for i in range(3):
    recommendation = Recommendation(
        id=f"rec{i+1}",
        title=f"Sample Recommendation {i+1}",
        description=f"This is a sample recommendation {i+1}",
        type="task_prioritization",
        priority=3-i,
        created_at=datetime.now(),
        expires_at=None,
        user_id="user123",
        metadata={"source": "sample"}
    )
    RECOMMENDATIONS.append(recommendation)

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AstroAssistance API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/tasks", response_model=List[Task])
async def get_tasks(token: str = Depends(oauth2_scheme)):
    """Get all tasks."""
    return TASKS

@app.post("/api/tasks", response_model=Task, status_code=status.HTTP_201_CREATED)
async def create_task(task: TaskCreate, token: str = Depends(oauth2_scheme)):
    """Create a new task."""
    task_id = f"task{len(TASKS) + 1}"
    now = datetime.now()
    
    new_task = Task(
        id=task_id,
        title=task.title,
        description=task.description,
        category=task.category,
        priority=task.priority,
        status=TaskStatus.NOT_STARTED,
        due_date=task.due_date,
        created_at=now,
        updated_at=now,
        estimated_duration=task.estimated_duration,
        tags=task.tags,
        user_id="user123"  # Mock user ID
    )
    
    TASKS.append(new_task)
    return new_task

@app.get("/api/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str, token: str = Depends(oauth2_scheme)):
    """Get a task by ID."""
    for task in TASKS:
        if task.id == task_id:
            return task
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.put("/api/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, task_update: TaskUpdate, token: str = Depends(oauth2_scheme)):
    """Update a task."""
    for i, task in enumerate(TASKS):
        if task.id == task_id:
            # Update task fields
            update_data = task_update.dict(exclude_unset=True)
            
            # Handle completion status
            if update_data.get("status") == TaskStatus.COMPLETED and task.status != TaskStatus.COMPLETED:
                update_data["completed_at"] = datetime.now()
            
            # Update the task
            updated_task = task.copy(update=update_data)
            updated_task.updated_at = datetime.now()
            TASKS[i] = updated_task
            
            return updated_task
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: str, token: str = Depends(oauth2_scheme)):
    """Delete a task."""
    for i, task in enumerate(TASKS):
        if task.id == task_id:
            TASKS.pop(i)
            return
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/api/recommendations", response_model=List[Recommendation])
async def get_recommendations(token: str = Depends(oauth2_scheme)):
    """Get recommendations."""
    return RECOMMENDATIONS

@app.post("/api/token")
async def get_token():
    """Get a mock token for testing."""
    return {
        "access_token": "mock_token_for_testing",
        "token_type": "bearer"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)