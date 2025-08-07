"""
Simple script to run AstroAssistance API server with minimal dependencies.
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create necessary directories
os.makedirs(os.path.join(project_root, "data", "raw"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "synthetic"), exist_ok=True)
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

# Create sample data
sample_tasks = [
    {
        "id": "task1",
        "title": "Complete project proposal",
        "description": "Write a detailed proposal for the new project",
        "category": "WORK",
        "priority": "HIGH",
        "status": "IN_PROGRESS",
        "due_date": (datetime.now().isoformat()),
        "created_at": (datetime.now().isoformat()),
        "updated_at": (datetime.now().isoformat()),
        "estimated_duration": 120,
        "tags": ["project", "proposal", "deadline"],
        "user_id": "user123"
    },
    {
        "id": "task2",
        "title": "Schedule doctor appointment",
        "description": "Call the clinic to schedule annual checkup",
        "category": "PERSONAL",
        "priority": "MEDIUM",
        "status": "NOT_STARTED",
        "due_date": (datetime.now().isoformat()),
        "created_at": (datetime.now().isoformat()),
        "updated_at": (datetime.now().isoformat()),
        "estimated_duration": 15,
        "tags": ["health", "appointment"],
        "user_id": "user123"
    },
    {
        "id": "task3",
        "title": "Prepare presentation",
        "description": "Create slides for the team meeting",
        "category": "WORK",
        "priority": "HIGH",
        "status": "NOT_STARTED",
        "due_date": (datetime.now().isoformat()),
        "created_at": (datetime.now().isoformat()),
        "updated_at": (datetime.now().isoformat()),
        "estimated_duration": 90,
        "tags": ["presentation", "meeting"],
        "user_id": "user123"
    }
]

# Save sample data
with open(os.path.join(project_root, "data", "synthetic", "tasks.json"), "w") as f:
    json.dump(sample_tasks, f, indent=2)

print("Created sample data")

# Start API server
try:
    from src.api.main import app
    import uvicorn
    
    print("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
except ImportError as e:
    print(f"Error importing API modules: {e}")
    print("Please make sure you have installed the required dependencies:")
    print("pip install fastapi uvicorn pydantic")