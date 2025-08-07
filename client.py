"""
Simple client to interact with AstroAssistance API.
"""
import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def get_token():
    """Get authentication token."""
    response = requests.post(f"{BASE_URL}/api/token")
    return response.json()["access_token"]

def get_tasks(token):
    """Get all tasks."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/tasks", headers=headers)
    return response.json()

def create_task(token, task_data):
    """Create a new task."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/api/tasks", json=task_data, headers=headers)
    return response.json()

def get_recommendations(token):
    """Get recommendations."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/recommendations", headers=headers)
    return response.json()

def main():
    """Main function."""
    # Get token
    token = get_token()
    print(f"Got token: {token}")
    
    # Get tasks
    tasks = get_tasks(token)
    print("\nCurrent tasks:")
    for task in tasks:
        print(f"- {task['title']} (Priority: {task['priority']}, Status: {task['status']})")
    
    # Create a new task
    new_task = {
        "title": "Test the AstroAssistance API",
        "description": "Create a simple client to test the API functionality",
        "category": "WORK",
        "priority": "HIGH",
        "due_date": (datetime.now() + timedelta(days=1)).isoformat(),
        "estimated_duration": 60,
        "tags": ["api", "testing"]
    }
    
    created_task = create_task(token, new_task)
    print(f"\nCreated new task: {created_task['title']}")
    
    # Get recommendations
    recommendations = get_recommendations(token)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec['title']}: {rec['description']}")

if __name__ == "__main__":
    main()