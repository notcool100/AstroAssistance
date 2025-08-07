#!/usr/bin/env python3
"""
AstroAssistance API Test Client
------------------------------
This script tests the AstroAssistance API by making requests to various endpoints
and displaying the results.

Author: Senior AI Engineer
"""
import requests
import json
import sys
from datetime import datetime, timedelta
from tabulate import tabulate
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AstroAssistance-Client')

# API base URL
BASE_URL = "http://localhost:8000"

def get_token():
    """Get authentication token."""
    try:
        response = requests.post(f"{BASE_URL}/api/token")
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get token: {e}")
        sys.exit(1)

def get_tasks(token):
    """Get all tasks."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{BASE_URL}/api/tasks", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get tasks: {e}")
        return []

def create_task(token, task_data):
    """Create a new task."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(f"{BASE_URL}/api/tasks", json=task_data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create task: {e}")
        return None

def get_recommendations(token):
    """Get recommendations."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{BASE_URL}/api/recommendations", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get recommendations: {e}")
        return []

def display_tasks(tasks):
    """Display tasks in a tabular format."""
    if not tasks:
        print("No tasks found.")
        return
    
    # Prepare data for tabulate
    headers = ["ID", "Title", "Priority", "Status", "Due Date", "Est. Duration"]
    rows = []
    
    for task in tasks:
        due_date = task.get("due_date", "N/A")
        if due_date and due_date != "N/A":
            # Convert ISO format to more readable format
            try:
                due_date = datetime.fromisoformat(due_date).strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass
        
        rows.append([
            task.get("id", "N/A"),
            task.get("title", "N/A"),
            task.get("priority", "N/A"),
            task.get("status", "N/A"),
            due_date,
            f"{task.get('estimated_duration', 'N/A')} min" if task.get('estimated_duration') else "N/A"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def display_recommendations(recommendations):
    """Display recommendations in a tabular format."""
    if not recommendations:
        print("No recommendations found.")
        return
    
    # Prepare data for tabulate
    headers = ["ID", "Title", "Type", "Priority", "Description"]
    rows = []
    
    for rec in recommendations:
        rows.append([
            rec.get("id", "N/A"),
            rec.get("title", "N/A"),
            rec.get("type", "N/A"),
            rec.get("priority", "N/A"),
            rec.get("description", "N/A")
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AstroAssistance API Test Client")
    parser.add_argument("--create-task", action="store_true", help="Create a new task")
    parser.add_argument("--title", help="Title for the new task")
    parser.add_argument("--description", help="Description for the new task")
    parser.add_argument("--priority", choices=["LOW", "MEDIUM", "HIGH", "URGENT"], default="MEDIUM", help="Priority for the new task")
    parser.add_argument("--category", choices=["WORK", "PERSONAL", "HEALTH", "EDUCATION", "FINANCE", "SOCIAL", "OTHER"], default="WORK", help="Category for the new task")
    parser.add_argument("--duration", type=int, help="Estimated duration in minutes for the new task")
    args = parser.parse_args()
    
    print("🚀 AstroAssistance API Test Client")
    print("----------------------------------")
    
    # Get token
    token = get_token()
    logger.info(f"Authentication successful")
    
    # Create a task if requested
    if args.create_task:
        if not args.title:
            print("Error: --title is required when creating a task")
            sys.exit(1)
        
        new_task = {
            "title": args.title,
            "description": args.description or f"Task created on {datetime.now().strftime('%Y-%m-%d')}",
            "category": args.category,
            "priority": args.priority,
            "due_date": (datetime.now() + timedelta(days=1)).isoformat(),
            "estimated_duration": args.duration or 60,
            "tags": ["api", "testing"]
        }
        
        created_task = create_task(token, new_task)
        if created_task:
            print("\n✅ Task created successfully:")
            display_tasks([created_task])
    
    # Get and display tasks
    print("\n📋 Current Tasks:")
    tasks = get_tasks(token)
    display_tasks(tasks)
    
    # Get and display recommendations
    print("\n💡 Recommendations:")
    recommendations = get_recommendations(token)
    display_recommendations(recommendations)
    
    print("\n🔗 API Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)