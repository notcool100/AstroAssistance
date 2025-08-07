#!/usr/bin/env python3
"""
AstroAssistance Production Deployment Script
-------------------------------------------
This script handles the deployment of the AstroAssistance system,
including environment setup, dependency management, and service startup.

Author: Senior AI Engineer
"""
import os
import sys
import subprocess
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger('AstroAssistance-Deployment')

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PATH = PROJECT_ROOT / "venv"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, DATA_DIR / "raw", DATA_DIR / "processed", 
                 DATA_DIR / "synthetic", MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

def run_command(command, shell=False, env=None):
    """Run a shell command and log output."""
    logger.info(f"Running command: {command}")
    try:
        if isinstance(command, list) and not shell:
            result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=env)
        logger.info(f"Command output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None

def setup_environment():
    """Set up the Python virtual environment."""
    logger.info("Setting up virtual environment...")
    
    # Check if venv exists
    if not VENV_PATH.exists():
        run_command([sys.executable, "-m", "venv", str(VENV_PATH)])
    
    # Determine the Python executable in the virtual environment
    if os.name == 'nt':  # Windows
        python_executable = VENV_PATH / "Scripts" / "python.exe"
        pip_executable = VENV_PATH / "Scripts" / "pip.exe"
    else:  # Unix/Linux/Mac
        python_executable = VENV_PATH / "bin" / "python"
        pip_executable = VENV_PATH / "bin" / "pip"
    
    # Upgrade pip
    run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])
    
    return python_executable, pip_executable

def install_core_dependencies(pip_executable):
    """Install core dependencies required for the application."""
    logger.info("Installing core dependencies...")
    
    # Core dependencies that are essential for the application
    core_deps = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-jose",
        "PyJWT",
        "PyYAML",
        "loguru",
        "numpy",
        "pandas",
        "scikit-learn",
        "requests",
        "python-dotenv",
        "sqlalchemy",
        "psycopg2-binary"
    ]
    
    # Set environment variables for pip
    env = os.environ.copy()
    env["TMPDIR"] = "/tmp/pip_build_dir"
    
    # Install core dependencies
    run_command([str(pip_executable), "install"] + core_deps, env=env)
    
    return True

def generate_sample_data():
    """Generate sample data for the application."""
    logger.info("Generating sample data...")
    
    # Sample tasks data
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
    with open(DATA_DIR / "synthetic" / "tasks.json", "w") as f:
        json.dump(sample_tasks, f, indent=2)
    
    logger.info(f"Sample data saved to {DATA_DIR / 'synthetic' / 'tasks.json'}")
    return True

def start_api_server(python_executable):
    """Start the API server."""
    logger.info("Starting API server...")
    
    # Use the minimal_api.py script we created earlier
    api_script = PROJECT_ROOT / "minimal_api.py"
    
    # Start the server
    cmd = f"{python_executable} {api_script}"
    
    # In a production environment, you would use a process manager like supervisord
    # or systemd to manage the API server process
    logger.info(f"API server command: {cmd}")
    logger.info("In production, use a process manager like supervisord or systemd")
    logger.info("For now, we'll start the server in the foreground")
    
    # Execute the command directly (this will block)
    os.system(cmd)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="AstroAssistance Deployment Script")
    parser.add_argument("--setup-only", action="store_true", help="Only set up the environment without starting the server")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample data")
    args = parser.parse_args()
    
    logger.info("Starting AstroAssistance deployment...")
    
    # Setup environment
    python_executable, pip_executable = setup_environment()
    logger.info(f"Using Python: {python_executable}")
    
    # Install dependencies
    install_core_dependencies(pip_executable)
    
    # Generate sample data if requested or if it doesn't exist
    if args.generate_data or not (DATA_DIR / "synthetic" / "tasks.json").exists():
        generate_sample_data()
    
    # Start API server if not in setup-only mode
    if not args.setup_only:
        start_api_server(python_executable)
    
    logger.info("Deployment completed successfully")

if __name__ == "__main__":
    main()