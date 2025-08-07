#!/usr/bin/env python3
"""
AstroAssistance Scheduled Training
---------------------------------
This script sets up scheduled training for AstroAssistance's AI models.
It can be run as a cron job or as a systemd timer.

Author: Senior AI Engineer
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scheduled_training.log')
    ]
)
logger = logging.getLogger('AstroAssistance-Scheduler')

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def run_training():
    """Run the training script."""
    logger.info("Starting scheduled training")
    
    # Path to the training script
    training_script = PROJECT_ROOT / "train_models.py"
    
    # Path to the Python executable in the virtual environment
    if os.name == 'nt':  # Windows
        python_executable = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:  # Unix/Linux/Mac
        python_executable = PROJECT_ROOT / "venv" / "bin" / "python"
    
    # Command to run
    cmd = [str(python_executable), str(training_script), "--generate-recommendations"]
    
    # Log file for the training output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"training_{timestamp}.log"
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Logging output to: {log_file}")
    
    try:
        # Run the training script and capture output
        with open(log_file, 'w') as f:
            process = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        
        logger.info("Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error code: {e.returncode}")
        return False

def setup_cron_job():
    """Set up a cron job for scheduled training."""
    logger.info("Setting up cron job for scheduled training")
    
    # Path to the current script
    script_path = os.path.abspath(__file__)
    
    # Cron expression for daily training at 2 AM
    cron_expression = "0 2 * * *"
    
    # Command to add to crontab
    cron_cmd = f"{cron_expression} {sys.executable} {script_path} --run-training"
    
    logger.info(f"Cron command: {cron_cmd}")
    
    # Check if the cron job already exists
    try:
        existing_crontab = subprocess.check_output(["crontab", "-l"], universal_newlines=True)
    except subprocess.CalledProcessError:
        existing_crontab = ""
    
    if cron_cmd in existing_crontab:
        logger.info("Cron job already exists")
        return True
    
    # Add the new cron job
    new_crontab = existing_crontab + f"\n{cron_cmd}\n"
    
    try:
        process = subprocess.run(["crontab", "-"], input=new_crontab, universal_newlines=True, check=True)
        logger.info("Cron job added successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add cron job: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AstroAssistance Scheduled Training")
    parser.add_argument("--run-training", action="store_true", help="Run the training process")
    parser.add_argument("--setup-cron", action="store_true", help="Set up a cron job for scheduled training")
    args = parser.parse_args()
    
    if args.run_training:
        run_training()
    elif args.setup_cron:
        setup_cron_job()
    else:
        # If no arguments provided, run the training
        run_training()

if __name__ == "__main__":
    main()