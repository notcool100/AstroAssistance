"""
Main entry point for AstroAssistance.
"""
import os
import argparse
import sys
from pathlib import Path

from src.core.config import config_manager
from src.core.logger import app_logger
from src.data_processing.data_generator import DataGenerator
from src.data_processing.data_processor import DataProcessor
from src.training.train_models import (
    train_task_completion_model,
    train_recommendation_model,
    train_reinforcement_learning_model
)


def setup_environment():
    """Set up the environment."""
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Create necessary directories
    os.makedirs(os.path.join(project_root, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "data", "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)


def generate_data(num_users: int = 50):
    """
    Generate synthetic data.
    
    Args:
        num_users: Number of users to generate
    """
    app_logger.info(f"Generating synthetic data for {num_users} users")
    
    # Generate data
    generator = DataGenerator(seed=42)
    file_paths = generator.generate_all_data(num_users=num_users)
    
    app_logger.info(f"Generated data files: {file_paths}")


def process_data():
    """Process data for model training."""
    app_logger.info("Processing data for model training")
    
    # Process data
    processor = DataProcessor()
    dataset_paths = processor.prepare_all_datasets()
    
    app_logger.info(f"Prepared datasets: {dataset_paths}")


def train_models(model_type: str = "all", use_wandb: bool = False, use_mlflow: bool = False):
    """
    Train models.
    
    Args:
        model_type: Type of model to train ("all", "task", "recommendation", "rl")
        use_wandb: Whether to use Weights & Biases for tracking
        use_mlflow: Whether to use MLflow for tracking
    """
    app_logger.info(f"Training models: {model_type}")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Train models
    if model_type in ["all", "task"]:
        app_logger.info("Training task completion model")
        task_model = train_task_completion_model(data_processor, use_wandb=use_wandb)
    
    if model_type in ["all", "recommendation"]:
        app_logger.info("Training recommendation model")
        rec_model = train_recommendation_model(data_processor, use_wandb=use_wandb)
    
    if model_type in ["all", "rl"]:
        app_logger.info("Training reinforcement learning model")
        rl_model = train_reinforcement_learning_model(use_wandb=use_wandb)
    
    app_logger.info("Model training complete")


def start_api():
    """Start the API server."""
    app_logger.info("Starting API server")
    
    # Import here to avoid circular imports
    from src.api.main import app
    import uvicorn
    
    # Get API configuration
    host = config_manager.get("api.host", "0.0.0.0")
    port = config_manager.get("api.port", 8000)
    debug = config_manager.get("api.debug", True)
    
    # Start server
    uvicorn.run(app, host=host, port=port, reload=debug)


def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="AstroAssistance - Self-learning AI productivity assistant")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate data command
    generate_parser = subparsers.add_parser("generate-data", help="Generate synthetic data")
    generate_parser.add_argument("--num-users", type=int, default=50, help="Number of users to generate")
    
    # Process data command
    process_parser = subparsers.add_parser("process-data", help="Process data for model training")
    
    # Train models command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--model", type=str, choices=["all", "task", "recommendation", "rl"], default="all",
                             help="Model to train (default: all)")
    train_parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for tracking")
    train_parser.add_argument("--mlflow", action="store_true", help="Use MLflow for tracking")
    
    # Start API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Execute command
    if args.command == "generate-data":
        generate_data(num_users=args.num_users)
    elif args.command == "process-data":
        process_data()
    elif args.command == "train":
        train_models(model_type=args.model, use_wandb=args.wandb, use_mlflow=args.mlflow)
    elif args.command == "api":
        start_api()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()