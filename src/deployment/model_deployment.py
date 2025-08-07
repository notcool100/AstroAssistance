"""
Model deployment utilities for AstroAssistance.
"""
import os
import json
import shutil
import time
import subprocess
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import threading

from src.core.config import config_manager
from src.core.logger import app_logger
from src.models.task_completion_model import TaskCompletionModel
from src.models.recommendation_model import RecommendationModel
from src.models.reinforcement_learning_model import ReinforcementLearningModel


class ModelDeployer:
    """Handles model deployment to production environments."""
    
    def __init__(self):
        """Initialize the model deployer."""
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.project_root, "models")
        self.deployment_dir = os.path.join(self.project_root, "deployment")
        self.archive_dir = os.path.join(self.deployment_dir, "archive")
        
        # Create directories if they don't exist
        os.makedirs(self.deployment_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Load deployment configuration
        self.deployment_config = config_manager.get("deployment", {})
        self.deployment_target = self.deployment_config.get("target", "local")
        self.auto_deploy = self.deployment_config.get("auto_deploy", False)
        self.deployment_interval = self.deployment_config.get("interval_hours", 24)
        
        # Initialize deployment tracking
        self.last_deployment_time = datetime.now()
        self.deployment_in_progress = False
        
        # Initialize background thread
        self.running = False
        self.background_thread = None
    
    def start_auto_deployment(self) -> None:
        """Start automatic deployment process."""
        if not self.auto_deploy:
            app_logger.info("Auto-deployment is disabled in configuration")
            return
        
        if self.running:
            app_logger.warning("Auto-deployment is already running")
            return
        
        # Start background thread
        self.running = True
        self.background_thread = threading.Thread(target=self._auto_deployment_process, daemon=True)
        self.background_thread.start()
        
        app_logger.info("Auto-deployment process started")
    
    def stop_auto_deployment(self) -> None:
        """Stop automatic deployment process."""
        if not self.running:
            app_logger.warning("Auto-deployment is not running")
            return
        
        # Stop background thread
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        app_logger.info("Auto-deployment process stopped")
    
    def deploy_models(self, models: List[str] = None, force: bool = False) -> bool:
        """
        Deploy models to the target environment.
        
        Args:
            models: List of model names to deploy (None for all)
            force: Whether to force deployment even if no changes are detected
            
        Returns:
            True if deployment was successful, False otherwise
        """
        if self.deployment_in_progress:
            app_logger.warning("Deployment already in progress, skipping")
            return False
        
        self.deployment_in_progress = True
        
        try:
            # Determine which models to deploy
            if models is None:
                models = ["task_completion_model", "recommendation_model", "reinforcement_learning_model"]
            
            app_logger.info(f"Deploying models: {models}")
            
            # Check if models have changed
            if not force and not self._have_models_changed(models):
                app_logger.info("No model changes detected, skipping deployment")
                self.deployment_in_progress = False
                return True
            
            # Deploy based on target
            if self.deployment_target == "local":
                success = self._deploy_local(models)
            elif self.deployment_target == "docker":
                success = self._deploy_docker(models)
            elif self.deployment_target == "kubernetes":
                success = self._deploy_kubernetes(models)
            elif self.deployment_target == "aws":
                success = self._deploy_aws(models)
            else:
                app_logger.error(f"Unknown deployment target: {self.deployment_target}")
                success = False
            
            # Update last deployment time
            if success:
                self.last_deployment_time = datetime.now()
                app_logger.info(f"Models deployed successfully to {self.deployment_target}")
            else:
                app_logger.error(f"Model deployment to {self.deployment_target} failed")
            
            self.deployment_in_progress = False
            return success
        
        except Exception as e:
            app_logger.error(f"Error during model deployment: {str(e)}")
            self.deployment_in_progress = False
            return False
    
    def rollback_deployment(self, version: str = None) -> bool:
        """
        Rollback to a previous deployment.
        
        Args:
            version: Version to rollback to (None for most recent)
            
        Returns:
            True if rollback was successful, False otherwise
        """
        if self.deployment_in_progress:
            app_logger.warning("Deployment in progress, cannot rollback")
            return False
        
        self.deployment_in_progress = True
        
        try:
            # Get available versions
            versions = self._get_archived_versions()
            
            if not versions:
                app_logger.error("No archived versions found for rollback")
                self.deployment_in_progress = False
                return False
            
            # Determine which version to rollback to
            if version is None:
                # Use most recent version
                version = versions[0]
            elif version not in versions:
                app_logger.error(f"Version {version} not found in archived versions")
                self.deployment_in_progress = False
                return False
            
            app_logger.info(f"Rolling back to version {version}")
            
            # Get archive directory for version
            version_dir = os.path.join(self.archive_dir, version)
            
            # Copy models from archive to deployment directory
            for model_file in os.listdir(version_dir):
                src_path = os.path.join(version_dir, model_file)
                dest_path = os.path.join(self.deployment_dir, model_file)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dest_path)
            
            # Restart services if needed
            if self.deployment_target == "local":
                # No need to restart for local deployment
                pass
            elif self.deployment_target == "docker":
                self._restart_docker_services()
            elif self.deployment_target == "kubernetes":
                self._restart_kubernetes_services()
            elif self.deployment_target == "aws":
                self._restart_aws_services()
            
            app_logger.info(f"Rollback to version {version} completed successfully")
            self.deployment_in_progress = False
            return True
        
        except Exception as e:
            app_logger.error(f"Error during rollback: {str(e)}")
            self.deployment_in_progress = False
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """
        Get the current deployment status.
        
        Returns:
            Dictionary with deployment status information
        """
        # Get deployed model versions
        deployed_models = self._get_deployed_model_versions()
        
        # Get available archived versions
        archived_versions = self._get_archived_versions()
        
        # Create status dictionary
        status = {
            "deployment_target": self.deployment_target,
            "auto_deploy": self.auto_deploy,
            "deployment_interval": self.deployment_interval,
            "last_deployment_time": self.last_deployment_time.isoformat(),
            "deployment_in_progress": self.deployment_in_progress,
            "deployed_models": deployed_models,
            "archived_versions": archived_versions
        }
        
        return status
    
    def _auto_deployment_process(self) -> None:
        """Background process for automatic deployment."""
        while self.running:
            try:
                # Check if it's time to deploy
                time_since_last_deployment = datetime.now() - self.last_deployment_time
                if time_since_last_deployment.total_seconds() >= self.deployment_interval * 3600:
                    app_logger.info("Auto-deployment interval reached, deploying models")
                    self.deploy_models()
                
                # Sleep for a while
                time.sleep(3600)  # Check every hour
            
            except Exception as e:
                app_logger.error(f"Error in auto-deployment process: {str(e)}")
                time.sleep(3600)  # Sleep for an hour on error
    
    def _have_models_changed(self, models: List[str]) -> bool:
        """
        Check if models have changed since last deployment.
        
        Args:
            models: List of model names to check
            
        Returns:
            True if models have changed, False otherwise
        """
        # Get deployed model versions
        deployed_versions = self._get_deployed_model_versions()
        
        # Check each model
        for model_name in models:
            # Get model file path
            model_file = f"{model_name}.pt"
            model_path = os.path.join(self.models_dir, model_file)
            
            # Check if model exists
            if not os.path.exists(model_path):
                continue
            
            # Get model modification time
            model_mtime = os.path.getmtime(model_path)
            
            # Check if model has changed
            if model_name not in deployed_versions or model_mtime > deployed_versions[model_name]["timestamp"]:
                return True
        
        return False
    
    def _deploy_local(self, models: List[str]) -> bool:
        """
        Deploy models locally.
        
        Args:
            models: List of model names to deploy
            
        Returns:
            True if deployment was successful, False otherwise
        """
        try:
            # Create version directory in archive
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_dir = os.path.join(self.archive_dir, version)
            os.makedirs(version_dir, exist_ok=True)
            
            # Copy current deployment to archive
            for file_name in os.listdir(self.deployment_dir):
                file_path = os.path.join(self.deployment_dir, file_name)
                if os.path.isfile(file_path) and not file_name.startswith("."):
                    shutil.copy2(file_path, os.path.join(version_dir, file_name))
            
            # Deploy each model
            for model_name in models:
                # Get model file path
                model_file = f"{model_name}.pt"
                model_path = os.path.join(self.models_dir, model_file)
                
                # Check if model exists
                if not os.path.exists(model_path):
                    app_logger.warning(f"Model {model_name} not found, skipping")
                    continue
                
                # Copy model to deployment directory
                shutil.copy2(model_path, os.path.join(self.deployment_dir, model_file))
                
                # Create metadata file
                metadata = {
                    "model_name": model_name,
                    "version": version,
                    "timestamp": os.path.getmtime(model_path),
                    "deployed_at": datetime.now().isoformat()
                }
                
                with open(os.path.join(self.deployment_dir, f"{model_name}_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
            
            return True
        
        except Exception as e:
            app_logger.error(f"Error during local deployment: {str(e)}")
            return False
    
    def _deploy_docker(self, models: List[str]) -> bool:
        """
        Deploy models to Docker containers.
        
        Args:
            models: List of model names to deploy
            
        Returns:
            True if deployment was successful, False otherwise
        """
        try:
            # First deploy locally
            if not self._deploy_local(models):
                return False
            
            # Get Docker configuration
            docker_config = self.deployment_config.get("docker", {})
            container_name = docker_config.get("container_name", "astro-assistance")
            models_volume = docker_config.get("models_volume", "/models")
            
            # Copy models to Docker volume
            cmd = f"docker cp {self.deployment_dir}/. {container_name}:{models_volume}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                app_logger.error(f"Error copying models to Docker container: {result.stderr}")
                return False
            
            # Restart container
            self._restart_docker_services()
            
            return True
        
        except Exception as e:
            app_logger.error(f"Error during Docker deployment: {str(e)}")
            return False
    
    def _deploy_kubernetes(self, models: List[str]) -> bool:
        """
        Deploy models to Kubernetes cluster.
        
        Args:
            models: List of model names to deploy
            
        Returns:
            True if deployment was successful, False otherwise
        """
        try:
            # First deploy locally
            if not self._deploy_local(models):
                return False
            
            # Get Kubernetes configuration
            k8s_config = self.deployment_config.get("kubernetes", {})
            namespace = k8s_config.get("namespace", "default")
            deployment_name = k8s_config.get("deployment_name", "astro-assistance")
            models_volume = k8s_config.get("models_volume", "/models")
            
            # Get pod name
            cmd = f"kubectl get pods -n {namespace} -l app={deployment_name} -o jsonpath='{{.items[0].metadata.name}}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                app_logger.error(f"Error getting Kubernetes pod name: {result.stderr}")
                return False
            
            pod_name = result.stdout.strip()
            
            # Copy models to Kubernetes pod
            cmd = f"kubectl cp {self.deployment_dir}/. {namespace}/{pod_name}:{models_volume}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                app_logger.error(f"Error copying models to Kubernetes pod: {result.stderr}")
                return False
            
            # Restart deployment
            self._restart_kubernetes_services()
            
            return True
        
        except Exception as e:
            app_logger.error(f"Error during Kubernetes deployment: {str(e)}")
            return False
    
    def _deploy_aws(self, models: List[str]) -> bool:
        """
        Deploy models to AWS.
        
        Args:
            models: List of model names to deploy
            
        Returns:
            True if deployment was successful, False otherwise
        """
        try:
            # First deploy locally
            if not self._deploy_local(models):
                return False
            
            # Get AWS configuration
            aws_config = self.deployment_config.get("aws", {})
            s3_bucket = aws_config.get("s3_bucket", "astro-assistance-models")
            s3_prefix = aws_config.get("s3_prefix", "models")
            
            # Upload models to S3
            cmd = f"aws s3 sync {self.deployment_dir} s3://{s3_bucket}/{s3_prefix}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                app_logger.error(f"Error uploading models to S3: {result.stderr}")
                return False
            
            # Restart services
            self._restart_aws_services()
            
            return True
        
        except Exception as e:
            app_logger.error(f"Error during AWS deployment: {str(e)}")
            return False
    
    def _restart_docker_services(self) -> None:
        """Restart Docker services."""
        try:
            # Get Docker configuration
            docker_config = self.deployment_config.get("docker", {})
            container_name = docker_config.get("container_name", "astro-assistance")
            
            # Restart container
            cmd = f"docker restart {container_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                app_logger.error(f"Error restarting Docker container: {result.stderr}")
            else:
                app_logger.info(f"Docker container {container_name} restarted successfully")
        
        except Exception as e:
            app_logger.error(f"Error restarting Docker services: {str(e)}")
    
    def _restart_kubernetes_services(self) -> None:
        """Restart Kubernetes services."""
        try:
            # Get Kubernetes configuration
            k8s_config = self.deployment_config.get("kubernetes", {})
            namespace = k8s_config.get("namespace", "default")
            deployment_name = k8s_config.get("deployment_name", "astro-assistance")
            
            # Restart deployment
            cmd = f"kubectl rollout restart deployment/{deployment_name} -n {namespace}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                app_logger.error(f"Error restarting Kubernetes deployment: {result.stderr}")
            else:
                app_logger.info(f"Kubernetes deployment {deployment_name} restarted successfully")
        
        except Exception as e:
            app_logger.error(f"Error restarting Kubernetes services: {str(e)}")
    
    def _restart_aws_services(self) -> None:
        """Restart AWS services."""
        try:
            # Get AWS configuration
            aws_config = self.deployment_config.get("aws", {})
            lambda_function = aws_config.get("lambda_function", "astro-assistance")
            ecs_cluster = aws_config.get("ecs_cluster")
            ecs_service = aws_config.get("ecs_service")
            
            # Update Lambda function configuration to trigger a restart
            if lambda_function:
                cmd = f"aws lambda update-function-configuration --function-name {lambda_function} --description 'Updated: {datetime.now().isoformat()}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    app_logger.error(f"Error updating Lambda function: {result.stderr}")
                else:
                    app_logger.info(f"Lambda function {lambda_function} updated successfully")
            
            # Restart ECS service
            if ecs_cluster and ecs_service:
                cmd = f"aws ecs update-service --cluster {ecs_cluster} --service {ecs_service} --force-new-deployment"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    app_logger.error(f"Error restarting ECS service: {result.stderr}")
                else:
                    app_logger.info(f"ECS service {ecs_service} restarted successfully")
        
        except Exception as e:
            app_logger.error(f"Error restarting AWS services: {str(e)}")
    
    def _get_deployed_model_versions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get versions of deployed models.
        
        Returns:
            Dictionary mapping model names to version information
        """
        deployed_models = {}
        
        # Check metadata files in deployment directory
        for file_name in os.listdir(self.deployment_dir):
            if file_name.endswith("_metadata.json"):
                file_path = os.path.join(self.deployment_dir, file_name)
                
                try:
                    with open(file_path, "r") as f:
                        metadata = json.load(f)
                    
                    model_name = metadata.get("model_name")
                    if model_name:
                        deployed_models[model_name] = metadata
                
                except Exception as e:
                    app_logger.error(f"Error reading metadata file {file_name}: {str(e)}")
        
        return deployed_models
    
    def _get_archived_versions(self) -> List[str]:
        """
        Get list of archived versions.
        
        Returns:
            List of version strings, sorted by date (newest first)
        """
        versions = []
        
        # Get directories in archive
        for dir_name in os.listdir(self.archive_dir):
            dir_path = os.path.join(self.archive_dir, dir_name)
            if os.path.isdir(dir_path):
                versions.append(dir_name)
        
        # Sort by date (newest first)
        versions.sort(reverse=True)
        
        return versions


# Singleton instance
model_deployer = ModelDeployer()


def deploy_models(models: List[str] = None, force: bool = False) -> bool:
    """
    Deploy models to the target environment.
    
    Args:
        models: List of model names to deploy (None for all)
        force: Whether to force deployment even if no changes are detected
        
    Returns:
        True if deployment was successful, False otherwise
    """
    return model_deployer.deploy_models(models, force)


def rollback_deployment(version: str = None) -> bool:
    """
    Rollback to a previous deployment.
    
    Args:
        version: Version to rollback to (None for most recent)
        
    Returns:
        True if rollback was successful, False otherwise
    """
    return model_deployer.rollback_deployment(version)


def get_deployment_status() -> Dict[str, Any]:
    """
    Get the current deployment status.
    
    Returns:
        Dictionary with deployment status information
    """
    return model_deployer.get_deployment_status()


def start_auto_deployment() -> None:
    """Start automatic deployment process."""
    model_deployer.start_auto_deployment()


def stop_auto_deployment() -> None:
    """Stop automatic deployment process."""
    model_deployer.stop_auto_deployment()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AstroAssistance Model Deployment")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy models")
    deploy_parser.add_argument("--models", nargs="+", help="Models to deploy (default: all)")
    deploy_parser.add_argument("--force", action="store_true", help="Force deployment even if no changes detected")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a previous deployment")
    rollback_parser.add_argument("--version", help="Version to rollback to (default: most recent)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get deployment status")
    
    # Auto-deployment commands
    start_parser = subparsers.add_parser("start-auto", help="Start automatic deployment")
    stop_parser = subparsers.add_parser("stop-auto", help="Stop automatic deployment")
    
    args = parser.parse_args()
    
    if args.command == "deploy":
        success = deploy_models(args.models, args.force)
        print(f"Deployment {'successful' if success else 'failed'}")
    
    elif args.command == "rollback":
        success = rollback_deployment(args.version)
        print(f"Rollback {'successful' if success else 'failed'}")
    
    elif args.command == "status":
        status = get_deployment_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "start-auto":
        start_auto_deployment()
        print("Automatic deployment started")
    
    elif args.command == "stop-auto":
        stop_auto_deployment()
        print("Automatic deployment stopped")
    
    else:
        parser.print_help()