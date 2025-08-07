"""
Configuration loader for AstroAssistance.
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigManager:
    """Manages configuration for the AstroAssistance system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = os.path.join(project_root, "config", "config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated path to the configuration value.
            default: Default value to return if key is not found.
            
        Returns:
            The configuration value or default.
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self.config
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value.
            value: New value to set.
        """
        keys = key.split(".")
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save the current configuration to the YAML file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Create a singleton instance
config_manager = ConfigManager()