"""
Configuration loader for Spotify Recommendation System.
Loads settings from config.yaml and provides utility functions.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the recommendation system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Get project root (parent of spotify_recommender)
            current_dir = Path(__file__).parent
            config_path = current_dir / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent.parent
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., 'paths.raw_data_dir' or 'model.lightgbm.learning_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str, create_dir: bool = False) -> Path:
        """
        Get absolute path from configuration.
        
        Args:
            key: Path key in config (e.g., 'paths.raw_data_dir')
            create_dir: If True, create directory if it doesn't exist
            
        Returns:
            Absolute Path object
        """
        relative_path = self.get(key)
        if relative_path is None:
            raise ValueError(f"Path key '{key}' not found in config")
        
        # Convert to absolute path relative to project root
        abs_path = self.project_root / relative_path
        
        # Create directory if requested and path is a directory
        if create_dir and not abs_path.suffix:  # No file extension = directory
            abs_path.mkdir(parents=True, exist_ok=True)
        
        return abs_path
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get all paths from configuration as absolute paths."""
        paths = {}
        for key, value in self.config.get('paths', {}).items():
            paths[key] = self.project_root / value
        return paths
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = None):
    """Reload configuration from file."""
    global _config
    _config = Config(config_path)
    return _config


# Convenience functions
def get_project_root() -> Path:
    """Get project root directory."""
    return get_config().project_root


def get_data_dir(subdir: str = None) -> Path:
    """
    Get data directory path.
    
    Args:
        subdir: Subdirectory name ('raw', 'processed', etc.)
        
    Returns:
        Absolute path to data directory
    """
    if subdir:
        return get_config().get_path(f'paths.{subdir}_data_dir', create_dir=True)
    return get_config().project_root / 'data'


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"Project Root: {config.project_root}")
    print(f"\nAll Paths:")
    for key, path in config.get_all_paths().items():
        print(f"  {key}: {path}")
    
    print(f"\nModel Config:")
    print(f"  LightGBM learning_rate: {config.get('model.lightgbm.learning_rate')}")
    print(f"  Top K recommendations: {config.get('recommendation.top_k')}")

