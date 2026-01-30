"""
Configuration loader and validator
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for training and export"""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate(self):
        """Validate configuration parameters"""
        # Add validation logic here
        if 'model' not in self.config:
            raise ValueError("Missing 'model' section in config")
        
        if 'training' not in self.config:
            raise ValueError("Missing 'training' section in config")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation
        
        Args:
            key: Configuration key (e.g., 'training.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.config[key]
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"
