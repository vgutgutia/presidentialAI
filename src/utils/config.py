"""
Configuration loading and management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "project": {
            "name": "marine-debris-detection",
            "seed": 42,
        },
        "device": "auto",
        "data": {
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "normalization": {
                "mean": [0.0582, 0.0556, 0.0480, 0.1011, 0.1257, 0.0902],
                "std": [0.0276, 0.0267, 0.0308, 0.0522, 0.0560, 0.0479],
            },
            "patch_size": 256,
            "overlap": 64,
            "num_workers": 4,
        },
        "model": {
            "backbone": "mit_b2",
            "num_classes": 2,
            "in_channels": 6,
            "pretrained": True,
        },
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "loss": {
                "type": "combined",
                "ce_weight": 0.5,
                "dice_weight": 0.5,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 20,
            },
        },
        "inference": {
            "tile_size": 512,
            "overlap": 128,
            "batch_size": 4,
            "confidence_threshold": 0.5,
            "min_area_pixels": 100,
        },
        "outputs": {
            "models_dir": "outputs/models",
            "predictions_dir": "outputs/predictions",
            "logs_dir": "outputs/logs",
        },
    }


class Config:
    """
    Configuration wrapper class with dot notation access.
    
    Usage:
        config = Config.from_yaml('config.yaml')
        lr = config.training.learning_rate
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config_dict = load_config(path)
        return cls(config_dict)
    
    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        return cls(get_default_config())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return getattr(self, key, default)
