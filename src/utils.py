"""
Utility functions for the ML pipeline
"""

import yaml
import os

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from YAML configuration file"""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    try:
        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)
        return params
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {params_path}: {e}")

def create_directories(params: dict):
    """Create necessary directories based on file paths in parameters"""
    directories_to_create = set()
    
    def extract_dirs(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and ('_path' in key or key == 'path'):
                    dir_path = os.path.dirname(value)
                    if dir_path and dir_path != '.':
                        directories_to_create.add(dir_path)
                elif isinstance(value, (dict, list)):
                    extract_dirs(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    extract_dirs(item)
    
    extract_dirs(params)
    
    for directory in directories_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def validate_params(params: dict):
    """Validate configuration parameters"""
    required_sections = ['data', 'preprocessing', 'model_selection', 'evaluation']
    
    for section in required_sections:
        if section not in params:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate model selection config
    if not params['model_selection']['models']:
        raise ValueError("No models configured for selection")
    
    valid_metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    if params['model_selection']['scoring_metric'] not in valid_metrics:
        raise ValueError(f"Invalid scoring metric. Must be one of: {valid_metrics}")
    
    print("✅ Configuration validation passed")

if __name__ == "__main__":
    try:
        params = load_params()
        validate_params(params)
        create_directories(params)
        print("All utility functions working correctly!")
    except Exception as e:
        print(f"Error: {e}")