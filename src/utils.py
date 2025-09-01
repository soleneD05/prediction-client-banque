import yaml

def load_params(params_path: str = "params.yaml") -> dict:
    """
    Load parameters from the params.yaml file.

    Args:
        params_path (str): Path to the params.yaml file (default is "params.yaml")

    Returns:
        dict: Dictionary containing the parameters
    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params
