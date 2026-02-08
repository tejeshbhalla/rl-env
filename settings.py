"""Simple settings loader."""
import json
from pathlib import Path


class Settings:
    """Simple settings loader from JSON file."""
    
    def __init__(self, settings_dict):
        """Initialize settings from a dictionary."""
        for key, value in settings_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Settings(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def load(cls, file_path="settings.json"):
        """Load settings from JSON file."""
        path = Path(file_path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data)


# Load default settings
settings = Settings.load()


if __name__ == "__main__":
    print("Loaded settings:")
    print(f"  Model: {settings.model_info.model_name}")
    print(f"  API Key: {settings.model_info.api_key[:20]}...")
